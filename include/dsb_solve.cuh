// dsb_solve.cuh -- one dSB request solved by one CTA, entirely on chip.
//
// This is the service form of the two fastest exact paths of dsb-gpu:
//
//   * bit   (PM1 / TERNARY): couplings as bit planes, the coupling sum
//            sum_c J_rc s_c evaluated with XOR/AND + POPC, exact;
//   * int8  (INT8):          int8 couplings, int8 +-1 signs, DP4A with int32
//            accumulation, exact for |J| <= 127.
//
// In dsb-gpu both are whole-GPU kernels for one large instance (grid.sync or
// cuBLAS per step).  A service gets many SMALL instances instead, so here one
// request = one CTA: J, the sign rows of all B replicas and the field h sit in
// shared memory, the state (x, y) sits in registers, and the per-step barrier
// is one __syncthreads().  Nothing leaves the SM between the first and the
// last step; independent requests run on independent CTAs.
//
// Thread mapping: pair p = b * npad + r (replica b, row r), thread t owns
// pairs t, t + T, t + 2T, ...  Because npad and T are multiples of 32, a warp
// always holds 32 consecutive rows of ONE replica, so the new signs of a warp
// are exactly one 32-bit word of that replica's sign row: __ballot_sync
// produces it and lane 0 stores it -- no atomics, no shuffles.
//
// Floating point: the update uses __f*_rn intrinsics (no FMA contraction), so
// the trajectory equals the CPU reference in dsb_ref.hpp bit for bit.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "dsb_model.h"
#include "dsb_proto.h"
#include "dsb_queue.h"

namespace dsbgpu {

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// The exit flag lives in host-mapped memory: every read of it is a PCIe
// transaction.  Hundreds of spinning threads reading it on every loop
// iteration saturate the link (and starve the NIC's DMA / doorbells), so a
// spin loop asks for it at most once per DSB_EXIT_POLL_NS -- the shutdown
// latency, which is all the flag is for.
#define DSB_EXIT_POLL_NS 500000ull
__device__ __forceinline__ bool should_exit(const dsb_ctx& ctx, unsigned long long& next_check) {
  const unsigned long long now = gtimer();
  if (now < next_check) return false;
  next_check = now + DSB_EXIT_POLL_NS;
  return *ctx.exit_flag != 0u;
}

__device__ __forceinline__ void advance(float& x, float& y, float acc, float p,
                                        float xi, float dt) {
  // delta = 1:  y += (-(1 - p) x + xi acc) dt ;  x += dt y
  const float a  = __fsub_rn(1.0f, p);
  const float t1 = __fmul_rn(a, x);
  const float t2 = __fmul_rn(xi, acc);
  const float f  = __fsub_rn(t2, t1);
  y = __fadd_rn(y, __fmul_rn(f, dt));
  const float v = __fmul_rn(dt, y);
  x = __fadd_rn(x, v);
  if (fabsf(x) > 1.0f) {
    x = (x > 0.f) ? 1.0f : -1.0f;
    y = 0.0f;
  }
}

// ---------------------------------------------------------------------------
// coupling sums, sum_{c != r} J_rc s_c (without h)

template <int FMT>
struct Coupling;

template <>
struct Coupling<DSB_FMT_PM1> {
  // n - 2 popc(J_r ^ S) + s_r : padding columns cancel, the cleared diagonal
  // bit contributed -s_r and is added back.
  __device__ __forceinline__ static int sum(const uint8_t* sm, const dsb_smem_plan& P,
                                            const uint32_t* S, int b, int r) {
    const uint32_t* J  = reinterpret_cast<const uint32_t*>(sm + P.off_j);
    const uint4*    ja = reinterpret_cast<const uint4*>(J + r * P.wpad);
    const uint32_t* sr = S + b * P.wpad;
    const uint4*    sa = reinterpret_cast<const uint4*>(sr);
    const int       nq = (int(P.words) + 3) >> 2;
    int pop = 0;
    for (int q = 0; q < nq; ++q) {
      const uint4 j = ja[q], s = sa[q];
      pop += __popc(j.x ^ s.x) + __popc(j.y ^ s.y) + __popc(j.z ^ s.z) + __popc(j.w ^ s.w);
    }
    const int s_r = ((sr[r >> 5] >> (r & 31)) & 1u) ? 1 : -1;
    return int(P.n) - 2 * pop + s_r;
  }
};

template <>
struct Coupling<DSB_FMT_TERNARY> {
  // 2 (popc(P&S) - popc(M&S)) - (|P| - |M|)
  __device__ __forceinline__ static int sum(const uint8_t* sm, const dsb_smem_plan& P,
                                            const uint32_t* S, int b, int r) {
    const uint4* pa = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint32_t*>(sm + P.off_j) + r * P.wpad);
    const uint4* ma = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint32_t*>(sm + P.off_m) + r * P.wpad);
    const uint4* sa = reinterpret_cast<const uint4*>(S + b * P.wpad);
    const int    nq = (int(P.words) + 3) >> 2;
    int d = 0;
    for (int q = 0; q < nq; ++q) {
      const uint4 p = pa[q], m = ma[q], s = sa[q];
      d += (__popc(p.x & s.x) - __popc(m.x & s.x)) + (__popc(p.y & s.y) - __popc(m.y & s.y)) +
           (__popc(p.z & s.z) - __popc(m.z & s.z)) + (__popc(p.w & s.w) - __popc(m.w & s.w));
    }
    const int32_t* rc = reinterpret_cast<const int32_t*>(sm + P.off_rc);
    return 2 * d - rc[r];
  }
};

template <>
struct Coupling<DSB_FMT_INT8> {
  // DP4A over the int8 row; padding columns and the diagonal are 0 in J.
  __device__ __forceinline__ static int sum(const uint8_t* sm, const dsb_smem_plan& P,
                                            const uint32_t* S, int b, int r) {
    const uint4* ja = reinterpret_cast<const uint4*>(sm + P.off_j + r * P.lds);
    const uint4* sa = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint8_t*>(S) + b * P.lds);
    const int nq = int(P.npad) >> 4;
    int acc = 0;
    for (int q = 0; q < nq; ++q) {
      const uint4 j = ja[q], s = sa[q];
      acc = __dp4a(int(j.x), int(s.x), acc);
      acc = __dp4a(int(j.y), int(s.y), acc);
      acc = __dp4a(int(j.z), int(s.z), acc);
      acc = __dp4a(int(j.w), int(s.w), acc);
    }
    return acc;
  }
};

// ---------------------------------------------------------------------------
// sign publication

template <int FMT>
__device__ __forceinline__ void publish(uint8_t* Sbuf, const dsb_smem_plan& P, int b, int r,
                                        bool live, float x) {
  if (FMT == DSB_FMT_INT8) {
    Sbuf[b * P.lds + r] = live ? (x > 0.f ? uint8_t(1) : uint8_t(0xFF)) : uint8_t(0);
  } else {
    const unsigned bits = __ballot_sync(0xFFFFFFFFu, live && x > 0.f);
    if ((threadIdx.x & 31u) == 0u)
      reinterpret_cast<uint32_t*>(Sbuf)[b * P.wpad + (r >> 5)] = bits;
  }
}

// ---------------------------------------------------------------------------
// load one request into shared memory and integrate it

struct Scalars {
  dsb_smem_plan plan;
  int      status;
  int      best;
  int      cur;
  uint32_t model_id;
  float    xi;
  float    dt;
  int      steps;
  uint32_t seed;
  int      has_h;
  unsigned long long t_end;
};

template <int FMT>
__device__ void integrate(const dsb_ctx& ctx, uint8_t* sm, Scalars& sc, const uint8_t* frame) {
  const dsb_smem_plan& P = sc.plan;
  const int tid = int(threadIdx.x);
  const int n = int(P.n), npad = int(P.npad), B = int(P.replicas);
  const int pairs = B * npad;

  // ---- coupling rows, h, zeroed sign buffers and energies -----------------
  const uint8_t* model = ctx.model_pool + size_t(sc.model_id) * DSB_MODEL_SLOT_BYTES;
  if (FMT == DSB_FMT_INT8) {
    const int ldw = int(dsb_int8_ld(P.n)) >> 2, ldsw = int(P.lds) >> 2;
    const int32_t* src = reinterpret_cast<const int32_t*>(model);
    int32_t* dst = reinterpret_cast<int32_t*>(sm + P.off_j);
    for (int i = tid; i < npad * ldsw; i += DSB_THREADS) {
      const int r = i / ldsw, w = i - r * ldsw;
      dst[i] = (r < n && w < ldw) ? __ldcg(src + r * ldw + w) : 0;
    }
  } else {
    const int W = int(P.words), Wp = int(P.wpad);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(model);
    uint32_t* dj = reinterpret_cast<uint32_t*>(sm + P.off_j);
    uint32_t* dm = reinterpret_cast<uint32_t*>(sm + P.off_m);
    for (int i = tid; i < npad * Wp; i += DSB_THREADS) {
      const int r = i / Wp, w = i - r * Wp;
      const bool live = (r < n) && (w < W);
      dj[i] = live ? __ldcg(src + r * W + w) : 0u;
      if (FMT == DSB_FMT_TERNARY) dm[i] = live ? __ldcg(src + n * W + r * W + w) : 0u;
    }
    if (FMT == DSB_FMT_TERNARY) {
      const int32_t* rcs = reinterpret_cast<const int32_t*>(src + 2 * n * W);
      int32_t* rcd = reinterpret_cast<int32_t*>(sm + P.off_rc);
      for (int r = tid; r < npad; r += DSB_THREADS) rcd[r] = (r < n) ? __ldcg(rcs + r) : 0;
    }
  }
  {
    int32_t* h = reinterpret_cast<int32_t*>(sm + P.off_h);
    const short* hs = reinterpret_cast<const short*>(frame + DSB_HDR_BYTES);
    for (int r = tid; r < npad; r += DSB_THREADS) h[r] = (sc.has_h && r < n) ? int(__ldcg(hs + r)) : 0;
    uint4* s4 = reinterpret_cast<uint4*>(sm + P.off_s);
    for (int i = tid; i < int(2 * P.s_buf / 16); i += DSB_THREADS) s4[i] = make_uint4(0, 0, 0, 0);
    unsigned long long* e2 = reinterpret_cast<unsigned long long*>(sm + P.off_e);
    for (int b = tid; b < B; b += DSB_THREADS) e2[b] = 0ull;
  }
  __syncthreads();

  const int32_t* h = reinterpret_cast<const int32_t*>(sm + P.off_h);
  uint8_t* S0 = sm + P.off_s;

  // ---- state in registers --------------------------------------------------
  float    x[DSB_PT], y[DSB_PT];
  uint32_t rb[DSB_PT];
#pragma unroll
  for (int k = 0; k < DSB_PT; ++k) {
    const int p = tid + k * DSB_THREADS;
    x[k] = 0.f;
    y[k] = 0.f;
    rb[k] = 0u;
    if (p < pairs) {
      const int b = p / npad, r = p - b * npad;
      rb[k] = (uint32_t(b) << 16) | uint32_t(r);
      if (r < n) {
        x[k] = __fsub_rn(__fmul_rn(dsb_u24(sc.seed, b, r, 0), 0.02f), 0.01f);
        y[k] = __fsub_rn(__fmul_rn(dsb_u24(sc.seed, b, r, 1), 0.02f), 0.01f);
      }
      publish<FMT>(S0, P, b, r, r < n, x[k]);
    }
  }
  __syncthreads();

  // ---- K steps, one barrier each (double-buffered signs) -------------------
  const float xi = sc.xi, dt = sc.dt;
  const int K = sc.steps;
  const float km1 = float(K - 1);
  int cur = 0;
  for (int step = 0; step < K; ++step) {
    const float pf = __fdiv_rn(float(step), km1);
    const uint32_t* Sc = reinterpret_cast<const uint32_t*>(S0 + cur * P.s_buf);
    uint8_t* Sn = S0 + (cur ^ 1) * P.s_buf;
#pragma unroll
    for (int k = 0; k < DSB_PT; ++k) {
      const int p = tid + k * DSB_THREADS;
      if (p < pairs) {  // warp-uniform: pairs is a multiple of 32
        const int b = int(rb[k] >> 16), r = int(rb[k] & 0xFFFFu);
        const bool live = r < n;
        if (live) {
          const int acc = Coupling<FMT>::sum(sm, P, Sc, b, r) + h[r];
          advance(x[k], y[k], float(acc), pf, xi, dt);
        }
        publish<FMT>(Sn, P, b, r, live, x[k]);
      }
    }
    __syncthreads();
    cur ^= 1;
  }

  // ---- energies of the final signs: 2E = sum_r s_r (-(Js)_r - 2 h_r) -------
  {
    const uint32_t* Sc = reinterpret_cast<const uint32_t*>(S0 + cur * P.s_buf);
    unsigned long long* e2 = reinterpret_cast<unsigned long long*>(sm + P.off_e);
#pragma unroll
    for (int k = 0; k < DSB_PT; ++k) {
      const int p = tid + k * DSB_THREADS;
      if (p < pairs) {
        const int b = int(rb[k] >> 16), r = int(rb[k] & 0xFFFFu);
        int c = 0;
        if (r < n) {
          const int s = x[k] > 0.f ? 1 : -1;
          c = -s * Coupling<FMT>::sum(sm, P, Sc, b, r) - 2 * s * h[r];
        }
        c = __reduce_add_sync(0xFFFFFFFFu, c);
        if ((tid & 31) == 0) atomicAdd(&e2[b], (unsigned long long)(long long)c);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    const long long* e2 = reinterpret_cast<const long long*>(sm + P.off_e);
    int best = 0;
    for (int b = 1; b < B; ++b)
      if (e2[b] < e2[best]) best = b;
    sc.best = best;
    sc.cur = cur;
  }
  __syncthreads();
}

// Copies the reply spins (bit words) of replica sc.best into dst.
template <int FMT>
__device__ void best_spins(const uint8_t* sm, const Scalars& sc, uint32_t* dst) {
  const dsb_smem_plan& P = sc.plan;
  const uint8_t* S = sm + P.off_s + sc.cur * P.s_buf;
  for (int w = int(threadIdx.x); w < int(P.words); w += DSB_THREADS) {
    uint32_t word = 0u;
    if (FMT == DSB_FMT_INT8) {
      const int8_t* row = reinterpret_cast<const int8_t*>(S) + sc.best * P.lds;
      for (int i = 0; i < 32; ++i) {
        const int r = w * 32 + i;
        if (r < int(P.n) && row[r] > 0) word |= 1u << i;
      }
    } else {
      word = reinterpret_cast<const uint32_t*>(S)[sc.best * P.wpad + w];
    }
    dst[w] = word;
  }
}

// ---------------------------------------------------------------------------
// one work item -> one reply frame in the staging area; returns its length.
// All threads of the CTA call it.

struct ItemView {
  const uint8_t*     frame;
  unsigned long long t_rx;
  uint32_t           bytes;
  uint32_t           kind;
};

__device__ inline void reply_common(uint8_t* out, const uint8_t* req, uint8_t type) {
  dsb_hdr_common* o = reinterpret_cast<dsb_hdr_common*>(out);
  const dsb_hdr_common* q = reinterpret_cast<const dsb_hdr_common*>(req);
  for (int i = 0; i < 6; ++i) {
    o->d_mac[i] = q->s_mac[i];
    o->s_mac[i] = q->d_mac[i];
  }
  o->ethertype = q->ethertype;
  o->pad0[0] = o->pad0[1] = 0;
  o->version = DSB_PROTO_VERSION;
  o->type = type;
  o->flags = q->flags;  // echoed
  o->req_id = q->req_id;
  o->client_ts = q->client_ts;
}

__device__ uint32_t serve_item(const dsb_ctx& ctx, const ItemView& it, uint8_t* sm,
                               uint32_t worker, unsigned long long t_start) {
  __shared__ Scalars sc;
  const int tid = int(threadIdx.x);
  const uint32_t limit = ctx.smem_bytes - DSB_SMEM_RESERVE;
  uint8_t* hdr = sm + limit;                        // 64 B request header copy
  uint8_t* stage = sm + ctx.smem_bytes - 512u;      // reply frame

  if (tid < 4) {
    reinterpret_cast<uint4*>(hdr)[tid] = __ldcg(reinterpret_cast<const uint4*>(it.frame) + tid);
  }
  __syncthreads();

  if (it.kind == DSB_WORK_ACK) {
    if (tid == 0) {
      const dsb_load_frag* f = reinterpret_cast<const dsb_load_frag*>(hdr);
      dsb_load_ack* a = reinterpret_cast<dsb_load_ack*>(stage);
      reply_common(stage, hdr, DSB_MSG_LOAD_ACK);
      const uint32_t id = f->model_id;
      a->model_id = f->model_id;
      a->n = f->n;
      a->status = DSB_OK;
      if (id >= DSB_MAX_MODELS) {
        a->status = DSB_ERR_LOAD_RANGE;
      } else {
        const volatile dsb_model_desc* m = ctx.models + id;
        a->status = (m->ready_gen == f->generation) ? DSB_OK : DSB_ERR_LOAD_RANGE;
      }
      a->reserved0 = 0;
      a->generation = f->generation;
      a->total_bytes = f->total_bytes;
      for (int i = 0; i < 16; ++i) a->reserved1[i] = 0;
    }
    __syncthreads();
    return DSB_HDR_BYTES;
  }

  // ---- SOLVE: validate --------------------------------------------------------
  if (tid == 0) {
    const dsb_solve_req* q = reinterpret_cast<const dsb_solve_req*>(hdr);
    int st = DSB_OK;
    sc.model_id = q->model_id;
    if (q->model_id >= DSB_MAX_MODELS) {
      st = DSB_ERR_NO_MODEL;
    } else {
      const volatile dsb_model_desc* m = ctx.models + q->model_id;
      const uint32_t ready = m->ready_gen;
      if (ready == 0u) {
        st = DSB_ERR_NO_MODEL;
      } else {
        __threadfence();
        const uint32_t n = m->n, fmt = m->format;
        const uint32_t need = dsb_solve_frame_bytes(q->n, q->h_format);
        if (q->n != n || q->steps < 2 || q->h_format > 1 || it.bytes < need)
          st = DSB_ERR_SHAPE;
        else if (dsb_make_plan(n, q->replicas, fmt, limit, &sc.plan) != 0)
          st = DSB_ERR_SHAPE;
        sc.xi = (q->xi > 0.f) ? q->xi : m->xi_default;
      }
    }
    sc.status = st;
    sc.dt = q->dt;
    sc.steps = q->steps;
    sc.seed = q->seed;
    sc.has_h = q->h_format;
    sc.best = 0;
    sc.cur = 0;
  }
  __syncthreads();

  if (sc.status == DSB_OK) {
    switch (sc.plan.format) {
      case DSB_FMT_PM1:     integrate<DSB_FMT_PM1>(ctx, sm, sc, it.frame); break;
      case DSB_FMT_TERNARY: integrate<DSB_FMT_TERNARY>(ctx, sm, sc, it.frame); break;
      default:              integrate<DSB_FMT_INT8>(ctx, sm, sc, it.frame); break;
    }
    uint32_t* spins = reinterpret_cast<uint32_t*>(stage + DSB_HDR_BYTES);
    switch (sc.plan.format) {
      case DSB_FMT_PM1:     best_spins<DSB_FMT_PM1>(sm, sc, spins); break;
      case DSB_FMT_TERNARY: best_spins<DSB_FMT_TERNARY>(sm, sc, spins); break;
      default:              best_spins<DSB_FMT_INT8>(sm, sc, spins); break;
    }
  }

  if (tid == 0) {
    const unsigned long long t_end = gtimer();
    sc.t_end = t_end;
    const dsb_solve_req* q = reinterpret_cast<const dsb_solve_req*>(hdr);
    dsb_result* o = reinterpret_cast<dsb_result*>(stage);
    reply_common(stage, hdr, DSB_MSG_RESULT);
    o->model_id = q->model_id;
    o->n = q->n;
    o->status = uint16_t(sc.status);
    o->best_replica = uint16_t(sc.best);
    o->energy_x2 = (sc.status == DSB_OK)
                       ? reinterpret_cast<const long long*>(sm + sc.plan.off_e)[sc.best]
                       : 0ll;
    o->queue_ns = (it.t_rx != 0ull && t_start > it.t_rx) ? uint32_t(t_start - it.t_rx) : 0u;
    o->solve_ns = uint32_t(t_end - t_start);
    o->tx_ns = 0u;
    o->worker = worker;
  }
  __syncthreads();
  return (sc.status == DSB_OK) ? dsb_result_frame_bytes(sc.plan.n) : uint32_t(DSB_HDR_BYTES);
}

// ---------------------------------------------------------------------------
// persistent worker: take tickets from the work ring forever

__device__ void worker_loop(const dsb_ctx& ctx, uint32_t worker) {
  extern __shared__ __align__(16) uint8_t dsb_smem[];
  __shared__ ItemView item;
  __shared__ int quit;
  __shared__ unsigned long long t_start, rseq, ticket;
  __shared__ uint32_t len;
  const int tid = int(threadIdx.x);
  unsigned long long next_check = 0ull;  // thread 0 only: throttled exit-flag poll

  for (;;) {
    if (tid == 0) {
      const unsigned long long t = atomicAdd(ctx.work_claim, 1ull);
      volatile dsb_work_item* w = ctx.items + (t & (DSB_WORK_CAP - 1u));
      quit = 0;
      ticket = t;
      while (w->seq != t + 1ull) {
        if (should_exit(ctx, next_check)) {
          quit = 1;
          break;
        }
      }
      if (!quit) {
        __threadfence();
        item.frame = (const uint8_t*)w->frame;
        item.t_rx = w->t_rx;
        item.bytes = w->bytes;
        item.kind = w->kind;
        t_start = gtimer();
      }
    }
    __syncthreads();
    if (quit) break;

    const uint32_t l = serve_item(ctx, item, dsb_smem, worker, t_start);

    if (tid == 0) {
      len = l;
      const unsigned long long r = atomicAdd(ctx.rsp_head, 1ull);
      const volatile unsigned long long* rel = ctx.rsp_released;
      while (r - *rel >= DSB_RSP_CAP) {
        if (should_exit(ctx, next_check)) {
          quit = 1;
          break;
        }
      }
      rseq = r;
    }
    __syncthreads();
    if (quit) break;

    const uint32_t slot = uint32_t(rseq & (DSB_RSP_CAP - 1u));
    {
      const uint4* src = reinterpret_cast<const uint4*>(dsb_smem + ctx.smem_bytes - 512u);
      uint4* dst = reinterpret_cast<uint4*>(ctx.rsp_buf + size_t(slot) * DSB_RSP_SLOT);
      for (int i = tid; i < int((len + 15u) >> 4); i += DSB_THREADS) dst[i] = src[i];
      // every writer orders its own stores before the ready word below
      if (ctx.rsp_on_host)
        __threadfence_system();
      else
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
      ctx.rsp_tend[slot] = gtimer();
      // ready word carries the length in its top 16 bits: seq + 1 | len << 48
      const unsigned long long word = (rseq + 1ull) | ((unsigned long long)len << 48);
      if (ctx.rsp_on_host)
        __threadfence_system();
      else
        __threadfence();
      *(volatile unsigned long long*)(ctx.rsp_ready + slot) = word;
      // the request (item and frame) is no longer needed: release the work slot
      volatile dsb_work_item* w = ctx.items + (ticket & (DSB_WORK_CAP - 1u));
      w->ack = ticket + 1ull;
      atomicAdd(ctx.work_done, 1ull);
      if (item.kind == DSB_WORK_SOLVE) {
        const uint16_t st = reinterpret_cast<const dsb_result*>(dsb_smem + ctx.smem_bytes - 512u)->status;
        atomicAdd(st == DSB_OK ? &ctx.stats->solved : &ctx.stats->solve_errors, 1ull);
      }
    }
  }
}

// ready-word helpers for consumers (host or device)
__host__ __device__ inline unsigned long long dsb_ready_seq(unsigned long long w) {
  return w & 0x0000FFFFFFFFFFFFull;
}
__host__ __device__ inline uint32_t dsb_ready_len(unsigned long long w) {
  return uint32_t(w >> 48);
}

// Kernels usable by every server.
__global__ void __launch_bounds__(DSB_THREADS) worker_kernel(dsb_ctx ctx) {
  worker_loop(ctx, blockIdx.x);
}

// Launch-per-request baseline: one frame in, one reply out.
__global__ void __launch_bounds__(DSB_THREADS)
    solve_once_kernel(dsb_ctx ctx, const uint8_t* frame, uint32_t bytes, uint8_t* out,
                      uint32_t* out_len) {
  extern __shared__ __align__(16) uint8_t dsb_smem[];
  ItemView it;
  it.frame = frame;
  it.t_rx = 0ull;
  it.bytes = bytes;
  it.kind = DSB_WORK_SOLVE;
  const unsigned long long t0 = gtimer();
  const uint32_t l = serve_item(ctx, it, dsb_smem, 0u, t0);
  const uint4* src = reinterpret_cast<const uint4*>(dsb_smem + ctx.smem_bytes - 512u);
  for (int i = int(threadIdx.x); i < int((l + 15u) >> 4); i += DSB_THREADS)
    reinterpret_cast<uint4*>(out)[i] = src[i];
  if (threadIdx.x == 0) *out_len = l;
}

}  // namespace dsbgpu
