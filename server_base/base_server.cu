// base_server.cu -- the conventional CPU+GPU architectures, same wire format,
// same solver code as the DOCA server.
//
//   sudo ./dsb_base_server --iface ens7np0 --mode launch [--streams 4]
//   sudo ./dsb_base_server --iface ens7np0 --mode proxy  [--workers 64]
//
// launch  (CPU-driven, the textbook path)
//     kernel socket recv -> pinned copy -> cudaMemcpyAsync H2D -> kernel launch
//     (one CTA) -> cudaMemcpyAsync D2H -> event -> sendto.  Up to --streams
//     requests are in flight on separate CUDA streams.
//
// proxy   (CPU-proxied persistent kernel)
//     kernel socket recv -> copy into host-mapped frame pool -> work item in a
//     host-mapped ring -> persistent worker CTAs (the same worker_loop the
//     DOCA server runs) -> reply in a host-mapped ring -> CPU polls -> sendto.
//     This removes launches and copies but keeps the CPU on the data path.
//
// Both run the network loop on one busy-polling CPU thread (pin it with
// taskset).  LOAD fragments are reassembled on the CPU and copied to the GPU.
#include <sched.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dsb_host.hpp"
#include "gpu_ctx.hpp"
#include "rawsock.hpp"

using namespace dsbgpu;

static volatile sig_atomic_t g_quit = 0;
static void on_signal(int) { g_quit = 1; }

static uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

struct Opt {
  std::string iface = "ens7np0", mode = "launch";
  int device = 0, streams = 4, workers = 0;
  uint32_t smem_kb = 0;
};

// ---- model reassembly (CPU) ----------------------------------------------------
struct PendingModel {
  uint32_t generation = 0, total = 0, frag_count = 0, seen = 0;
  uint16_t n = 0;
  uint8_t format = 0;
  float xi = 0.f;
  bool done = false;
  std::vector<uint8_t> data;
  std::vector<bool> have;
};

static void send_ack(RawSock& sock, const uint8_t* f) {
  const dsb_load_frag* h = reinterpret_cast<const dsb_load_frag*>(f);
  uint8_t ack[DSB_HDR_BYTES] = {0};
  const dsb_hdr_common* q = reinterpret_cast<const dsb_hdr_common*>(f);
  dsbhost::Mac dst, src;
  std::memcpy(dst.b, q->s_mac, 6);
  std::memcpy(src.b, q->d_mac, 6);
  dsbhost::fill_common(ack, dst, src, DSB_MSG_LOAD_ACK, q->req_id, q->client_ts);
  dsb_load_ack* a = reinterpret_cast<dsb_load_ack*>(ack);
  a->model_id = h->model_id;
  a->n = h->n;
  a->status = DSB_OK;
  a->generation = h->generation;
  a->total_bytes = h->total_bytes;
  sock.send(ack, sizeof(ack));
}

// Same acceptance rules as the GPU receiver: fragments are tracked by index,
// duplicates are ignored, and a duplicate of a completed generation re-ACKs
// (the client retries when an ACK is lost).
static void handle_load(GpuCtx& g, RawSock& sock, std::vector<PendingModel>& pm, const uint8_t* f,
                        size_t len) {
  const dsb_load_frag* h = reinterpret_cast<const dsb_load_frag*>(f);
  const uint32_t need = dsb_model_bytes(h->n, h->format);
  if (h->model_id >= DSB_MAX_MODELS || h->generation == 0 || need == 0 ||
      h->total_bytes > DSB_MODEL_SLOT_BYTES || h->total_bytes < need || h->frag_count == 0 ||
      h->frag_count > DSB_MAX_FRAGS || h->frag_idx >= h->frag_count || (h->frag_bytes & 3u) ||
      h->offset + h->frag_bytes > ((h->total_bytes + 3u) & ~3u) || DSB_HDR_BYTES + h->frag_bytes > len)
    return;
  PendingModel& m = pm[h->model_id];
  if (m.generation != h->generation) {
    m = PendingModel();
    m.generation = h->generation;
    m.total = h->total_bytes;
    m.frag_count = h->frag_count;
    m.n = h->n;
    m.format = h->format;
    m.xi = h->xi_default;
    m.data.assign((m.total + 3u) & ~3u, 0);
    m.have.assign(m.frag_count, false);
  }
  if (h->total_bytes != m.total || h->frag_count != m.frag_count || h->n != m.n || h->format != m.format)
    return;
  if (m.have[h->frag_idx]) {
    if (m.done) send_ack(sock, f);
    return;
  }
  std::memcpy(m.data.data() + h->offset, f + DSB_HDR_BYTES, h->frag_bytes);
  m.have[h->frag_idx] = true;
  if (++m.seen < m.frag_count) return;
  m.done = true;
  std::vector<uint8_t> packed(m.data.begin(), m.data.begin() + m.total);
  g.upload_model(h->model_id, m.n, m.format, m.xi, packed, m.generation);
  send_ack(sock, f);
  std::printf("model %u loaded: n=%u format=%u %u bytes\n", h->model_id, m.n, m.format, m.total);
}

// ---- launch mode ----------------------------------------------------------------
struct Slot {
  cudaStream_t s;
  cudaEvent_t done;
  uint8_t *h_in, *h_out, *d_in, *d_out;
  uint32_t* d_len;
  uint32_t* h_len;
  bool busy = false;
  uint64_t t_in = 0;
};

static int run_launch(const Opt& o, GpuCtx& g, RawSock& sock) {
  CK(cudaFuncSetAttribute(solve_once_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          int(g.ctx.smem_bytes)));
  std::vector<Slot> slots(o.streams);
  for (Slot& s : slots) {
    CK(cudaStreamCreateWithFlags(&s.s, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    CK(cudaMallocHost(&s.h_in, DSB_MAX_FRAME));
    CK(cudaMallocHost(&s.h_out, DSB_RSP_SLOT));
    CK(cudaMallocHost(&s.h_len, 4));
    CK(cudaMalloc(&s.d_in, DSB_MAX_FRAME));
    CK(cudaMalloc(&s.d_out, DSB_RSP_SLOT));
    CK(cudaMalloc(&s.d_len, 4));
  }
  std::vector<PendingModel> pm(DSB_MAX_MODELS);
  std::vector<uint8_t> buf(DSB_MAX_FRAME + 64);
  unsigned long long served = 0;
  uint64_t last = now_ns();
  while (!g_quit) {
    // completions first
    for (Slot& s : slots) {
      if (!s.busy || cudaEventQuery(s.done) != cudaSuccess) continue;
      const uint32_t len = *s.h_len;
      dsb_result* r = reinterpret_cast<dsb_result*>(s.h_out);
      // baseline convention: queue_ns = 0 and tx_ns = CPU residence - solve_ns,
      // so queue + solve + tx is the server residence time as for the DOCA server
      const uint64_t res = now_ns() - s.t_in;
      r->queue_ns = 0;
      r->tx_ns = uint32_t(res > r->solve_ns ? res - r->solve_ns : 0);
      sock.send(s.h_out, len);
      s.busy = false;
      ++served;
    }
    // one frame, only if a slot is free
    Slot* free_slot = nullptr;
    for (Slot& s : slots)
      if (!s.busy) { free_slot = &s; break; }
    if (!free_slot) continue;
    const ssize_t n = sock.recv(buf.data(), buf.size());
    if (n <= 0) {
      if (now_ns() - last > 1000000000ull) {
        std::printf("served %llu\n", served);
        std::fflush(stdout);
        last = now_ns();
      }
      continue;
    }
    if (!dsbhost::is_dsb_frame(buf.data(), n)) continue;
    const uint8_t type = buf[17];
    if (type == DSB_MSG_LOAD) {
      handle_load(g, sock, pm, buf.data(), size_t(n));
    } else if (type == DSB_MSG_SOLVE && n <= DSB_MAX_FRAME) {
      Slot& s = *free_slot;
      s.t_in = now_ns();
      std::memcpy(s.h_in, buf.data(), size_t(n));
      CK(cudaMemcpyAsync(s.d_in, s.h_in, size_t(n), cudaMemcpyHostToDevice, s.s));
      solve_once_kernel<<<1, DSB_THREADS, g.ctx.smem_bytes, s.s>>>(g.ctx, s.d_in, uint32_t(n), s.d_out, s.d_len);
      CK(cudaMemcpyAsync(s.h_len, s.d_len, 4, cudaMemcpyDeviceToHost, s.s));
      CK(cudaMemcpyAsync(s.h_out, s.d_out, DSB_RSP_SLOT, cudaMemcpyDeviceToHost, s.s));
      CK(cudaEventRecord(s.done, s.s));
      s.busy = true;
    }
  }
  CK(cudaDeviceSynchronize());
  std::printf("served %llu requests\n", served);
  return 0;
}

// ---- proxy mode -------------------------------------------------------------------
static int run_proxy(const Opt& o, GpuCtx& g, RawSock& sock) {
  int workers = resident_workers((const void*)worker_kernel, g.ctx.smem_bytes, o.device);
  if (o.workers > 0) workers = std::min(workers, o.workers);
  const uint32_t nframes = DSB_WORK_CAP;
  uint8_t* pool_d = nullptr;
  uint8_t* pool_h = host_mapped<uint8_t>(size_t(nframes) * DSB_MAX_FRAME, &pool_d);
  std::vector<uint16_t> free_slots;
  for (uint32_t i = 0; i < nframes; ++i) free_slots.push_back(uint16_t(nframes - 1 - i));
  std::vector<uint16_t> client_flags(nframes, 0);
  std::vector<uint64_t> t_in(nframes, 0);

  cudaStream_t ws;
  CK(cudaStreamCreateWithFlags(&ws, cudaStreamNonBlocking));
  worker_kernel<<<workers, DSB_THREADS, g.ctx.smem_bytes, ws>>>(g.ctx);
  CK(cudaGetLastError());
  std::printf("proxy: %d persistent worker CTAs, %u B shared memory each\n", workers, g.ctx.smem_bytes);

  HostProducer prod{&g};
  HostConsumer cons{&g};
  std::vector<PendingModel> pm(DSB_MAX_MODELS);
  unsigned long long served = 0;
  uint64_t last = now_ns();
  while (!g_quit) {
    // replies
    uint32_t len;
    while (const uint8_t* rep = cons.peek(len)) {
      uint8_t* r = const_cast<uint8_t*>(rep);
      dsb_hdr_common* c = reinterpret_cast<dsb_hdr_common*>(r);
      const uint16_t slot = c->flags;  // we stamped the frame-pool slot on ingress
      c->flags = client_flags[slot];
      if (r[17] == DSB_MSG_RESULT) {  // same convention as launch mode
        dsb_result* res = reinterpret_cast<dsb_result*>(r);
        const uint64_t t = now_ns() - t_in[slot];
        res->queue_ns = 0;
        res->tx_ns = uint32_t(t > res->solve_ns ? t - res->solve_ns : 0);
      }
      sock.send(r, len);
      free_slots.push_back(slot);
      cons.pop();
      ++served;
    }
    if (free_slots.empty()) continue;
    const uint16_t slot = free_slots.back();
    uint8_t* fh = pool_h + size_t(slot) * DSB_MAX_FRAME;
    const ssize_t n = sock.recv(fh, DSB_MAX_FRAME);
    if (n <= 0) {
      if (cudaStreamQuery(ws) != cudaErrorNotReady) {
        std::fprintf(stderr, "worker kernel ended: %s\n", cudaGetErrorString(cudaStreamQuery(ws)));
        return 1;
      }
      if (now_ns() - last > 1000000000ull) {
        std::printf("served %llu\n", served);
        std::fflush(stdout);
        last = now_ns();
      }
      continue;
    }
    if (!dsbhost::is_dsb_frame(fh, n)) continue;
    if (fh[17] == DSB_MSG_LOAD) {
      handle_load(g, sock, pm, fh, size_t(n));
    } else if (fh[17] == DSB_MSG_SOLVE) {
      dsb_hdr_common* c = reinterpret_cast<dsb_hdr_common*>(fh);
      client_flags[slot] = c->flags;
      c->flags = slot;
      t_in[slot] = now_ns();
      while (!prod.push(pool_d + size_t(slot) * DSB_MAX_FRAME, uint32_t(n), DSB_WORK_SOLVE, cons.next)) {
      }
      free_slots.pop_back();
    }
  }
  *g.exit_h = 1;
  CK(cudaStreamSynchronize(ws));
  std::printf("served %llu requests\n", served);
  return 0;
}

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (char*)""; };
    if (!std::strcmp(argv[i], "--iface")) o.iface = next();
    else if (!std::strcmp(argv[i], "--mode")) o.mode = next();
    else if (!std::strcmp(argv[i], "--device")) o.device = std::atoi(next());
    else if (!std::strcmp(argv[i], "--streams")) o.streams = std::max(1, std::atoi(next()));
    else if (!std::strcmp(argv[i], "--workers")) o.workers = std::atoi(next());
    else if (!std::strcmp(argv[i], "--smem")) o.smem_kb = std::atoi(next());
    else {
      std::fprintf(stderr, "usage: %s --iface IF --mode launch|proxy [--streams S] [--workers W] "
                           "[--device D] [--smem KB]\n", argv[0]);
      return 2;
    }
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  try {
    CK(cudaSetDevice(o.device));
    const uint32_t smem = pick_smem(o.device, o.smem_kb * 1024u);
    RawSock sock;
    sock.open(o.iface);
    std::printf("dsb_base_server: %s on %s (MTU %d), GPU %d, mode %s\n",
                "listening", o.iface.c_str(), sock.mtu, o.device, o.mode.c_str());
    if (sock.mtu < DSB_MAX_FRAME - 14)
      std::printf("warning: MTU %d < %d; large requests/fragments will be dropped (set MTU 9000)\n",
                  sock.mtu, DSB_MAX_FRAME - 14);
    GpuCtx g;
    g.init(o.mode == "proxy", smem);
    const int rc = (o.mode == "proxy") ? run_proxy(o, g, sock) : run_launch(o, g, sock);
    g.destroy();
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
