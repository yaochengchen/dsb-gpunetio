// selftest.cu -- the GPU solver against the CPU reference, bit for bit.
//
//   ./selftest [--device D] [--smem KB]
//
// 1. single-launch path (solve_once_kernel): every case, spins and 2E must
//    equal dsbref::solve exactly;
// 2. persistent path (worker_kernel + host-mapped work/reply rings): the same
//    requests pushed through the queues, replies matched by req_id.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "dsb_host.hpp"
#include "dsb_ref.hpp"
#include "gpu_ctx.hpp"

using namespace dsbgpu;

struct Case {
  int format, n, B, K;
  bool with_h;
};

struct Built {
  dsbref::Instance inst;
  std::vector<int16_t> h;
  dsbhost::SolveParams sp;
  std::vector<uint8_t> frame;
  dsbref::Result ref;
};

static int check_reply(const Built& b, const uint8_t* rep, uint32_t len, const char* tag) {
  const dsb_result* r = reinterpret_cast<const dsb_result*>(rep);
  if (r->status != DSB_OK) {
    std::printf("  [%s] FAIL status %u\n", tag, r->status);
    return 1;
  }
  const uint32_t W = dsb_words(b.inst.n);
  if (len != dsb_result_frame_bytes(b.inst.n)) {
    std::printf("  [%s] FAIL length %u\n", tag, len);
    return 1;
  }
  std::vector<uint32_t> spins(W);
  std::memcpy(spins.data(), rep + DSB_HDR_BYTES, 4 * W);
  const int64_t e = dsbref::energy_x2(b.inst, b.sp.n && !b.h.empty() ? b.h.data() : nullptr, spins);
  const bool same_spins = spins == b.ref.best_spins;
  const bool ok = r->energy_x2 == b.ref.energy_x2 && e == r->energy_x2 && same_spins &&
                  int(r->best_replica) == b.ref.best;
  if (!ok) {
    std::printf("  [%s] FAIL fmt %d n %d: gpu 2E %lld (recomputed %lld, replica %u) ref 2E %lld "
                "(replica %d) spins %s\n",
                tag, b.inst.format, b.inst.n, (long long)r->energy_x2, (long long)e,
                r->best_replica, (long long)b.ref.energy_x2, b.ref.best,
                same_spins ? "equal" : "differ");
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  int device = 0;
  uint32_t smem_kb = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--smem") && i + 1 < argc) smem_kb = std::atoi(argv[++i]);
  }
  CK(cudaSetDevice(device));
  const uint32_t smem = pick_smem(device, smem_kb * 1024u);
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, device));
  std::printf("device %d: %s, sm_%d%d, %d SMs, worker smem %u B\n", device, prop.name, prop.major,
              prop.minor, prop.multiProcessorCount, smem);

  GpuCtx g;
  g.init(true, smem);
  CK(cudaFuncSetAttribute(solve_once_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));

  const std::vector<Case> cases = {
      {DSB_FMT_PM1, 2, 1, 2, false},       {DSB_FMT_PM1, 33, 3, 50, true},
      {DSB_FMT_PM1, 128, 8, 200, false},   {DSB_FMT_PM1, 256, 16, 100, true},
      {DSB_FMT_PM1, 500, 8, 60, false},    {DSB_FMT_TERNARY, 64, 4, 100, false},
      {DSB_FMT_TERNARY, 200, 8, 150, true}, {DSB_FMT_TERNARY, 384, 8, 50, false},
      {DSB_FMT_INT8, 40, 2, 80, true},     {DSB_FMT_INT8, 128, 8, 120, true},
      {DSB_FMT_INT8, 160, 16, 60, true},
  };
  const dsbhost::Mac srv{{2, 0, 0, 0, 0, 1}}, cli{{2, 0, 0, 0, 0, 2}};

  std::vector<Built> built;
  int fails = 0, id = 0;
  for (const Case& c : cases) {
    dsb_smem_plan plan;
    if (dsb_make_plan(c.n, c.B, c.format, smem - DSB_SMEM_RESERVE, &plan) != 0) {
      std::printf("skip fmt %d n %d B %d: does not fit %u B\n", c.format, c.n, c.B, smem);
      continue;
    }
    Built b;
    if (c.format == DSB_FMT_PM1) b.inst = dsbref::make_pm1(c.n, 100 + id);
    else if (c.format == DSB_FMT_TERNARY) b.inst = dsbref::make_ternary(c.n, 0.08, false, 100 + id);
    else b.inst = dsbref::make_portfolio(c.n, 3, std::max(1, c.n / 8), 100 + id).inst;
    if (c.with_h) {
      std::mt19937 rg(id);
      b.h.resize(c.n);
      for (auto& v : b.h) v = int16_t(int(rg() % 41) - 20) * (c.format == DSB_FMT_INT8 ? 50 : 1);
    }
    const auto packed = dsbref::pack(b.inst);
    g.upload_model(uint16_t(id), uint16_t(c.n), uint8_t(c.format), b.inst.xi_default, packed, 1);
    b.sp.model_id = uint16_t(id);
    b.sp.n = uint16_t(c.n);
    b.sp.steps = uint16_t(c.K);
    b.sp.replicas = uint16_t(c.B);
    b.sp.seed = 1000u + id;
    b.sp.dt = 1.0f;
    b.sp.xi = 0.f;
    dsbhost::solve_frame(b.frame, srv, cli, uint32_t(id), 0, b.sp, c.with_h ? b.h.data() : nullptr);
    b.ref = dsbref::solve(b.inst, c.with_h ? b.h.data() : nullptr, c.K, c.B, b.sp.seed, 0.f, 1.0f);
    built.push_back(std::move(b));
    ++id;
  }

  // ---- 1. single launch ------------------------------------------------------
  uint8_t *d_frame = nullptr, *d_out = nullptr;
  uint32_t* d_len = nullptr;
  CK(cudaMalloc(&d_frame, DSB_MAX_FRAME));
  CK(cudaMalloc(&d_out, DSB_RSP_SLOT));
  CK(cudaMalloc(&d_len, 4));
  for (const Built& b : built) {
    CK(cudaMemcpy(d_frame, b.frame.data(), b.frame.size(), cudaMemcpyHostToDevice));
    solve_once_kernel<<<1, DSB_THREADS, smem>>>(g.ctx, d_frame, uint32_t(b.frame.size()), d_out, d_len);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    uint8_t rep[DSB_RSP_SLOT];
    uint32_t len = 0;
    CK(cudaMemcpy(rep, d_out, DSB_RSP_SLOT, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&len, d_len, 4, cudaMemcpyDeviceToHost));
    const int f = check_reply(b, rep, len, "once");
    fails += f;
    if (!f)
      std::printf("  [once] ok  fmt %d n %4d B %3d K %4d : 2E %lld  solve %u ns\n", b.inst.format,
                  b.inst.n, b.sp.replicas, b.sp.steps, (long long)b.ref.energy_x2,
                  reinterpret_cast<const dsb_result*>(rep)->solve_ns);
  }

  // ---- 2. persistent workers through the rings -------------------------------
  {
    const int workers = std::min(16, resident_workers((const void*)worker_kernel, smem, device));
    uint8_t* pool_d = nullptr;
    uint8_t* pool_h = host_mapped<uint8_t>(size_t(built.size()) * 4 * DSB_MAX_FRAME, &pool_d);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    worker_kernel<<<workers, DSB_THREADS, smem, s>>>(g.ctx);
    CK(cudaGetLastError());
    HostProducer prod{&g};
    HostConsumer cons{&g};
    size_t k = 0;
    const size_t total = built.size() * 4;
    for (size_t i = 0; i < total; ++i) {
      const Built& b = built[i % built.size()];
      uint8_t* fh = pool_h + i * DSB_MAX_FRAME;
      std::memcpy(fh, b.frame.data(), b.frame.size());
      reinterpret_cast<dsb_hdr_common*>(fh)->req_id = uint32_t(i);
      while (!prod.push(pool_d + i * DSB_MAX_FRAME, uint32_t(b.frame.size()), DSB_WORK_SOLVE, cons.next)) {
      }
    }
    int got = 0;
    while (got < int(total)) {
      uint32_t len = 0;
      const uint8_t* rep = cons.peek(len);
      if (!rep) {
        if (cudaStreamQuery(s) != cudaErrorNotReady) {
          std::printf("  [ring] FAIL worker kernel ended early: %s\n", cudaGetErrorString(cudaGetLastError()));
          return 1;
        }
        continue;
      }
      const uint32_t rid = reinterpret_cast<const dsb_hdr_common*>(rep)->req_id;
      fails += check_reply(built[rid % built.size()], rep, len, "ring");
      cons.pop();
      ++got;
      ++k;
    }
    *g.exit_h = 1;
    CK(cudaStreamSynchronize(s));
    std::printf("  [ring] %d replies through %d persistent workers checked\n", got, workers);
    cudaFreeHost(pool_h);
  }

  std::printf(fails ? "SELFTEST FAILED (%d)\n" : "SELFTEST PASSED\n", fails);
  g.destroy();
  return fails ? 1 : 0;
}
