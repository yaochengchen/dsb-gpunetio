// dsb_bench.cu -- GPU-only measurements, no NIC required.
//
//   ./dsb_bench single     --format pm1|ternary|int8 --n N --B B --K K [--reps R]
//        one request per launch: in-kernel solve time, launch-to-completion
//        time, and the host round trip H2D + launch + D2H (the per-request
//        cost of the conventional CPU-driven path, network excluded).
//
//   ./dsb_bench persistent --format ... --n N --B B --K K [--count C] [--rate R]
//        persistent worker CTAs fed by a GPU generator warp and drained by a
//        GPU consumer warp through the same rings the DOCA server uses: the
//        GPU-resident pipeline without the NIC.  Rate 0 = as fast as possible.
//
//   ./dsb_bench proxy      --format ... --n N --B B --K K [--count C] [--window W] [--drain]
//        persistent workers, CPU producer and consumer through host-mapped
//        rings: the CPU-proxied pipeline without the NIC.  Defaults (window 256,
//        at most one push and one pop per loop iteration) reproduce the
//        original runs.  --drain pops every ready reply and then pushes until
//        the window is full in each iteration, as base_server.cu drains its
//        reply ring; --window 4096 matches the server's frame pool (DSB_WORK_CAP).
//
// Every mode prints one CSV line (header with --header).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dsb_host.hpp"
#include "dsb_ref.hpp"
#include "gpu_ctx.hpp"

using namespace dsbgpu;

struct Opt {
  std::string mode = "single";
  int format = DSB_FMT_PM1, n = 128, B = 8, K = 100, reps = 200, device = 0;
  long count = 100000;
  double rate = 0;  // requests per second, 0 = max
  uint32_t smem_kb = 0;
  int workers = 0;
  bool header = false;
  uint32_t window = 256;  // proxy mode: requests outstanding (frame-pool slots)
  bool drain = false;     // proxy mode: drain all ready replies / refill the window per iteration
};

static int parse_format(const char* s) {
  if (!std::strcmp(s, "pm1")) return DSB_FMT_PM1;
  if (!std::strcmp(s, "ternary")) return DSB_FMT_TERNARY;
  if (!std::strcmp(s, "int8")) return DSB_FMT_INT8;
  std::fprintf(stderr, "unknown format %s\n", s);
  std::exit(2);
}
static const char* fmt_name(int f) { return f == 1 ? "pm1" : f == 2 ? "ternary" : "int8"; }

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = size_t(p / 100.0 * (v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}

static double now_ns() {
  return double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
}

// ---- GPU generator / consumer (one block, warp 0 and warp 1, lane 0 each) ----
struct GenArgs {
  const uint8_t* frames;  // [nframes][DSB_MAX_FRAME] device
  uint32_t nframes, frame_bytes;
  long count;
  unsigned long long interval_ns;  // 0 = unpaced
  uint32_t* lat_ns;                // [count]
  unsigned long long* done_flag;   // consumer sets when count replies seen
  unsigned long long* t_first;     // [2] first send, last reply
};

__global__ void gen_consume_kernel(dsb_ctx ctx, GenArgs a) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned long long next_check = 0ull;  // throttled exit-flag poll (host-mapped flag)
  if (lane != 0) return;
  if (warp == 0) {
    unsigned long long next_t = gtimer();
    a.t_first[0] = next_t;
    for (long i = 0; i < a.count; ++i) {
      if (a.interval_ns) {
        while (gtimer() < next_t) {}
        next_t += a.interval_ns;
      }
      dsb_work_item* it = ctx.items + (i & (DSB_WORK_CAP - 1));
      while (!dsb_slot_free(*(volatile unsigned long long*)&it->ack, (unsigned long long)i)) {
        if (should_exit(ctx, next_check)) return;
      }
      // each frame slot carries its own req_id; reuse frames round robin
      const uint8_t* f = a.frames + size_t(i % a.nframes) * DSB_MAX_FRAME;
      const unsigned long long t = gtimer();
      it->t_rx = t;
      it->frame = f;
      it->bytes = a.frame_bytes;
      it->kind = DSB_WORK_SOLVE;
      __threadfence();
      *(volatile unsigned long long*)&it->seq = (unsigned long long)i + 1ull;
    }
  } else if (warp == 1) {
    // Frames are reused round robin, so a reply's req_id names the frame, not
    // the request.  The reply carries queue_ns = t_start - t_rx and
    // solve_ns = t_end - t_start; rsp_tend is stamped right after the reply
    // is written, so  latency = (now - tend) + solve_ns + queue_ns  counts from
    // the generator's publish to the consumer's observation, except for the
    // reply copy itself (well under a microsecond).
    for (long i = 0; i < a.count; ++i) {
      const uint32_t slot = uint32_t(i & (DSB_RSP_CAP - 1));
      unsigned long long w;
      do {
        w = *(volatile unsigned long long*)(ctx.rsp_ready + slot);
        if (should_exit(ctx, next_check)) return;
      } while (dsb_ready_seq(w) != (unsigned long long)i + 1ull);
      const unsigned long long now = gtimer();
      __threadfence();
      const volatile dsb_result* r =
          reinterpret_cast<const volatile dsb_result*>(ctx.rsp_buf + size_t(slot) * DSB_RSP_SLOT);
      const unsigned long long tend = *(volatile unsigned long long*)(ctx.rsp_tend + slot);
      a.lat_ns[i] = uint32_t((now > tend ? now - tend : 0ull) + r->queue_ns + r->solve_ns);
      *(volatile unsigned long long*)ctx.rsp_released = (unsigned long long)i + 1ull;
    }
    a.t_first[1] = gtimer();
    *(volatile unsigned long long*)a.done_flag = 1ull;
  }
}

int main(int argc, char** argv) {
  Opt o;
  if (argc > 1 && argv[1][0] != '-') o.mode = argv[1];
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (char*)"0"; };
    if (!std::strcmp(argv[i], "--format")) o.format = parse_format(next());
    else if (!std::strcmp(argv[i], "--n")) o.n = std::atoi(next());
    else if (!std::strcmp(argv[i], "--B")) o.B = std::atoi(next());
    else if (!std::strcmp(argv[i], "--K")) o.K = std::atoi(next());
    else if (!std::strcmp(argv[i], "--reps")) o.reps = std::atoi(next());
    else if (!std::strcmp(argv[i], "--count")) o.count = std::atol(next());
    else if (!std::strcmp(argv[i], "--rate")) o.rate = std::atof(next());
    else if (!std::strcmp(argv[i], "--device")) o.device = std::atoi(next());
    else if (!std::strcmp(argv[i], "--smem")) o.smem_kb = std::atoi(next());
    else if (!std::strcmp(argv[i], "--workers")) o.workers = std::atoi(next());
    else if (!std::strcmp(argv[i], "--header")) o.header = true;
    else if (!std::strcmp(argv[i], "--window")) o.window = uint32_t(std::atoi(next()));
    else if (!std::strcmp(argv[i], "--drain")) o.drain = true;
  }
  CK(cudaSetDevice(o.device));
  const uint32_t smem = pick_smem(o.device, o.smem_kb * 1024u);
  dsb_smem_plan plan{};
  if (dsb_make_plan(o.n, o.B, o.format, smem - DSB_SMEM_RESERVE, &plan) != 0) {
    std::fprintf(stderr, "n=%d B=%d %s does not fit %u B of shared memory (needs %u)\n", o.n, o.B,
                 fmt_name(o.format), smem, plan.bytes);
    return 2;
  }

  // instance, model, frames
  dsbref::Instance inst;
  std::vector<int16_t> h(o.n, 0);
  if (o.format == DSB_FMT_PM1) inst = dsbref::make_pm1(o.n, 1);
  else if (o.format == DSB_FMT_TERNARY) inst = dsbref::make_ternary(o.n, 0.05, true, 1);
  else inst = dsbref::make_portfolio(o.n, 4, std::max(1, o.n / 10), 1).inst;
  std::mt19937 rg(3);
  for (auto& v : h) v = int16_t(int(rg() % 21) - 10);

  const bool host_q = (o.mode == "proxy");
  GpuCtx g;
  g.init(host_q, smem);
  g.upload_model(0, uint16_t(o.n), uint8_t(o.format), inst.xi_default, dsbref::pack(inst), 1);
  dsbhost::SolveParams sp;
  sp.n = uint16_t(o.n);
  sp.steps = uint16_t(o.K);
  sp.replicas = uint16_t(o.B);
  const dsbhost::Mac m1{{2, 0, 0, 0, 0, 1}}, m2{{2, 0, 0, 0, 0, 2}};
  std::vector<uint8_t> frame;
  const uint32_t fbytes = dsbhost::solve_frame(frame, m1, m2, 0, 0, sp, h.data());

  if (o.header)
    std::printf("mode,format,n,B,K,workers,count,rate_req_s,p50_us,p99_us,p999_us,mean_us,extra\n");

  if (o.mode == "single") {
    CK(cudaFuncSetAttribute(solve_once_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    uint8_t *d_frame, *d_out, *h_frame, *h_out;
    uint32_t* d_len;
    CK(cudaMalloc(&d_frame, DSB_MAX_FRAME));
    CK(cudaMalloc(&d_out, DSB_RSP_SLOT));
    CK(cudaMalloc(&d_len, 4));
    CK(cudaMallocHost(&h_frame, DSB_MAX_FRAME));
    CK(cudaMallocHost(&h_out, DSB_RSP_SLOT));
    std::memcpy(h_frame, frame.data(), fbytes);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CK(cudaMemcpy(d_frame, h_frame, fbytes, cudaMemcpyHostToDevice));
    std::vector<double> solve, launch, rtt;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int r = 0; r < o.reps + 10; ++r) {
      // launch only
      CK(cudaEventRecord(e0, s));
      solve_once_kernel<<<1, DSB_THREADS, smem, s>>>(g.ctx, d_frame, fbytes, d_out, d_len);
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      // host round trip
      const double t0 = now_ns();
      CK(cudaMemcpyAsync(d_frame, h_frame, fbytes, cudaMemcpyHostToDevice, s));
      solve_once_kernel<<<1, DSB_THREADS, smem, s>>>(g.ctx, d_frame, fbytes, d_out, d_len);
      CK(cudaMemcpyAsync(h_out, d_out, DSB_RSP_SLOT, cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      const double t1 = now_ns();
      if (r < 10) continue;  // warm-up
      launch.push_back(ms * 1e3);
      rtt.push_back((t1 - t0) * 1e-3);
      solve.push_back(reinterpret_cast<const dsb_result*>(h_out)->solve_ns * 1e-3);
    }
    double ms = 0;
    for (double v : rtt) ms += v;
    std::printf("single-solve,%s,%d,%d,%d,1,%d,,%.2f,%.2f,%.2f,,in-kernel\n", fmt_name(o.format), o.n,
                o.B, o.K, o.reps, pct(solve, 50), pct(solve, 99), pct(solve, 99.9));
    std::printf("single-launch,%s,%d,%d,%d,1,%d,,%.2f,%.2f,%.2f,,event-timed\n", fmt_name(o.format),
                o.n, o.B, o.K, o.reps, pct(launch, 50), pct(launch, 99), pct(launch, 99.9));
    std::printf("single-roundtrip,%s,%d,%d,%d,1,%d,%.0f,%.2f,%.2f,%.2f,%.2f,h2d+launch+d2h\n",
                fmt_name(o.format), o.n, o.B, o.K, o.reps, 1e6 / (ms / rtt.size()), pct(rtt, 50),
                pct(rtt, 99), pct(rtt, 99.9), ms / rtt.size());
    return 0;
  }

  int workers = resident_workers((const void*)worker_kernel, smem, o.device) - 1;  // leave room
  if (o.workers > 0) workers = std::min(workers, o.workers);
  cudaStream_t ws, gs;
  CK(cudaStreamCreateWithFlags(&ws, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking));

  if (o.mode == "persistent") {
    const uint32_t nframes = 64;
    uint8_t* frames = device_zero<uint8_t>(size_t(nframes) * DSB_MAX_FRAME);
    for (uint32_t i = 0; i < nframes; ++i) {
      reinterpret_cast<dsb_hdr_common*>(frame.data())->req_id = i;
      CK(cudaMemcpy(frames + size_t(i) * DSB_MAX_FRAME, frame.data(), fbytes, cudaMemcpyHostToDevice));
    }
    GenArgs a;
    a.frames = frames;
    a.nframes = nframes;
    a.frame_bytes = fbytes;
    a.count = o.count;
    a.interval_ns = o.rate > 0 ? (unsigned long long)(1e9 / o.rate) : 0ull;
    a.lat_ns = device_zero<uint32_t>(o.count);
    unsigned long long* done_d = nullptr;
    unsigned long long* done_h = host_mapped<unsigned long long>(1, &done_d);
    a.done_flag = done_d;
    a.t_first = device_zero<unsigned long long>(2);
    gen_consume_kernel<<<1, 64, 0, gs>>>(g.ctx, a);  // first, so it is resident
    CK(cudaGetLastError());
    worker_kernel<<<workers, DSB_THREADS, smem, ws>>>(g.ctx);
    CK(cudaGetLastError());
    while (__atomic_load_n(done_h, __ATOMIC_ACQUIRE) == 0) {
      if (cudaStreamQuery(gs) != cudaErrorNotReady) break;
    }
    *g.exit_h = 1;
    CK(cudaStreamSynchronize(ws));
    CK(cudaStreamSynchronize(gs));
    std::vector<uint32_t> lat(o.count);
    unsigned long long tt[2];
    CK(cudaMemcpy(lat.data(), a.lat_ns, 4 * o.count, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tt, a.t_first, 16, cudaMemcpyDeviceToHost));
    std::vector<double> us(lat.begin() + std::min<long>(o.count, 1000), lat.end());
    for (auto& v : us) v *= 1e-3;
    double mean = 0;
    for (double v : us) mean += v;
    mean /= std::max<size_t>(1, us.size());
    const double secs = double(tt[1] - tt[0]) * 1e-9;
    std::printf("persistent-gpu,%s,%d,%d,%d,%d,%ld,%.0f,%.2f,%.2f,%.2f,%.2f,offered=%.0f\n",
                fmt_name(o.format), o.n, o.B, o.K, workers, o.count, o.count / secs, pct(us, 50),
                pct(us, 99), pct(us, 99.9), mean, o.rate);
    return 0;
  }

  if (o.mode == "proxy") {
    // frames in host-mapped memory, CPU produces and consumes
    const uint32_t nframes = std::max(1u, std::min(o.window, DSB_WORK_CAP));
    uint8_t* pool_d = nullptr;
    uint8_t* pool_h = host_mapped<uint8_t>(size_t(nframes) * DSB_MAX_FRAME, &pool_d);
    for (uint32_t i = 0; i < nframes; ++i) {
      std::memcpy(pool_h + size_t(i) * DSB_MAX_FRAME, frame.data(), fbytes);
      reinterpret_cast<dsb_hdr_common*>(pool_h + size_t(i) * DSB_MAX_FRAME)->req_id = i;
    }
    worker_kernel<<<workers, DSB_THREADS, smem, ws>>>(g.ctx);
    CK(cudaGetLastError());
    HostProducer prod{&g};
    prod.outstanding_limit = std::min<unsigned long long>(DSB_WORK_CAP, nframes);
    HostConsumer cons{&g};
    std::vector<double> t_send(o.count), lat;
    std::vector<uint32_t> slot_of(o.count), free_slots;
    for (uint32_t i = 0; i < nframes; ++i) free_slots.push_back(nframes - 1 - i);
    lat.reserve(o.count);
    long sent = 0, got = 0;
    const double t0 = now_ns();
    // push one request into frame slot f (false if the ring refused it)
    auto push_one = [&]() -> bool {
      const uint32_t f = free_slots.back();
      uint8_t* fh = pool_h + size_t(f) * DSB_MAX_FRAME;
      reinterpret_cast<dsb_hdr_common*>(fh)->req_id = uint32_t(sent);
      t_send[sent] = now_ns();
      if (!prod.push(pool_d + size_t(f) * DSB_MAX_FRAME, fbytes, DSB_WORK_SOLVE, cons.next)) return false;
      free_slots.pop_back();
      slot_of[sent] = f;
      ++sent;
      return true;
    };
    // pop the next reply if it is ready
    auto pop_one = [&]() -> bool {
      uint32_t len;
      const uint8_t* rep = cons.peek(len);
      if (!rep) return false;
      const uint32_t rid = reinterpret_cast<const dsb_hdr_common*>(rep)->req_id;
      lat.push_back((now_ns() - t_send[rid]) * 1e-3);
      free_slots.push_back(slot_of[rid]);
      cons.pop();
      ++got;
      return true;
    };
    while (got < o.count) {
      if (o.drain) {
        // like base_server.cu: drain every ready reply, then refill the window
        while (pop_one()) {}
        while (sent < o.count && !free_slots.empty() && push_one()) {}
      } else {
        // original loop: at most one push and one pop per iteration;
        // a frame slot is reused only after the reply of its previous request
        if (sent < o.count && !free_slots.empty()) push_one();
        pop_one();
      }
    }
    const double secs = (now_ns() - t0) * 1e-9;
    *g.exit_h = 1;
    CK(cudaStreamSynchronize(ws));
    std::vector<double> us(lat.begin() + std::min<long>(o.count, 1000), lat.end());
    double mean = 0;
    for (double v : us) mean += v;
    mean /= std::max<size_t>(1, us.size());
    std::printf("proxy-cpu,%s,%d,%d,%d,%d,%ld,%.0f,%.2f,%.2f,%.2f,%.2f,closed-window=%u%s\n",
                fmt_name(o.format), o.n, o.B, o.K, workers, o.count, o.count / secs, pct(us, 50),
                pct(us, 99), pct(us, 99.9), mean, nframes, o.drain ? " drain" : "");
    return 0;
  }
  std::fprintf(stderr, "unknown mode %s\n", o.mode.c_str());
  return 2;
}
