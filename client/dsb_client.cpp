// dsb_client.cpp -- load generator and latency recorder for the dSB service.
// Speaks raw Ethernet (AF_PACKET), so the same client drives the DOCA server
// and both CPU+GPU baselines.  Needs root.
//
//   sudo ./dsb_client --iface enP2s2np0 --server 58:a2:e1:db:db:80
//        --format int8 --n 128 --B 8 --K 200 --count 200000
//        [--rate 20000 [--poisson]] [--window 1] [--models 4]
//        [--csv out.csv] [--verify 2000] [--verify-ref 20] [--tag doca]
//
//   --rate 0 (default) is closed loop with --window requests outstanding;
//   --rate R > 0 is open loop at R requests/s (fixed interval, or Poisson).
//
// Every request gets its own seed and a field h from a pre-generated pool
// (for int8, a portfolio: expected returns follow a random walk; J -- the
// covariance -- is uploaded once per model).  RTT is measured on the client
// with CLOCK_MONOTONIC between sendto() and recv().  After the run the
// client recomputes 2E from the returned spins for --verify requests and
// re-runs the full CPU reference trajectory for --verify-ref requests; both
// must match exactly.
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "dsb_host.hpp"
#include "dsb_ref.hpp"
#include "rawsock.hpp"

static uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

static void pin(int cpu) {
  if (cpu < 0) return;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

struct Opt {
  std::string iface = "enP2s2np0", server = "58:a2:e1:db:db:80", csv, tag = "run";
  int format = DSB_FMT_INT8, n = 128, B = 8, K = 200, models = 1, window = 1;
  long count = 100000, warmup = 1000, verify = 2000, verify_ref = 20;
  double rate = 0;
  bool poisson = false, no_h = false;
  int tx_cpu = -1, rx_cpu = -1;
  uint32_t seed = 1;
  double timeout_s = 2.0;
  float dt = 1.0f;
};

static int parse_format(const char* s) {
  if (!std::strcmp(s, "pm1")) return DSB_FMT_PM1;
  if (!std::strcmp(s, "ternary")) return DSB_FMT_TERNARY;
  if (!std::strcmp(s, "int8")) return DSB_FMT_INT8;
  std::fprintf(stderr, "unknown format %s\n", s);
  std::exit(2);
}

struct Rec {
  uint64_t t_send = 0, t_recv = 0;
  int64_t e2 = 0;
  uint32_t queue_ns = 0, solve_ns = 0, tx_ns = 0, worker = 0;
  uint16_t status = 0xFFFF;
};

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    auto nx = [&]() { return (i + 1 < argc) ? argv[++i] : (char*)"0"; };
    const char* a = argv[i];
    if (!std::strcmp(a, "--iface")) o.iface = nx();
    else if (!std::strcmp(a, "--server")) o.server = nx();
    else if (!std::strcmp(a, "--format")) o.format = parse_format(nx());
    else if (!std::strcmp(a, "--n")) o.n = std::atoi(nx());
    else if (!std::strcmp(a, "--B")) o.B = std::atoi(nx());
    else if (!std::strcmp(a, "--K")) o.K = std::atoi(nx());
    else if (!std::strcmp(a, "--models")) o.models = std::max(1, std::atoi(nx()));
    else if (!std::strcmp(a, "--window")) o.window = std::max(1, std::atoi(nx()));
    else if (!std::strcmp(a, "--count")) o.count = std::atol(nx());
    else if (!std::strcmp(a, "--warmup")) o.warmup = std::atol(nx());
    else if (!std::strcmp(a, "--rate")) o.rate = std::atof(nx());
    else if (!std::strcmp(a, "--poisson")) o.poisson = true;
    else if (!std::strcmp(a, "--no-h")) o.no_h = true;
    else if (!std::strcmp(a, "--csv")) o.csv = nx();
    else if (!std::strcmp(a, "--tag")) o.tag = nx();
    else if (!std::strcmp(a, "--verify")) o.verify = std::atol(nx());
    else if (!std::strcmp(a, "--verify-ref")) o.verify_ref = std::atol(nx());
    else if (!std::strcmp(a, "--tx-cpu")) o.tx_cpu = std::atoi(nx());
    else if (!std::strcmp(a, "--rx-cpu")) o.rx_cpu = std::atoi(nx());
    else if (!std::strcmp(a, "--seed")) o.seed = uint32_t(std::atol(nx()));
    else if (!std::strcmp(a, "--timeout")) o.timeout_s = std::atof(nx());
    else if (!std::strcmp(a, "--dt")) o.dt = float(std::atof(nx()));
    else {
      std::fprintf(stderr, "unknown option %s (see the header of dsb_client.cpp)\n", a);
      return 2;
    }
  }

  RawSock sock;
  dsbhost::Mac srv;
  try {
    if (!dsbhost::parse_mac(o.server.c_str(), srv)) throw std::runtime_error("bad --server MAC");
    sock.open(o.iface);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  const dsbhost::Mac me = sock.mac;
  const uint32_t solve_bytes = dsb_solve_frame_bytes(o.n, o.no_h ? 0u : 1u);
  if (solve_bytes > DSB_MAX_FRAME) {
    std::fprintf(stderr, "n=%d does not fit one %u-byte request frame\n", o.n, DSB_MAX_FRAME);
    return 2;
  }
  std::printf("client %02x:%02x:%02x:%02x:%02x:%02x on %s (MTU %d) -> server %s\n", me.b[0], me.b[1],
              me.b[2], me.b[3], me.b[4], me.b[5], o.iface.c_str(), sock.mtu, o.server.c_str());
  // pre-flight: every frame this run sends must pass the interface MTU, otherwise
  // sendto() fails with EMSGSIZE and the server never sees a single frame
  {
    const size_t need = std::max<size_t>(solve_bytes, DSB_HDR_BYTES + DSB_LOAD_FRAG_BYTES);
    if (need > sock.max_frame()) {
      std::fprintf(stderr, "error: this run sends %zu-byte frames but %s has MTU %d (max frame %zu)\n"
                           "       fix: sudo ip link set %s mtu 9000\n",
                   need, o.iface.c_str(), sock.mtu, sock.max_frame(), o.iface.c_str());
      return 1;
    }
  }

  // ---- instances, fields ----------------------------------------------------
  std::vector<dsbref::Instance> inst(o.models);
  std::vector<dsbref::Portfolio> port(o.models);
  for (int m = 0; m < o.models; ++m) {
    const uint64_t s = o.seed * 1000ull + m;
    if (o.format == DSB_FMT_PM1) inst[m] = dsbref::make_pm1(o.n, s);
    else if (o.format == DSB_FMT_TERNARY) inst[m] = dsbref::make_ternary(o.n, 0.05, true, s);
    else {
      port[m] = dsbref::make_portfolio(o.n, 4, std::max(1, o.n / 10), s);
      inst[m] = port[m].inst;
    }
  }
  const int HPOOL = 256;
  std::vector<std::vector<int16_t>> hpool(size_t(o.models) * HPOOL);
  {
    std::mt19937_64 g(o.seed);
    std::normal_distribution<double> N01;
    for (int m = 0; m < o.models; ++m) {
      std::vector<double> mu(o.n);
      for (auto& v : mu) v = 0.02 * N01(g);
      for (int k = 0; k < HPOOL; ++k) {
        auto& h = hpool[size_t(m) * HPOOL + k];
        if (o.format == DSB_FMT_INT8) {
          for (auto& v : mu) v = 0.98 * v + 0.004 * N01(g);  // returns drift tick to tick
          h = dsbref::portfolio_field(port[m], mu);
        } else {
          h.resize(o.n);
          for (auto& v : h) v = int16_t(std::lround(3.0 * N01(g)));
        }
      }
    }
  }

  // ---- upload models ---------------------------------------------------------
  std::vector<uint8_t> rbuf(DSB_MAX_FRAME + 64);
  for (int m = 0; m < o.models; ++m) {
    const auto packed = dsbref::pack(inst[m]);
    if (packed.size() > DSB_MODEL_SLOT_BYTES) {
      std::fprintf(stderr, "model of %zu bytes exceeds the %u-byte model slot\n", packed.size(), DSB_MODEL_SLOT_BYTES);
      return 2;
    }
    const uint32_t gen = uint32_t(now_ns() & 0x7FFFFFFF) | 1u;
    const auto frames = dsbhost::load_frames(srv, me, uint16_t(m), uint16_t(o.n), uint8_t(o.format), gen,
                                             inst[m].xi_default, packed);
    bool acked = false;
    for (int attempt = 0; attempt < 5 && !acked; ++attempt) {
      for (const auto& f : frames) {
        if (!sock.send(f.data(), f.size())) {
          std::fprintf(stderr, "error: sendto(%zu bytes) on %s failed: %s\n", f.size(), o.iface.c_str(),
                       sock.send_error());
          return 1;
        }
        const uint64_t t = now_ns();
        while (now_ns() - t < 2000) {}  // 2 us between fragments
      }
      const uint64_t t0 = now_ns();
      while (now_ns() - t0 < 500000000ull) {
        const ssize_t r = sock.recv(rbuf.data(), rbuf.size());
        if (r < DSB_HDR_BYTES || rbuf[17] != DSB_MSG_LOAD_ACK) continue;
        const dsb_load_ack* a = reinterpret_cast<const dsb_load_ack*>(rbuf.data());
        if (a->model_id == m && a->generation == gen) {
          acked = (a->status == DSB_OK);
          break;
        }
      }
    }
    if (!acked) {
      std::fprintf(stderr,
                   "model %d: %zu LOAD fragments sent, no LOAD_ACK within 5 x 0.5 s.\n"
                   "  check on the server: rx_frames must have increased by %zu per attempt.\n"
                   "  rx_frames stayed 0  -> frames never reached the GPU queue: server not running,\n"
                   "                         wrong --server MAC, another server still holds the Ethertype,\n"
                   "                         link down (ethtool %s)\n"
                   "  rx_frames increased -> the reply path is broken (server tx_frames / txerr)\n",
                   m, frames.size(), frames.size(), o.iface.c_str());
      return 1;
    }
    std::printf("model %d: n=%d format=%d %zu bytes in %zu fragments, acknowledged\n", m, o.n, o.format,
                packed.size(), frames.size());
  }

  // ---- pre-built request frames ------------------------------------------------
  const long total = o.warmup + o.count;
  std::vector<Rec> rec(total);
  const long keep = std::min<long>(total, o.verify + o.warmup);
  std::vector<std::vector<uint32_t>> spins(keep);
  std::atomic<long> received{0};
  std::atomic<long> in_flight{0};
  std::atomic<bool> stop{false};

  std::thread rx([&]() {
    pin(o.rx_cpu);
    std::vector<uint8_t> b(DSB_MAX_FRAME + 64);
    while (!stop.load(std::memory_order_relaxed)) {
      const ssize_t r = sock.recv(b.data(), b.size());
      if (r < DSB_HDR_BYTES || b[17] != DSB_MSG_RESULT) continue;
      const uint64_t t = now_ns();
      const dsb_result* res = reinterpret_cast<const dsb_result*>(b.data());
      const uint32_t id = res->c.req_id;
      if (id >= uint32_t(total) || rec[id].t_recv) continue;
      Rec& q = rec[id];
      q.t_recv = t;
      q.status = res->status;
      q.e2 = res->energy_x2;
      q.queue_ns = res->queue_ns;
      q.solve_ns = res->solve_ns;
      q.tx_ns = res->tx_ns;
      q.worker = res->worker;
      if (long(id) < keep && res->status == DSB_OK) {
        const uint32_t W = dsb_words(o.n);
        spins[id].resize(W);
        std::memcpy(spins[id].data(), b.data() + DSB_HDR_BYTES, 4 * W);
      }
      received.fetch_add(1, std::memory_order_relaxed);
      in_flight.fetch_sub(1, std::memory_order_release);
    }
  });

  pin(o.tx_cpu);
  std::vector<uint8_t> frame;
  dsbhost::SolveParams sp;
  sp.n = uint16_t(o.n);
  sp.steps = uint16_t(o.K);
  sp.replicas = uint16_t(o.B);
  sp.dt = o.dt;
  std::mt19937_64 arr(o.seed + 7);
  std::exponential_distribution<double> expo(o.rate > 0 ? o.rate : 1.0);
  const uint64_t t_start = now_ns();
  uint64_t t_next = t_start;
  long timeouts = 0;
  for (long i = 0; i < total; ++i) {
    const int m = int(i % o.models);
    sp.model_id = uint16_t(m);
    sp.seed = o.seed * 7919u + uint32_t(i);
    const int16_t* h = o.no_h ? nullptr : hpool[size_t(m) * HPOOL + (i % HPOOL)].data();
    if (o.rate > 0) {
      while (now_ns() < t_next) {}
      t_next += o.poisson ? uint64_t(expo(arr) * 1e9) : uint64_t(1e9 / o.rate);
    } else {
      const uint64_t tw = now_ns();
      while (in_flight.load(std::memory_order_acquire) >= o.window) {
        if (now_ns() - tw > uint64_t(o.timeout_s * 1e9)) {  // lost reply: give up on it
          in_flight.fetch_sub(1);
          ++timeouts;
          break;
        }
      }
    }
    const uint64_t ts = now_ns();
    dsbhost::solve_frame(frame, srv, me, uint32_t(i), ts, sp, h);
    rec[i].t_send = ts;
    in_flight.fetch_add(1, std::memory_order_acq_rel);
    if (!sock.send(frame.data(), frame.size())) {
      in_flight.fetch_sub(1);
      if (sock.send_fail == 1)
        std::fprintf(stderr, "warning: sendto failed: %s (counted as lost)\n", sock.send_error());
    }
  }
  const uint64_t t_sent = now_ns();
  while (received.load() < total && now_ns() - t_sent < uint64_t(o.timeout_s * 1e9)) {}
  stop = true;
  rx.join();
  const uint64_t t_end = now_ns();

  // ---- statistics ----------------------------------------------------------------
  std::vector<double> rtt, srv_ns, net_ns;
  long ok = 0, err = 0, lost = 0;
  uint64_t first = 0, last = 0;
  for (long i = o.warmup; i < total; ++i) {
    const Rec& q = rec[i];
    if (!q.t_recv) { ++lost; continue; }
    if (q.status != DSB_OK) { ++err; continue; }
    ++ok;
    const double r = double(q.t_recv - q.t_send);
    const double s = double(q.queue_ns) + q.solve_ns + q.tx_ns;
    rtt.push_back(r * 1e-3);
    srv_ns.push_back(s * 1e-3);
    net_ns.push_back((r - s) * 1e-3);
    if (!first || q.t_send < first) first = q.t_send;
    last = std::max(last, q.t_recv);
  }
  auto pct = [](std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p / 100.0 * (v.size() - 1) + 0.5))];
  };
  auto mean = [](const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / v.size();
  };
  const double secs = (last > first) ? double(last - first) * 1e-9 : 1e-9;

  // ---- verification --------------------------------------------------------------
  long vchk = 0, vbad = 0, rchk = 0, rbad = 0;
  for (long i = 0; i < keep; ++i) {
    if (spins[i].empty()) continue;
    const int m = int(i % o.models);
    const int16_t* h = o.no_h ? nullptr : hpool[size_t(m) * HPOOL + (i % HPOOL)].data();
    ++vchk;
    if (dsbref::energy_x2(inst[m], h, spins[i]) != rec[i].e2) ++vbad;
    if (rchk < o.verify_ref) {
      ++rchk;
      const auto R = dsbref::solve(inst[m], h, o.K, o.B, o.seed * 7919u + uint32_t(i), 0.f, o.dt);
      if (R.energy_x2 != rec[i].e2 || R.best_spins != spins[i]) ++rbad;
    }
  }

  std::printf("\n[%s] n=%d B=%d K=%d format=%d models=%d %s\n", o.tag.c_str(), o.n, o.B, o.K, o.format,
              o.models, o.rate > 0 ? (o.poisson ? "open-loop Poisson" : "open-loop fixed") : "closed-loop");
  std::printf("  requests %ld ok %ld error %ld lost %ld timeouts %ld send_fail %llu, throughput %.1f kreq/s (offered %.1f)\n",
              o.count, ok, err, lost, timeouts, (unsigned long long)sock.send_fail, ok / secs * 1e-3,
              o.rate * 1e-3);
  std::printf("  RTT us     p50 %.2f  p90 %.2f  p99 %.2f  p99.9 %.2f  max %.2f  mean %.2f\n", pct(rtt, 50),
              pct(rtt, 90), pct(rtt, 99), pct(rtt, 99.9), pct(rtt, 100), mean(rtt));
  std::printf("  server us  p50 %.2f  p99 %.2f   (queue+solve+tx as reported)\n", pct(srv_ns, 50),
              pct(srv_ns, 99));
  std::printf("  wire+stack p50 %.2f  p99 %.2f   (RTT - server)\n", pct(net_ns, 50), pct(net_ns, 99));
  std::printf("  verify: energy %ld/%ld match, reference trajectory %ld/%ld match\n", vchk - vbad, vchk,
              rchk - rbad, rchk);
  std::printf("SUMMARY,%s,%d,%d,%d,%d,%d,%.0f,%d,%ld,%ld,%ld,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%ld,%ld\n",
              o.tag.c_str(), o.format, o.n, o.B, o.K, o.models, o.rate, o.window, ok, err, lost,
              ok / secs * 1e-3, pct(rtt, 50), pct(rtt, 99), pct(rtt, 99.9), mean(rtt), pct(srv_ns, 50),
              pct(srv_ns, 99), pct(net_ns, 50), vbad, rbad);

  if (!o.csv.empty()) {
    FILE* f = std::fopen(o.csv.c_str(), "w");
    if (f) {
      std::fprintf(f, "req_id,t_send_ns,rtt_ns,queue_ns,solve_ns,tx_ns,worker,status,energy_x2\n");
      for (long i = o.warmup; i < total; ++i) {
        const Rec& q = rec[i];
        std::fprintf(f, "%ld,%llu,%lld,%u,%u,%u,%u,%d,%lld\n", i, (unsigned long long)(q.t_send - t_start),
                     q.t_recv ? (long long)(q.t_recv - q.t_send) : -1LL, q.queue_ns, q.solve_ns, q.tx_ns,
                     q.worker, q.t_recv ? int(q.status) : -1, (long long)q.e2);
      }
      std::fclose(f);
    }
  }
  (void)t_end;
  return (vbad || rbad) ? 3 : 0;
}
