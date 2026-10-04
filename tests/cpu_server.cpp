// cpu_server.cpp -- a CPU-only stand-in for the servers, for testing the wire
// protocol, fragmentation, LOAD_ACK and the client's verification without a
// GPU or a ConnectX NIC (e.g. over a veth pair).  Not a baseline.
//
//   sudo ./cpu_server --iface vtest1
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "dsb_host.hpp"
#include "dsb_queue.h"
#include "dsb_ref.hpp"
#include "rawsock.hpp"

static volatile sig_atomic_t quit = 0;

// inverse of dsbref::pack
static dsbref::Instance unpack(const std::vector<uint8_t>& p, int n, int format, float xi) {
  dsbref::Instance I;
  I.n = n;
  I.format = format;
  I.xi_default = xi;
  I.J.assign(size_t(n) * n, 0);
  const uint32_t W = dsb_words(n);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      if (r == c) continue;
      int v = 0;
      if (format == DSB_FMT_PM1) {
        const uint32_t* J = reinterpret_cast<const uint32_t*>(p.data());
        v = ((J[r * W + (c >> 5)] >> (c & 31)) & 1) ? 1 : -1;
      } else if (format == DSB_FMT_TERNARY) {
        const uint32_t* P = reinterpret_cast<const uint32_t*>(p.data());
        const uint32_t* M = P + size_t(n) * W;
        v = ((P[r * W + (c >> 5)] >> (c & 31)) & 1) - int((M[r * W + (c >> 5)] >> (c & 31)) & 1);
      } else {
        v = reinterpret_cast<const int8_t*>(p.data())[size_t(r) * dsb_int8_ld(n) + c];
      }
      I.J[size_t(r) * n + c] = v;
    }
  return I;
}

int main(int argc, char** argv) {
  std::string iface = argc > 2 ? argv[2] : "vtest1";
  std::signal(SIGINT, [](int) { quit = 1; });
  RawSock s;
  s.open(iface);
  struct M { uint32_t gen = 0, total = 0, got = 0; int n = 0, fmt = 0; float xi = 0; std::vector<uint8_t> d; bool ready = false; dsbref::Instance I; };
  std::vector<M> models(DSB_MAX_MODELS);
  std::vector<uint8_t> b(DSB_MAX_FRAME + 64), out(DSB_RSP_SLOT);
  long served = 0;
  while (!quit) {
    const ssize_t r = s.recv(b.data(), b.size());
    if (r < DSB_HDR_BYTES || !dsbhost::is_dsb_frame(b.data(), r)) continue;
    const dsb_hdr_common* c = reinterpret_cast<const dsb_hdr_common*>(b.data());
    dsbhost::Mac dst, src;
    std::memcpy(dst.b, c->s_mac, 6);
    std::memcpy(src.b, c->d_mac, 6);
    if (c->type == DSB_MSG_LOAD) {
      const dsb_load_frag* h = reinterpret_cast<const dsb_load_frag*>(b.data());
      M& m = models[h->model_id];
      if (m.gen != h->generation) { m = M(); m.gen = h->generation; m.total = h->total_bytes; m.n = h->n; m.fmt = h->format; m.xi = h->xi_default; m.d.assign(m.total + 4, 0); }
      std::memcpy(m.d.data() + h->offset, b.data() + DSB_HDR_BYTES, h->frag_bytes);
      m.got += h->frag_bytes;
      if (m.got >= m.total && !m.ready) {
        m.d.resize(m.total);
        m.I = unpack(m.d, m.n, m.fmt, m.xi);
        m.ready = true;
        std::memset(out.data(), 0, DSB_HDR_BYTES);
        dsbhost::fill_common(out.data(), dst, src, DSB_MSG_LOAD_ACK, c->req_id, c->client_ts);
        dsb_load_ack* a = reinterpret_cast<dsb_load_ack*>(out.data());
        a->model_id = h->model_id; a->n = h->n; a->status = DSB_OK; a->generation = h->generation; a->total_bytes = h->total_bytes;
        s.send(out.data(), DSB_HDR_BYTES);
      }
    } else if (c->type == DSB_MSG_SOLVE) {
      const dsb_solve_req* q = reinterpret_cast<const dsb_solve_req*>(b.data());
      M& m = models[q->model_id];
      std::memset(out.data(), 0, out.size());
      dsbhost::fill_common(out.data(), dst, src, DSB_MSG_RESULT, c->req_id, c->client_ts);
      dsb_result* o = reinterpret_cast<dsb_result*>(out.data());
      o->model_id = q->model_id; o->n = q->n;
      uint32_t len = DSB_HDR_BYTES;
      if (!m.ready || m.n != q->n) {
        o->status = DSB_ERR_NO_MODEL;
      } else {
        const int16_t* hh = q->h_format ? reinterpret_cast<const int16_t*>(b.data() + DSB_HDR_BYTES) : nullptr;
        const auto R = dsbref::solve(m.I, hh, q->steps, q->replicas, q->seed, q->xi, q->dt);
        o->status = DSB_OK; o->best_replica = R.best; o->energy_x2 = R.energy_x2;
        std::memcpy(out.data() + DSB_HDR_BYTES, R.best_spins.data(), 4 * R.best_spins.size());
        len = dsb_result_frame_bytes(q->n);
      }
      s.send(out.data(), len);
      ++served;
    }
  }
  std::printf("served %ld\n", served);
  return 0;
}
