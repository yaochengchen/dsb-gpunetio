// dsb_host.hpp -- frame builders/parsers shared by the client, the baseline
// server, the self test and the benchmark.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dsb_model.h"
#include "dsb_proto.h"

namespace dsbhost {

struct Mac {
  uint8_t b[6];
};

inline bool parse_mac(const char* s, Mac& m) {
  unsigned v[6];
  if (std::sscanf(s, "%x:%x:%x:%x:%x:%x", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6)
    return false;
  for (int i = 0; i < 6; ++i) m.b[i] = uint8_t(v[i]);
  return true;
}

inline void fill_common(uint8_t* f, const Mac& dst, const Mac& src, uint8_t type, uint32_t req_id,
                        uint64_t client_ts) {
  dsb_hdr_common* c = reinterpret_cast<dsb_hdr_common*>(f);
  std::memcpy(c->d_mac, dst.b, 6);
  std::memcpy(c->s_mac, src.b, 6);
  c->ethertype = uint16_t((DSB_ETHERTYPE >> 8) | ((DSB_ETHERTYPE & 0xFF) << 8));
  c->pad0[0] = c->pad0[1] = 0;
  c->version = DSB_PROTO_VERSION;
  c->type = type;
  c->flags = 0;
  c->req_id = req_id;
  c->client_ts = client_ts;
}

/// All LOAD fragments of one packed model.
inline std::vector<std::vector<uint8_t>> load_frames(const Mac& dst, const Mac& src, uint16_t model_id,
                                                     uint16_t n, uint8_t format, uint32_t generation,
                                                     float xi_default,
                                                     const std::vector<uint8_t>& packed) {
  std::vector<std::vector<uint8_t>> out;
  const uint32_t total = uint32_t(packed.size());
  const uint32_t count = (total + DSB_LOAD_FRAG_BYTES - 1) / DSB_LOAD_FRAG_BYTES;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t off = i * DSB_LOAD_FRAG_BYTES;
    const uint32_t len = std::min<uint32_t>(DSB_LOAD_FRAG_BYTES, total - off);
    std::vector<uint8_t> f(DSB_HDR_BYTES + ((len + 3u) & ~3u), 0);
    fill_common(f.data(), dst, src, DSB_MSG_LOAD, i, 0);
    dsb_load_frag* h = reinterpret_cast<dsb_load_frag*>(f.data());
    h->model_id = model_id;
    h->n = n;
    h->format = format;
    h->frag_idx = uint16_t(i);
    h->total_bytes = total;
    h->offset = off;
    h->frag_bytes = (len + 3u) & ~3u;
    h->generation = generation;
    h->xi_default = xi_default;
    h->frag_count = uint16_t(count);
    std::memcpy(f.data() + DSB_HDR_BYTES, packed.data() + off, len);
    out.push_back(std::move(f));
  }
  return out;
}

struct SolveParams {
  uint16_t model_id = 0;
  uint16_t n = 0;
  uint16_t steps = 100;
  uint16_t replicas = 8;
  uint32_t seed = 1;
  float xi = 0.f;
  float dt = 1.0f;
};

/// Writes a SOLVE frame into `f` (resized) and returns its length.
inline uint32_t solve_frame(std::vector<uint8_t>& f, const Mac& dst, const Mac& src, uint32_t req_id,
                            uint64_t client_ts, const SolveParams& p, const int16_t* h) {
  const uint32_t len = dsb_solve_frame_bytes(p.n, h ? 1u : 0u);
  f.assign(len, 0);
  fill_common(f.data(), dst, src, DSB_MSG_SOLVE, req_id, client_ts);
  dsb_solve_req* q = reinterpret_cast<dsb_solve_req*>(f.data());
  q->model_id = p.model_id;
  q->n = p.n;
  q->steps = p.steps;
  q->replicas = p.replicas;
  q->seed = p.seed;
  q->xi = p.xi;
  q->dt = p.dt;
  q->h_format = h ? 1 : 0;
  if (h) std::memcpy(f.data() + DSB_HDR_BYTES, h, 2u * p.n);
  return len;
}

inline bool is_dsb_frame(const uint8_t* f, size_t len) {
  return len >= DSB_HDR_BYTES && f[12] == (DSB_ETHERTYPE >> 8) && f[13] == (DSB_ETHERTYPE & 0xFF);
}

}  // namespace dsbhost
