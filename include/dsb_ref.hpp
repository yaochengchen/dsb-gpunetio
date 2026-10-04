// dsb_ref.hpp -- host-side instance generators, packers and the CPU reference
// integrator.  The reference reproduces the GPU trajectory bit for bit: the
// coupling sum is an exact integer on both sides, and every floating-point
// operation of the update is written out in the same order (the GPU uses
// __f*_rn intrinsics, the host is compiled with -ffp-contract=off).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "dsb_model.h"

namespace dsbref {

// Dense integer instance: J row-major n x n (symmetric, zero diagonal).
struct Instance {
  int n = 0;
  int format = DSB_FMT_PM1;
  std::vector<int32_t> J;
  float xi_default = 0.f;
};

inline float auto_xi(const Instance& I) {
  double s = 0;
  for (int32_t v : I.J) s += double(v) * v;
  return s > 0 ? float(0.5 * std::sqrt(double(I.n - 1)) / std::sqrt(s)) : 0.f;
}

/// Sherrington-Kirkpatrick style dense +-1 instance.
inline Instance make_pm1(int n, uint64_t seed) {
  Instance I;
  I.n = n;
  I.format = DSB_FMT_PM1;
  I.J.assign(size_t(n) * n, 0);
  std::mt19937_64 g(seed);
  for (int r = 0; r < n; ++r)
    for (int c = r + 1; c < n; ++c) {
      int v = (g() & 1) ? 1 : -1;
      I.J[size_t(r) * n + c] = v;
      I.J[size_t(c) * n + r] = v;
    }
  I.xi_default = auto_xi(I);
  return I;
}

/// Sparse ternary instance (G-set like): each pair is an edge with
/// probability `density`, weight +-1 (or all -1 for a Max-Cut instance).
inline Instance make_ternary(int n, double density, bool maxcut, uint64_t seed) {
  Instance I;
  I.n = n;
  I.format = DSB_FMT_TERNARY;
  I.J.assign(size_t(n) * n, 0);
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(0, 1);
  for (int r = 0; r < n; ++r)
    for (int c = r + 1; c < n; ++c) {
      if (u(g) >= density) continue;
      int v = maxcut ? -1 : ((g() & 1) ? 1 : -1);
      I.J[size_t(r) * n + c] = v;
      I.J[size_t(c) * n + r] = v;
    }
  I.xi_default = auto_xi(I);
  return I;
}

/// Mean-variance portfolio selection with a cardinality penalty, as a QUBO
///   min_x  q x'Sx - mu'x + lambda (1'x - k)^2,   x in {0,1}^n
/// with a k-factor covariance S = F F' + D.  Mapped to Ising
///   J_ij = -Q_ij / 2 (i != j),  h_i = -(sum_j Q_ij + c_i) / 2
/// and quantized to int8 J; `hscale` maps a return vector to int16 h via
/// portfolio_field().  The covariance (J) is uploaded once; the expected
/// returns (h) change on every request.
struct Portfolio {
  Instance inst;
  std::vector<double> qrow;  // sum_j Q_ij of the quadratic part (incl. penalty)
  double scale = 1.0;        // integer units per QUBO unit
  double lambda = 0, q = 1;
  int k = 0;
};

inline Portfolio make_portfolio(int n, int factors, int k, uint64_t seed) {
  Portfolio P;
  P.k = k;
  std::mt19937_64 g(seed);
  std::normal_distribution<double> N01(0, 1);
  std::vector<double> F(size_t(n) * factors), D(n);
  for (auto& f : F) f = 0.2 * N01(g);
  for (auto& d : D) d = 0.01 + 0.02 * std::abs(N01(g));
  std::vector<double> S(size_t(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = (i == j) ? D[i] : 0.0;
      for (int f = 0; f < factors; ++f) s += F[size_t(i) * factors + f] * F[size_t(j) * factors + f];
      S[size_t(i) * n + j] = s;
    }
  double mean_offdiag = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) mean_offdiag += std::abs(S[size_t(i) * n + j]);
  mean_offdiag /= double(n) * (n - 1);
  P.q = 1.0;
  P.lambda = mean_offdiag;  // penalty comparable to a typical covariance
  std::vector<double> Q(size_t(n) * n);
  double qmax = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      Q[size_t(i) * n + j] = P.q * S[size_t(i) * n + j] + P.lambda;
      if (i != j) qmax = std::max(qmax, std::abs(Q[size_t(i) * n + j]));
    }
  // J_ij = -Q_ij/2 in integer units, |J| <= 127.
  P.scale = 127.0 / (0.5 * qmax);
  Instance& I = P.inst;
  I.n = n;
  I.format = DSB_FMT_INT8;
  I.J.assign(size_t(n) * n, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) {
        long v = std::lround(-0.5 * Q[size_t(i) * n + j] * P.scale);
        I.J[size_t(i) * n + j] = int32_t(std::max(-127L, std::min(127L, v)));
      }
  P.qrow.assign(n, 0);
  for (int i = 0; i < n; ++i) {
    double s = 0;
    for (int j = 0; j < n; ++j) s += Q[size_t(i) * n + j];
    P.qrow[i] = s;
  }
  I.xi_default = auto_xi(I);
  return P;
}

/// h for one tick of expected returns mu: c = -mu - 2 lambda k 1.
inline std::vector<int16_t> portfolio_field(const Portfolio& P, const std::vector<double>& mu) {
  const int n = P.inst.n;
  std::vector<int16_t> h(n);
  for (int i = 0; i < n; ++i) {
    double c = -mu[i] - 2.0 * P.lambda * P.k;
    long v = std::lround(-0.5 * (P.qrow[i] + c) * P.scale);
    h[i] = int16_t(std::max(-32767L, std::min(32767L, v)));
  }
  return h;
}

// ---------------------------------------------------------------------------
// packing into the wire/GPU layout of dsb_model.h

inline std::vector<uint8_t> pack(const Instance& I) {
  const int n = I.n;
  const uint32_t w = dsb_words(n);
  std::vector<uint8_t> out(dsb_model_bytes(n, I.format), 0);
  auto at = [&](int r, int c) { return I.J[size_t(r) * n + c]; };
  if (I.format == DSB_FMT_PM1) {
    uint32_t* pl = reinterpret_cast<uint32_t*>(out.data());
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) {
        if (c == r) continue;
        int v = at(r, c);
        if (v != 1 && v != -1) throw std::runtime_error("PM1 needs every off-diagonal J = +-1");
        if (v > 0) pl[size_t(r) * w + (c >> 5)] |= 1u << (c & 31);
      }
  } else if (I.format == DSB_FMT_TERNARY) {
    uint32_t* P = reinterpret_cast<uint32_t*>(out.data());
    uint32_t* M = P + size_t(n) * w;
    int32_t* rc = reinterpret_cast<int32_t*>(M + size_t(n) * w);
    for (int r = 0; r < n; ++r) {
      int32_t acc = 0;
      for (int c = 0; c < n; ++c) {
        if (c == r) continue;
        int v = at(r, c);
        if (v > 1 || v < -1) throw std::runtime_error("TERNARY needs J in {-1,0,1}");
        if (v > 0) { P[size_t(r) * w + (c >> 5)] |= 1u << (c & 31); ++acc; }
        if (v < 0) { M[size_t(r) * w + (c >> 5)] |= 1u << (c & 31); --acc; }
      }
      rc[r] = acc;
    }
  } else if (I.format == DSB_FMT_INT8) {
    const uint32_t ld = dsb_int8_ld(n);
    int8_t* J = reinterpret_cast<int8_t*>(out.data());
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) {
        if (c == r) continue;
        int v = at(r, c);
        if (v > 127 || v < -127) throw std::runtime_error("INT8 needs |J| <= 127");
        J[size_t(r) * ld + c] = int8_t(v);
      }
  } else {
    throw std::runtime_error("unknown format");
  }
  return out;
}

// ---------------------------------------------------------------------------
// the reference integrator

struct Result {
  std::vector<uint32_t> best_spins;  // ceil(n/32) words
  int64_t energy_x2 = 0;
  int best = 0;
  std::vector<int64_t> energies_x2;  // per replica
};

inline int spin(const std::vector<uint32_t>& bits, int i) {
  return ((bits[i >> 5] >> (i & 31)) & 1u) ? 1 : -1;
}

/// 2E = -s'Js - 2 h's for a packed spin vector.
inline int64_t energy_x2(const Instance& I, const int16_t* h, const std::vector<uint32_t>& bits) {
  const int n = I.n;
  int64_t e = 0;
  for (int r = 0; r < n; ++r) {
    int64_t a = 0;
    for (int c = 0; c < n; ++c) a += int64_t(I.J[size_t(r) * n + c]) * spin(bits, c);
    e -= a * spin(bits, r);
    if (h) e -= 2 * int64_t(h[r]) * spin(bits, r);
  }
  return e;
}

inline void advance(float& x, float& y, float acc, float p, float xi, float dt) {
  const float delta = 1.0f;
  const float a = delta - p;
  const float t1 = a * x;
  const float t2 = xi * acc;
  const float f = t2 - t1;
  y = y + f * dt;
  const float v = dt * y;
  x = x + v * delta;
  if (std::fabs(x) > 1.0f) {
    x = (x > 0.f) ? 1.0f : -1.0f;
    y = 0.0f;
  }
}

inline Result solve(const Instance& I, const int16_t* h, int steps, int replicas,
                    uint32_t seed, float xi, float dt) {
  const int n = I.n, B = replicas;
  if (xi <= 0.f) xi = I.xi_default;
  std::vector<float> x(size_t(n) * B), y(size_t(n) * B);
  std::vector<int> s(size_t(n) * B), s2(size_t(n) * B);
  for (int b = 0; b < B; ++b)
    for (int r = 0; r < n; ++r) {
      const float u0 = dsb_u24(seed, b, r, 0), u1 = dsb_u24(seed, b, r, 1);
      const float x0 = u0 * 0.02f - 0.01f;
      const float y0 = u1 * 0.02f - 0.01f;
      x[size_t(b) * n + r] = x0;
      y[size_t(b) * n + r] = y0;
      s[size_t(b) * n + r] = x0 > 0.f ? 1 : -1;
    }
  for (int k = 0; k < steps; ++k) {
    const float p = float(k) / float(steps - 1);
    for (int b = 0; b < B; ++b)
      for (int r = 0; r < n; ++r) {
        int64_t acc = h ? h[r] : 0;
        const int32_t* Jr = &I.J[size_t(r) * n];
        const int* sb = &s[size_t(b) * n];
        for (int c = 0; c < n; ++c) acc += int64_t(Jr[c]) * sb[c];
        float& xv = x[size_t(b) * n + r];
        float& yv = y[size_t(b) * n + r];
        advance(xv, yv, float(acc), p, xi, dt);
        s2[size_t(b) * n + r] = xv > 0.f ? 1 : -1;
      }
    s.swap(s2);
  }
  Result R;
  const uint32_t w = dsb_words(n);
  R.energies_x2.resize(B);
  int64_t best = INT64_MAX;
  std::vector<uint32_t> bits(w);
  for (int b = 0; b < B; ++b) {
    std::fill(bits.begin(), bits.end(), 0u);
    for (int r = 0; r < n; ++r)
      if (s[size_t(b) * n + r] > 0) bits[r >> 5] |= 1u << (r & 31);
    const int64_t e = energy_x2(I, h, bits);
    R.energies_x2[b] = e;
    if (e < best) {
      best = e;
      R.best = b;
      R.best_spins = bits;
    }
  }
  R.energy_x2 = best;
  return R;
}

}  // namespace dsbref
