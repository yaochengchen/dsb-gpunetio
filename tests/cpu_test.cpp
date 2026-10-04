// CPU-only checks: packing, the bit/int8 coupling identities used by the GPU
// kernel (evaluated here on the packed layout), the QUBO->Ising mapping, and
// that the reference integrator actually optimizes.  Build: make cpu_test
#include <cassert>
#include <cstdio>
#include <random>
#include "dsb_ref.hpp"

using namespace dsbref;

static int popc(uint32_t v) { return __builtin_popcount(v); }

// Same formulas as Coupling<FMT>::sum in dsb_solve.cuh, on the packed model.
static int coupling_from_packed(const Instance& I, const std::vector<uint8_t>& pk,
                                const std::vector<uint32_t>& S, int r) {
  const int n = I.n, W = dsb_words(n);
  if (I.format == DSB_FMT_PM1) {
    const uint32_t* J = reinterpret_cast<const uint32_t*>(pk.data());
    int pop = 0;
    for (int w = 0; w < W; ++w) pop += popc(J[r * W + w] ^ S[w]);
    const int sr = ((S[r >> 5] >> (r & 31)) & 1) ? 1 : -1;
    return n - 2 * pop + sr;
  }
  if (I.format == DSB_FMT_TERNARY) {
    const uint32_t* P = reinterpret_cast<const uint32_t*>(pk.data());
    const uint32_t* M = P + n * W;
    const int32_t* rc = reinterpret_cast<const int32_t*>(M + n * W);
    int d = 0;
    for (int w = 0; w < W; ++w) d += popc(P[r * W + w] & S[w]) - popc(M[r * W + w] & S[w]);
    return 2 * d - rc[r];
  }
  const int ld = dsb_int8_ld(n);
  const int8_t* J = reinterpret_cast<const int8_t*>(pk.data());
  int acc = 0;
  for (int c = 0; c < n; ++c) acc += J[r * ld + c] * spin(S, c);
  return acc;
}

static void check_identity(const Instance& I, uint64_t seed) {
  auto pk = pack(I);
  assert(pk.size() == dsb_model_bytes(I.n, I.format));
  std::mt19937_64 g(seed);
  std::vector<uint32_t> S(dsb_words(I.n), 0);
  for (int t = 0; t < 5; ++t) {
    std::fill(S.begin(), S.end(), 0u);
    for (int i = 0; i < I.n; ++i)
      if (g() & 1) S[i >> 5] |= 1u << (i & 31);
    for (int r = 0; r < I.n; ++r) {
      long exact = 0;
      for (int c = 0; c < I.n; ++c) exact += I.J[size_t(r) * I.n + c] * spin(S, c);
      const int got = coupling_from_packed(I, pk, S, r);
      if (got != exact) {
        std::printf("FAIL format %d n %d row %d: %d vs %ld\n", I.format, I.n, r, got, exact);
        std::exit(1);
      }
    }
  }
}

static void check_qubo_mapping() {
  // E_ising(s) - cost(x) must be constant over all x for a random small QUBO.
  const int n = 6;
  std::mt19937_64 g(7);
  std::normal_distribution<double> N01;
  std::vector<double> Q(n * n), c(n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j) Q[i * n + j] = Q[j * n + i] = N01(g);
  for (auto& v : c) v = N01(g);
  double first = 0;
  for (int m = 0; m < (1 << n); ++m) {
    double cost = 0, E = 0;
    for (int i = 0; i < n; ++i) {
      const int xi = (m >> i) & 1;
      cost += c[i] * xi;
      for (int j = 0; j < n; ++j) cost += Q[i * n + j] * xi * ((m >> j) & 1);
    }
    for (int i = 0; i < n; ++i) {
      const int si = ((m >> i) & 1) ? 1 : -1;
      double rs = 0;
      for (int j = 0; j < n; ++j) rs += Q[i * n + j];
      const double h = -0.5 * (rs + c[i]);
      E -= h * si;
      for (int j = 0; j < n; ++j)
        if (j != i) E -= 0.5 * (-0.5 * Q[i * n + j]) * si * (((m >> j) & 1) ? 1 : -1);
    }
    if (m == 0) first = E - cost;
    if (std::abs(E - cost - first) > 1e-9) {
      std::printf("FAIL qubo mapping\n");
      std::exit(1);
    }
  }
}

int main() {
  for (int n : {2, 31, 32, 33, 100, 257}) {
    check_identity(make_pm1(n, n), 1);
    check_identity(make_ternary(n, 0.1, false, n), 2);
    check_identity(make_ternary(n, 0.3, true, n + 1), 3);
    check_identity(make_portfolio(n, 3, std::max(1, n / 10), n).inst, 4);
  }
  std::printf("coupling identities: ok\n");
  check_qubo_mapping();
  std::printf("qubo->ising mapping: ok\n");

  // The reference should beat random spins by a wide margin.
  for (int fmt = 1; fmt <= 3; ++fmt) {
    const int n = 256;
    Instance I = fmt == 1 ? make_pm1(n, 11) : fmt == 2 ? make_ternary(n, 0.05, true, 11)
                                                       : make_portfolio(n, 4, 20, 11).inst;
    std::vector<int16_t> h(n, 0);
    if (fmt == 3) {
      auto P = make_portfolio(n, 4, 20, 11);
      std::vector<double> mu(n);
      std::mt19937_64 g(5);
      std::normal_distribution<double> N01;
      for (auto& m : mu) m = 0.05 * N01(g);
      h = portfolio_field(P, mu);
    }
    Result R = solve(I, h.data(), 200, 8, 42, 0.f, 1.0f);
    std::mt19937_64 g(99);
    double rnd = 0;
    for (int t = 0; t < 20; ++t) {
      std::vector<uint32_t> b(dsb_words(n));
      for (auto& w : b) w = uint32_t(g());
      if (n % 32) b.back() &= (1u << (n % 32)) - 1;
      rnd += energy_x2(I, h.data(), b) / 20.0;
    }
    const int64_t check = energy_x2(I, h.data(), R.best_spins);
    std::printf("format %d: dSB 2E = %lld (recomputed %lld), random 2E = %.0f\n", fmt,
                (long long)R.energy_x2, (long long)check, rnd);
    if (check != R.energy_x2 || R.energy_x2 >= rnd) {
      std::printf("FAIL reference\n");
      return 1;
    }
  }
  std::printf("reference integrator: ok\n");
  return 0;
}
