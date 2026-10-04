/*
 * ref.cu -- resident-matrix reference for the streaming mode (PU2).
 *
 * Integrates the same instance (stream_proto.h generator, same seed) with the
 * same update kernel (state.cu) but with the bit plane of J resident in GPU
 * memory, and prints the same per-step line as the receivers:
 *     pass k ... 2E <best> chk 0x<checksum>
 * so that `diff` on the checksum files shows whether the streamed trajectory
 * is bit-identical.  With --cpu the whole integration runs on the host in
 * plain C++ (compile with -ffp-contract=off), for n up to a few 10^4.
 *
 *   ./bin/x86_64/st_ref -N 100000 -B 8 -K 20 -i random -s 42 -c results/stream/ref_1e5.chk
 *   ./bin/x86_64/st_ref -N 10000  -B 8 -K 20 -i random -s 42 --cpu -c results/stream/cpu_1e4.chk
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "stream_kernels.cuh"

using namespace st;

extern "C" int st_state_alloc(uint32_t n, uint32_t B, void **out, uint32_t **signs, int32_t **acc, long long **e2,
			      unsigned long long **chk);
extern "C" int st_state_init(void *p, uint32_t seed, cudaStream_t s);
extern "C" int st_state_update(void *p, uint32_t k, uint32_t K, float xi, float dt, int do_update, cudaStream_t s);

/* J bit plane: row r, words 0..wtot-1 */
__global__ void gen_plane_kernel(uint32_t *J, uint32_t n, uint32_t wtot, uint32_t instance, uint64_t seed)
{
	const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= (size_t)n * wtot)
		return;
	const uint32_t r = (uint32_t)(t / wtot), w = (uint32_t)(t - (size_t)r * wtot);
	J[t] = st_uword(instance, seed, r, 32u * w, n);
}

/* one warp per row: acc[b][r] = sum_w popc(J[r][w] ^ signs[b][w]) */
template <int MAXB>
__global__ void resident_pass_kernel(const uint32_t *__restrict__ J, const uint32_t *__restrict__ signs,
				     int32_t *__restrict__ acc, uint32_t n, uint32_t B, uint32_t wtot)
{
	const uint32_t lane = threadIdx.x & 31u;
	const size_t row = (size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
	if (row >= n)
		return;
	int cnt[MAXB];
#pragma unroll
	for (int b = 0; b < MAXB; ++b)
		cnt[b] = 0;
	const uint32_t *Jr = J + row * wtot;
	for (uint32_t w = lane; w < wtot; w += 32u) {
		const uint32_t u = Jr[w];
#pragma unroll
		for (int b = 0; b < MAXB; ++b)
			if (b < (int)B)
				cnt[b] += __popc(u ^ signs[(size_t)b * wtot + w]);
	}
#pragma unroll
	for (int b = 0; b < MAXB; ++b) {
		if (b < (int)B) {
			int v = cnt[b];
			v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
			v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
			v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
			v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
			v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
			if (lane == 0)
				atomicAdd(acc + (size_t)b * n + row, v);
		}
	}
}

/* ---------------------------------------------------------------- host path */
static inline int popcount32(uint32_t x) { return __builtin_popcount(x); }

static void host_advance(float &x, float &y, float acc, float p, float xi, float dt)
{
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

static int run_cpu(uint32_t n, uint32_t B, uint32_t K, uint32_t instance, uint32_t seed, float xi, float dt, FILE *chkf)
{
	const uint32_t wtot = st_wtot(n), npad = wtot * 32u;
	std::vector<uint32_t> J((size_t)n * wtot);
	for (uint32_t r = 0; r < n; ++r)
		for (uint32_t w = 0; w < wtot; ++w)
			J[(size_t)r * wtot + w] = st_uword(instance, seed, r, 32u * w, n);
	std::vector<float> x((size_t)B * npad, 0.f), y((size_t)B * npad, 0.f);
	std::vector<uint32_t> signs((size_t)B * wtot, 0u);
	unsigned long long chk = 0;
	for (uint32_t b = 0; b < B; ++b)
		for (uint32_t r = 0; r < npad; ++r) {
			if (r < n) {
				const float x0 = (2.f * st_init_u(seed, b, r, 0) + -1.f) * 0.01f;
				const float y0 = (2.f * st_init_u(seed, b, r, 1) + -1.f) * 0.01f;
				x[(size_t)b * npad + r] = x0;
				y[(size_t)b * npad + r] = y0;
				if (x0 > 0.f)
					signs[(size_t)b * wtot + (r >> 5)] |= 1u << (r & 31u);
			}
			if ((r & 31u) == 31u)
				chk ^= chk_term(b, r >> 5, signs[(size_t)b * wtot + (r >> 5)]);
		}
	printf("# init chk 0x%016llx\n", chk);
	if (chkf) fprintf(chkf, "init 0x%016llx\n", chk);
	std::vector<int32_t> a((size_t)B * n);
	for (uint32_t k = 0; k <= K; ++k) {
		const int do_update = k < K;
		for (uint32_t b = 0; b < B; ++b)
			for (uint32_t r = 0; r < n; ++r) {
				int cnt = 0;
				const uint32_t *Jr = &J[(size_t)r * wtot];
				const uint32_t *sb = &signs[(size_t)b * wtot];
				for (uint32_t w = 0; w < wtot; ++w)
					cnt += popcount32(Jr[w] ^ sb[w]);
				const int s = ((sb[r >> 5] >> (r & 31u)) & 1u) ? 1 : -1;
				a[(size_t)b * n + r] = (int32_t)n - 2 * cnt - s;
			}
		long long best = 0;
		std::vector<long long> e2(B, 0);
		const float p = host_p(k, K);
		std::vector<uint32_t> ns((size_t)B * wtot, 0u);
		for (uint32_t b = 0; b < B; ++b) {
			for (uint32_t r = 0; r < n; ++r) {
				const int s = ((signs[(size_t)b * wtot + (r >> 5)] >> (r & 31u)) & 1u) ? 1 : -1;
				e2[b] -= (long long)s * (long long)a[(size_t)b * n + r];
				if (do_update) {
					float &xv = x[(size_t)b * npad + r], &yv = y[(size_t)b * npad + r];
					host_advance(xv, yv, (float)a[(size_t)b * n + r], p, xi, dt);
					if (xv > 0.f)
						ns[(size_t)b * wtot + (r >> 5)] |= 1u << (r & 31u);
				}
			}
			if (b == 0 || e2[b] < best) best = e2[b];
		}
		if (do_update) signs.swap(ns);
		chk = 0;
		for (uint32_t b = 0; b < B; ++b)
			for (uint32_t w = 0; w < wtot; ++w)
				chk ^= chk_term(b, w, signs[(size_t)b * wtot + w]);
		printf("pass %3u %s  best 2E %lld  chk 0x%016llx  2E[", k, do_update ? "upd" : "fin", best, chk);
		for (uint32_t b = 0; b < B; ++b) printf("%s%lld", b ? "," : "", e2[b]);
		printf("]\n");
		if (chkf) fprintf(chkf, "pass %u 2E %lld chk 0x%016llx\n", k, best, chk);
	}
	return 0;
}

int main(int argc, char **argv)
{
	uint32_t n = 10000, B = 8, K = 10, instance = ST_INST_RANDOM, seed = 42, dev = 0;
	float xi = 0.f, dt = ST_DEFAULT_DT;
	bool cpu = false;
	const char *chkfile = NULL;
	for (int i = 1; i < argc; ++i) {
		std::string s = argv[i];
		auto next = [&](void) -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
		if (s == "-N") n = (uint32_t)strtoul(next(), NULL, 10);
		else if (s == "-B") B = (uint32_t)strtoul(next(), NULL, 10);
		else if (s == "-K") K = (uint32_t)strtoul(next(), NULL, 10);
		else if (s == "-i") instance = (std::string(next()) == "random") ? ST_INST_RANDOM : ST_INST_MATTIS;
		else if (s == "-s") seed = (uint32_t)strtoul(next(), NULL, 10);
		else if (s == "-x") xi = strtof(next(), NULL);
		else if (s == "-d") dt = strtof(next(), NULL);
		else if (s == "-D") dev = (uint32_t)strtoul(next(), NULL, 10);
		else if (s == "-c") chkfile = next();
		else if (s == "--cpu") cpu = true;
		else { fprintf(stderr, "st_ref -N n -B B -K steps -i random|mattis -s seed [-x xi] [-d dt] [-D gpu] [-c chkfile] [--cpu]\n"); return 1; }
	}
	if (xi <= 0.f) xi = st_default_xi(n);
	printf("# st_ref %s n %u B %u K %u instance %s seed %u xi %.9g dt %g (Mattis ground state 2E = %lld)\n",
	       cpu ? "cpu" : "gpu-resident", n, B, K, instance == ST_INST_MATTIS ? "mattis" : "random", seed, xi, dt,
	       -(long long)n * ((long long)n - 1));
	FILE *chkf = chkfile ? fopen(chkfile, "w") : NULL;
	if (cpu) {
		int r = run_cpu(n, B, K, instance, seed, xi, dt, chkf);
		if (chkf) fclose(chkf);
		return r;
	}
	cudaSetDevice((int)dev);
	const uint32_t wtot = st_wtot(n);
	uint32_t *J = NULL;
	const size_t jbytes = (size_t)n * wtot * sizeof(uint32_t);
	if (cudaMalloc((void **)&J, jbytes) != cudaSuccess) {
		fprintf(stderr, "[ref] cannot allocate %.2f GB for the resident bit plane\n", jbytes / 1e9);
		return 1;
	}
	{
		const size_t total = (size_t)n * wtot;
		gen_plane_kernel<<<(unsigned)((total + 255) / 256), 256>>>(J, n, wtot, instance, seed);
		cudaDeviceSynchronize();
	}
	void *S; uint32_t *signs; int32_t *acc; long long *e2; unsigned long long *chk;
	if (st_state_alloc(n, B, &S, &signs, &acc, &e2, &chk)) return 1;
	st_state_init(S, seed, 0);
	cudaDeviceSynchronize();
	printf("# init chk 0x%016llx\n", *chk);
	if (chkf) fprintf(chkf, "init 0x%016llx\n", *chk);
	cudaEvent_t ev0, ev1;
	cudaEventCreate(&ev0); cudaEventCreate(&ev1);
	for (uint32_t k = 0; k <= K; ++k) {
		const int do_update = k < K;
		*chk = 0ull;
		for (uint32_t b = 0; b < B; ++b) e2[b] = 0;
		cudaEventRecord(ev0);
		resident_pass_kernel<ST_MAX_B><<<(unsigned)((n + 7) / 8), 256>>>(J, signs, acc, n, B, wtot);
		st_state_update(S, k, K, xi, dt, do_update, 0);
		cudaEventRecord(ev1);
		cudaDeviceSynchronize();
		float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
		long long best = e2[0];
		for (uint32_t b = 1; b < B; ++b) if (e2[b] < best) best = e2[b];
		printf("pass %3u %s  t %9.3f s  %7.1f GB/s  best 2E %lld  chk 0x%016llx  2E[", k, do_update ? "upd" : "fin",
		       ms * 1e-3, jbytes / (ms * 1e-3) / 1e9, best, *chk);
		for (uint32_t b = 0; b < B; ++b) printf("%s%lld", b ? "," : "", e2[b]);
		printf("]\n");
		if (chkf) fprintf(chkf, "pass %u 2E %lld chk 0x%016llx\n", k, best, *chk);
	}
	if (chkf) fclose(chkf);
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) { fprintf(stderr, "[ref] CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
	return 0;
}
