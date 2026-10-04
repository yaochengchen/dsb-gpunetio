/*
 * stream_kernels.cuh -- device code shared by the GPUNetIO receiver
 * (recv_kernel.cu), the kernel-stack receiver (cpu_recv.cu) and the
 * resident-matrix reference (ref.cu), so that all three evaluate the same
 * integer sums and the same IEEE update and can be compared bit for bit.
 *
 * Per replica b and row r the receivers accumulate
 *     acc[b][r] = sum over all column blocks of popc(u_r XOR v_b)
 * (one atomicAdd per frame per replica); with u = 1 <=> J = +1 and v = 1 <=>
 * s = +1 a column contributes +1 to J s when u = v, so the update kernel turns
 * the count into the exact coupling sum a_r = n - 2 acc - s_r (padding columns
 * have u = v = 0 and contribute +1 each, the diagonal is stored as u = 1 and
 * contributes +s_r; both are removed), the formula of bit_dense_multi, and applies the
 * dSB update of dsb_solve.cuh with the same intrinsics.
 */
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#include "stream_proto.h"

namespace st {

__device__ __forceinline__ unsigned long long gtimer()
{
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

/* The dSB update of dsb_solve.cuh (delta = 1), written with the same
 * round-to-nearest intrinsics so that the host reference (compiled with
 * -ffp-contract=off) reproduces it bit for bit. */
__device__ __forceinline__ void advance(float &x, float &y, float acc, float p, float xi, float dt)
{
	const float a = __fsub_rn(1.0f, p);
	const float t1 = __fmul_rn(a, x);
	const float t2 = __fmul_rn(xi, acc);
	const float f = __fsub_rn(t2, t1);
	y = __fadd_rn(y, __fmul_rn(f, dt));
	const float v = __fmul_rn(dt, y);
	x = __fadd_rn(x, v);
	if (fabsf(x) > 1.0f) {
		x = (x > 0.f) ? 1.0f : -1.0f;
		y = 0.0f;
	}
}

/* Everything the kernels need, in device memory. */
struct state {
	uint32_t n, B, wtot, nblk;
	float *x, *y;                /* [B][npad]  npad = 32 wtot */
	uint32_t *signs;             /* [B][wtot]  bit = 1 <=> s = +1; zero beyond n */
	int32_t *acc;                /* [B][n]     popcount accumulators of the current pass */
	long long *e2;               /* [B]        2E of the signs that were streamed in this pass */
	unsigned long long *chk;     /* [1]        order-independent checksum of the sign bitmaps */
};

__host__ __device__ inline uint32_t npad_of(uint32_t wtot) { return wtot * 32u; }

/* ---------------------------------------------------------------------------
 * One frame (row `row`, column block `cblk`) against the sign tiles of all B
 * replicas, by ONE WARP.  `tile` holds B x ST_WORDS words of the current
 * column block in shared memory (tile[b*ST_WORDS + w]); `payload` is the
 * 2048-word bit plane of the row (L2-coherent loads: it was written by the
 * NIC or another SM).  Lane l handles words l, l+32, ... (coalesced).
 * ------------------------------------------------------------------------- */
template <int MAXB>
__device__ __forceinline__ void accumulate_frame(const uint32_t *__restrict__ payload,
						 const uint32_t *__restrict__ tile, int32_t *__restrict__ acc,
						 uint32_t n, uint32_t B, uint32_t row, uint32_t lane)
{
	int cnt[MAXB];
#pragma unroll
	for (int b = 0; b < MAXB; ++b)
		cnt[b] = 0;
#pragma unroll 4
	for (uint32_t w = lane; w < ST_WORDS; w += 32u) {
		const uint32_t u = __ldcg(payload + w);
#pragma unroll
		for (int b = 0; b < MAXB; ++b)
			if (b < (int)B)
				cnt[b] += __popc(u ^ tile[b * ST_WORDS + w]);
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

/* Copies the sign tile of column block `cblk` for all B replicas into shared
 * memory; call from all threads of the block, followed by __syncthreads(). */
__device__ __forceinline__ void load_tile(uint32_t *__restrict__ tile, const uint32_t *__restrict__ signs,
					  uint32_t wtot, uint32_t B, uint32_t cblk, uint32_t tid, uint32_t nthreads)
{
	for (uint32_t i = tid; i < B * ST_WORDS; i += nthreads) {
		const uint32_t b = i / ST_WORDS, w = i - b * ST_WORDS;
		tile[i] = __ldcg(signs + (size_t)b * wtot + (size_t)cblk * ST_WORDS + w);
	}
}

/* ---------------------------------------------------------------------------
 * Checksum of the sign bitmaps: XOR over (b, w) of mix64(b, w, word).  Order
 * independent, so the receivers and the reference can compute it with atomics
 * and plain loops respectively and compare the printed value.
 * ------------------------------------------------------------------------- */
__host__ __device__ inline unsigned long long chk_term(uint32_t b, uint32_t w, uint32_t word)
{
	return (unsigned long long)st_mix64(((uint64_t)b << 48) ^ ((uint64_t)w << 32) ^ (uint64_t)word ^
					    0x9E3779B97F4A7C15ull);
}

/* ---------------------------------------------------------------------------
 * Initial state: x, y uniform in (-0.01, 0.01) from the hash, signs from x,
 * accumulators zero, checksum of the initial bitmaps.  One thread per (b, r)
 * over the padded rows, so a warp is one sign word.
 * ------------------------------------------------------------------------- */
static __global__ void init_kernel(state S, uint32_t seed)
{
	const uint32_t npad = npad_of(S.wtot);
	const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= (size_t)S.B * npad)
		return;
	const uint32_t b = (uint32_t)(t / npad), r = (uint32_t)(t - (size_t)b * npad);
	const bool live = r < S.n;
	float x = 0.f, y = 0.f;
	if (live) {
		/* bit_dense_multi: (2u - 1) * 0.01 */
		x = __fmul_rn(__fadd_rn(__fmul_rn(2.f, st_init_u(seed, b, r, 0)), -1.f), 0.01f);
		y = __fmul_rn(__fadd_rn(__fmul_rn(2.f, st_init_u(seed, b, r, 1)), -1.f), 0.01f);
		S.x[t] = x;
		S.y[t] = y;
		S.acc[(size_t)b * S.n + r] = 0;
	}
	const unsigned bits = __ballot_sync(0xFFFFFFFFu, live && x > 0.f);
	if ((threadIdx.x & 31u) == 0) {
		const uint32_t w = r >> 5;
		S.signs[(size_t)b * S.wtot + w] = bits;
		atomicXor(S.chk, chk_term(b, w, bits));
	}
	if (t < S.B)
		S.e2[t] = 0;
}

/* ---------------------------------------------------------------------------
 * End of a pass, ONE sign word (replica b, word w) by ONE WARP, lane = row
 * within the word: a_r = n - 2 acc - s_r; 2E += -s_r a_r (energy of the signs
 * that were streamed); if `do_update`, advance (x, y) with p = k/(K-1) and
 * publish the new signs; clear acc; checksum the (new) word.  Called by
 * update_kernel (reference, kernel-stack receiver) and, word by word, by the
 * idle worker blocks of the GPUNetIO receiver: the same per-thread arithmetic
 * and the same order-independent atomics, so the paths agree bit for bit.
 * S.chk and S.e2 must be zeroed by the host before the pass.
 * ------------------------------------------------------------------------- */
__device__ __forceinline__ void update_word(const state &S, uint32_t b, uint32_t w, uint32_t lane, uint32_t k, uint32_t K,
					    float xi, float dt, int do_update)
{
	const uint32_t npad = npad_of(S.wtot);
	const uint32_t r = w * 32u + lane;
	const size_t t = (size_t)b * npad + r;
	const bool live = r < S.n;
	const uint32_t old_word = S.signs[(size_t)b * S.wtot + w];
	float x = 0.f;
	long long e = 0;
	if (live) {
		const int s = ((old_word >> lane) & 1u) ? 1 : -1;
		const int32_t a = (int32_t)S.n - 2 * S.acc[(size_t)b * S.n + r] - s;
		e = -(long long)s * (long long)a;
		S.acc[(size_t)b * S.n + r] = 0;
		x = S.x[t];
		if (do_update) {
			float y = S.y[t];
			const float p = (K > 1u) ? __fdiv_rn((float)k, (float)(K - 1u)) : 0.f;
			advance(x, y, (float)a, p, xi, dt);
			S.x[t] = x;
			S.y[t] = y;
		}
	}
	/* warp reduce the energy term */
	e += __shfl_xor_sync(0xFFFFFFFFu, e, 16);
	e += __shfl_xor_sync(0xFFFFFFFFu, e, 8);
	e += __shfl_xor_sync(0xFFFFFFFFu, e, 4);
	e += __shfl_xor_sync(0xFFFFFFFFu, e, 2);
	e += __shfl_xor_sync(0xFFFFFFFFu, e, 1);
	const unsigned bits = do_update ? __ballot_sync(0xFFFFFFFFu, live && x > 0.f) : old_word;
	if (lane == 0) {
		atomicAdd((unsigned long long *)(S.e2 + b), (unsigned long long)e);
		if (do_update)
			S.signs[(size_t)b * S.wtot + w] = bits;
		atomicXor(S.chk, chk_term(b, w, bits));
	}
}

/* One thread per (b, r) over the padded rows (blocks of 256 = 8 words), so a
 * warp is one sign word. */
static __global__ void update_kernel(state S, uint32_t k, uint32_t K, float xi, float dt, int do_update)
{
	const uint32_t npad = npad_of(S.wtot);
	const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= (size_t)S.B * npad)
		return;
	const uint32_t b = (uint32_t)(t / npad), r = (uint32_t)(t - (size_t)b * npad);
	update_word(S, b, r >> 5, threadIdx.x & 31u, k, K, xi, dt, do_update);
}

/* Host-side float p = k/(K-1) with the same rounding as __fdiv_rn: a single
 * correctly rounded float division. */
inline float host_p(uint32_t k, uint32_t K)
{
	volatile float a = (float)k, c = (float)(K - 1u);
	return K > 1u ? (float)(a / c) : 0.f;
}

} // namespace st
