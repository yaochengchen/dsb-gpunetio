/*
 * state.cu -- allocation, initialization and per-pass update of the dSB state
 * (x, y, sign bitmaps, accumulators) in device memory.  Shared by the GPUNetIO
 * receiver, the kernel-stack receiver and the resident-matrix reference so the
 * three run the same kernels on the same layout.
 */
#include <stdio.h>
#include "stream_kernels.cuh"

using namespace st;

struct st_state_h {
	state S;
};

extern "C" int st_state_alloc(uint32_t n, uint32_t B, void **out, uint32_t **signs, int32_t **acc, long long **e2,
			      unsigned long long **chk)
{
	st_state_h *h = new st_state_h;
	state &S = h->S;
	S.n = n;
	S.B = B;
	S.nblk = st_nblk(n);
	S.wtot = st_wtot(n);
	const size_t npad = (size_t)npad_of(S.wtot);
	cudaError_t e = cudaSuccess;
	if ((e = cudaMalloc((void **)&S.x, sizeof(float) * B * npad)) != cudaSuccess ||
	    (e = cudaMalloc((void **)&S.y, sizeof(float) * B * npad)) != cudaSuccess ||
	    (e = cudaMalloc((void **)&S.signs, sizeof(uint32_t) * B * S.wtot)) != cudaSuccess ||
	    (e = cudaMalloc((void **)&S.acc, sizeof(int32_t) * B * (size_t)n)) != cudaSuccess ||
	    (e = cudaMallocManaged((void **)&S.e2, sizeof(long long) * B, cudaMemAttachGlobal)) != cudaSuccess ||
	    (e = cudaMallocManaged((void **)&S.chk, sizeof(unsigned long long), cudaMemAttachGlobal)) != cudaSuccess) {
		fprintf(stderr, "[state] alloc failed (n=%u B=%u): %s\n", n, B, cudaGetErrorString(e));
		return -1;
	}
	cudaMemset(S.signs, 0, sizeof(uint32_t) * B * S.wtot);
	cudaMemset(S.acc, 0, sizeof(int32_t) * B * (size_t)n);
	for (uint32_t b = 0; b < B; ++b)
		S.e2[b] = 0;
	*S.chk = 0ull;
	*out = h;
	*signs = S.signs;
	*acc = S.acc;
	*e2 = S.e2;
	*chk = S.chk;
	fprintf(stderr, "[state] n=%u B=%u: x,y %.1f MB each, signs %.1f MB, acc %.1f MB, %u column blocks, %u words/row\n",
		n, B, sizeof(float) * B * npad / 1e6, sizeof(uint32_t) * B * S.wtot / 1e6, sizeof(int32_t) * B * (double)n / 1e6,
		S.nblk, S.wtot);
	return 0;
}

extern "C" int st_state_init(void *p, uint32_t seed, cudaStream_t s)
{
	state &S = ((st_state_h *)p)->S;
	*S.chk = 0ull;
	const size_t total = (size_t)S.B * npad_of(S.wtot);
	const unsigned grid = (unsigned)((total + 255) / 256);
	init_kernel<<<grid, 256, 0, s>>>(S, seed);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int st_state_update(void *p, uint32_t k, uint32_t K, float xi, float dt, int do_update, cudaStream_t s)
{
	state &S = ((st_state_h *)p)->S;
	const size_t total = (size_t)S.B * npad_of(S.wtot);
	const unsigned grid = (unsigned)((total + 255) / 256);
	update_kernel<<<grid, 256, 0, s>>>(S, k, K, xi, dt, do_update);
	return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Device pointers of x and y ([B][npad]), for the GPUNetIO receiver's in-kernel update. */
extern "C" void st_state_xy(void *p, float **x, float **y)
{
	state &S = ((st_state_h *)p)->S;
	*x = S.x;
	*y = S.y;
}

/* Copies the sign bitmap of replica b to the host (for dumping the solution). */
extern "C" int st_state_get_signs(void *p, uint32_t b, uint32_t *host_words)
{
	state &S = ((st_state_h *)p)->S;
	return cudaMemcpy(host_words, S.signs + (size_t)b * S.wtot, sizeof(uint32_t) * S.wtot, cudaMemcpyDeviceToHost) ==
			       cudaSuccess ? 0 : -1;
}
