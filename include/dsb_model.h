/*
 * dsb_model.h -- packed coupling layouts and the per-request shared-memory
 * plan.  C99/C++/CUDA compatible; everything is a static inline function so
 * the client, both servers and the kernels agree on every byte.
 *
 * Packed model (what LOAD fragments carry, and what lives in GPU memory):
 *
 *   W = ceil(n/32)   (words per bit row)
 *
 *   PM1      plane[n][W] uint32      bit c of row r set <=> J_rc = +1, clear <=> -1
 *                                    (diagonal bit clear, padding bits clear)
 *   TERNARY  P[n][W] | M[n][W] uint32 | rowconst[n] int32
 *                                    P: J_rc = +1, M: J_rc = -1,
 *                                    rowconst[r] = |P_r| - |M_r|
 *   INT8     J[n][L] int8, L = roundup(n, 16); J_rr = 0, padding 0
 *
 * Per-request plan inside one worker CTA (all of it in shared memory):
 *
 *   npad = roundup(n, 32)            one warp = 32 consecutive rows of one replica
 *   bit formats: Wp = 4*odd >= W     row stride of J and of the sign rows, so that
 *                                    8 lanes' 16-byte loads hit distinct banks
 *   INT8:        Ls = 16*odd >= npad row stride (bytes) of J and of sign rows
 *
 *   PM1      J[npad][Wp]  + S[2][B][Wp]                    (uint32)
 *   TERNARY  P,M[npad][Wp] + rc[npad] + S[2][B][Wp]
 *   INT8     J[npad][Ls]  + S[2][B][Ls]                    (int8)
 *   common   h[npad] int32, e2[B] int64, scratch
 *
 * The state (x, y) never touches memory: each thread keeps DSB_PT (row,
 * replica) pairs in registers, so B * npad <= DSB_THREADS * DSB_PT.
 */
#ifndef DSB_MODEL_H_
#define DSB_MODEL_H_

#include <stdint.h>
#include <stddef.h>
#include "dsb_proto.h"


#define DSB_THREADS   256   /* threads per worker CTA */
#define DSB_PT        16    /* (row, replica) pairs held in registers per thread */
#define DSB_MAX_PAIRS (DSB_THREADS * DSB_PT)
#define DSB_MAX_REPLICAS 128

static DSB_HD inline uint32_t dsb_words(uint32_t n) { return (n + 31u) / 32u; }
static DSB_HD inline uint32_t dsb_npad(uint32_t n) { return (n + 31u) & ~31u; }
static DSB_HD inline uint32_t dsb_int8_ld(uint32_t n) { return (n + 15u) & ~15u; }

/* 4 * odd, >= w */
static DSB_HD inline uint32_t dsb_wpad(uint32_t w)
{
	uint32_t wp = (w + 3u) & ~3u;
	if (((wp >> 2) & 1u) == 0u) wp += 4u;
	return wp;
}

/* 16 * odd bytes, >= npad */
static DSB_HD inline uint32_t dsb_int8_lds(uint32_t npad)
{
	uint32_t l = (npad + 15u) & ~15u;
	if (((l >> 4) & 1u) == 0u) l += 16u;
	return l;
}

static DSB_HD inline uint32_t dsb_model_bytes(uint32_t n, uint32_t format)
{
	uint32_t w = dsb_words(n);
	switch (format) {
	case DSB_FMT_PM1:     return 4u * n * w;
	case DSB_FMT_TERNARY: return 8u * n * w + 4u * n;
	case DSB_FMT_INT8:    return n * dsb_int8_ld(n);
	default:              return 0u;
	}
}

/* Byte offsets of the regions of one request's shared-memory plan. */
struct dsb_smem_plan {
	uint32_t n, npad, words, wpad, lds, replicas, format;
	uint32_t off_j;      /* J (PM1/INT8) or P (TERNARY) */
	uint32_t off_m;      /* M (TERNARY) */
	uint32_t off_rc;     /* rowconst (TERNARY) */
	uint32_t off_s;      /* sign rows, two buffers */
	uint32_t s_buf;      /* bytes of one sign buffer */
	uint32_t off_h;      /* int32 h[npad] */
	uint32_t off_e;      /* int64 e2[B] */
	uint32_t bytes;      /* total */
};

static DSB_HD inline uint32_t dsb_align16(uint32_t x) { return (x + 15u) & ~15u; }

/* Returns 0 when the request fits `smem_limit` bytes and the register budget. */
static DSB_HD inline int dsb_make_plan(uint32_t n, uint32_t replicas, uint32_t format,
				       uint32_t smem_limit, struct dsb_smem_plan *p)
{
	uint32_t off = 0;
	if (n < 2u || n > DSB_MAX_N || replicas < 1u || replicas > DSB_MAX_REPLICAS)
		return -1;
	p->n = n;
	p->npad = dsb_npad(n);
	p->words = dsb_words(n);
	p->wpad = dsb_wpad(p->words);
	p->lds = dsb_int8_lds(p->npad);
	p->replicas = replicas;
	p->format = format;
	if (replicas * p->npad > (uint32_t)DSB_MAX_PAIRS)
		return -2;

	p->off_j = 0;
	p->off_m = 0;
	p->off_rc = 0;
	switch (format) {
	case DSB_FMT_PM1:
		off = 4u * p->npad * p->wpad;
		p->s_buf = 4u * replicas * p->wpad;
		break;
	case DSB_FMT_TERNARY:
		p->off_m = 4u * p->npad * p->wpad;
		p->off_rc = 2u * p->off_m;
		off = p->off_rc + 4u * p->npad;
		p->s_buf = 4u * replicas * p->wpad;
		break;
	case DSB_FMT_INT8:
		off = p->npad * p->lds;
		p->s_buf = dsb_align16(replicas * p->lds);
		break;
	default:
		return -3;
	}
	off = dsb_align16(off);
	p->off_s = off;
	off += 2u * p->s_buf;
	p->off_h = dsb_align16(off);
	off = p->off_h + 4u * p->npad;
	p->off_e = dsb_align16(off);
	off = p->off_e + 8u * replicas;
	p->bytes = dsb_align16(off);
	return p->bytes <= smem_limit ? 0 : -4;
}

#define DSB_MODEL_SLOT_BYTES (512u * 1024u)   /* per model id in the GPU pool */
#define DSB_MAX_FRAGS 384u  /* >= ceil(DSB_MODEL_SLOT_BYTES / DSB_LOAD_FRAG_BYTES) = 373 */
DSB_STATIC_ASSERT(DSB_MAX_FRAGS * DSB_LOAD_FRAG_BYTES >= DSB_MODEL_SLOT_BYTES, "fragment bitmap too small");
DSB_STATIC_ASSERT(DSB_MAX_FRAGS % 32u == 0u, "fragment bitmap words");

/* Model descriptor kept in GPU memory, one per model id. */
struct dsb_model_desc {
	uint32_t ready_gen;    /* == generation when the upload is complete; 0 = not ready */
	uint32_t generation;   /* generation currently being uploaded */
	uint32_t frags_seen;   /* distinct fragments received for `generation` */
	uint32_t total_bytes;
	uint16_t n;
	uint8_t  format;
	uint8_t  pad0;
	float    xi_default;
	uint32_t frag_count;
	uint32_t frag_bits[DSB_MAX_FRAGS / 32];  /* which fragments have arrived */
};


/* Deterministic initial state, shared by the GPU and the CPU reference.
 * splitmix64 of (seed, replica, row, stream); 24-bit uniform in [0,1). */
static DSB_HD inline uint64_t dsb_splitmix64(uint64_t z)
{
	z += 0x9E3779B97F4A7C15ull;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

static DSB_HD inline float dsb_u24(uint32_t seed, uint32_t replica, uint32_t row, uint32_t stream)
{
	uint64_t key = ((uint64_t)seed << 32) ^ ((uint64_t)replica << 20) ^ ((uint64_t)row << 2) ^ stream;
	return (float)(uint32_t)(dsb_splitmix64(key) >> 40) * (1.0f / 16777216.0f);
}

#endif /* DSB_MODEL_H_ */
