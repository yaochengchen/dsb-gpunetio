/*
 * stream_proto.h -- wire format and instance generator of the dSB streaming
 * mode (Section "Streaming mode" of the paper).
 *
 * Transport: raw Ethernet frames with Ethertype ST_ETHERTYPE, steered by DOCA
 * Flow to ONE cyclic GPU receive queue and received by one warp, exactly the
 * path of the request service (which is the TVG receiver's); the generator
 * sends from GPU send queues with wait_send as the TVG transmitter does.
 *
 * The coupling matrix J (dense +-1, one bit per coupling, PM1 layout of
 * dsb_model.h: bit u_ij = 1 <=> J_ij = +1, i.e. J_ij = 2 u_ij - 1; the
 * diagonal is stored as 1 and padding bits are 0) is never stored on the solver.  A
 * generator GPU produces it row block by row block and sends it as raw
 * Ethernet frames; on the solver GPU the receive warp hands each burst of frames to one
 * of W worker blocks, which consumes the frames in place in the NIC's cyclic
 * buffer and accumulates population counts into per-row sums.  One pass of the
 * stream is one dSB step for all B replicas.
 *
 * Frame = 64-byte header (Ethernet 14 + 2 pad + stream header) + 4032-byte payload = the bit plane of ONE row of J over ONE column
 * block of ST_COLS = 32256 columns (1008 32-bit words); 4096 B on the wire =
 * the request service's receive slot (DSB_MAX_FRAME), so the receive queue is
 * configured exactly as the validated one (16384 slots x 4096 B).  Note that
 * the request-service campaign itself never carried frames above 1472 B
 * (LOAD fragments); a frame that fills its slot exactly is what the TVG
 * receiver ran at 273 Gb/s (2048-B packets in 2048-B slots).  The last column
 * block is zero padded.  Order within a pass: for each column block c, rows
 * r = 0..n-1 ("column-block-major"); frame f = c*n + r goes to send/receive
 * queue f % NQ with per-queue sequence f / NQ.
 *
 * The generator spreads frames over its NQ send queues (frame f on queue
 * f % NQ with per-queue sequence f / NQ); PASS_END (one per send queue, after
 * the last data frame) carries the number of frames sent on that queue; the
 * solver answers PASS_ACK after its update, and the generator starts the next
 * pass on the ACK.
 *
 * Multi-byte stream-header fields are little-endian (both hosts are LE); only
 * the Ethertype is big-endian.
 *
 * This file is shared by the generator (PU3), the GPUNetIO receiver (PU2),
 * the kernel-stack receiver (PU2) and the resident-matrix reference (PU2).
 * It is independent of the request-service headers in ../include.
 */
#ifndef STREAM_PROTO_H_
#define STREAM_PROTO_H_

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define ST_HD __host__ __device__
#else
#define ST_HD
#endif

#define ST_MAGIC        0xD5B2u
#define ST_ETHERTYPE    0xD5B2u               /* steered by DOCA Flow exactly like the request service's 0xD5B1 */
#define ST_VERSION      2u
#define ST_HDR_BYTES    64u
#define ST_PAYLOAD      4032u                 /* bytes of bit plane per frame */
#define ST_WORDS        (ST_PAYLOAD / 4u)     /* 1008 words */
#define ST_COLS         (ST_PAYLOAD * 8u)     /* 32256 columns per block */
#define ST_FRAME        (ST_HDR_BYTES + ST_PAYLOAD)   /* 4096 B on the wire (+4 FCS) = DSB_MAX_FRAME */
#define ST_SLOT         4096u                 /* receive/send buffer stride per frame = the request service's RX slot (DSB_MAX_FRAME) */
#define ST_MAX_Q        16u                   /* generator send queues */
#define ST_MAX_B        8u                    /* sign tiles in shared memory: 8 KB each */

enum st_type {
	ST_STREAM   = 0x11,   /* one row x one column block of the bit plane */
	ST_PASS_END = 0x12,   /* last frame of a pass on this generator queue */
	ST_PASS_ACK = 0x91,   /* solver -> generator: pass consumed, next may start */
};

enum st_instance {
	ST_INST_RANDOM = 1,   /* J_ij = 2 u_ij - 1, u_ij = hash(seed, min, max) (companion paper) */
	ST_INST_MATTIS = 2,   /* J_ij = xi_i xi_j, xi = +-1 from hash: E_min = -n(n-1)/2  */
};

/* Raw Ethernet frame, as the request service: 14-byte Ethernet header, 2 pad
 * bytes, then the stream header on natural alignment.  All stream fields are
 * little-endian and 4-byte aligned relative to the frame start, which DOCA
 * places on a 64-byte boundary. */
#pragma pack(push, 1)
struct st_hdr {
	uint8_t  d_mac[6];        /*  0 */
	uint8_t  s_mac[6];        /*  6 */
	uint16_t ethertype;       /* 12  big-endian ST_ETHERTYPE */
	uint16_t magic;           /* 14  ST_MAGIC (little-endian) */
	uint8_t  type;            /* 16  enum st_type */
	uint8_t  queue;           /* 17  generator send queue */
	uint8_t  version;         /* 18  ST_VERSION */
	uint8_t  instance;        /* 19  enum st_instance (informational) */
	uint32_t pass;            /* 20  pass (= step) index, from 0 */
	uint32_t row;             /* 24  STREAM: row r; PASS_END: n; PASS_ACK: frames lost (0 = ok) */
	uint32_t seq;             /* 28  STREAM: per-queue sequence; PASS_END: frames sent on this queue; ACK: frames received */
	uint16_t cblk;            /* 32  STREAM: column block; PASS_END: number of column blocks */
	uint16_t gen_queues;      /* 34  PASS_END: number of generator send queues (PASS_ENDs per pass) */
	uint8_t  reserved[28];    /* 36..63 */
	/* 64: payload (STREAM only) */
};
#pragma pack(pop)

#if defined(__cplusplus)
static_assert(sizeof(struct st_hdr) == ST_HDR_BYTES, "stream header is 64 bytes");
#else
_Static_assert(sizeof(struct st_hdr) == ST_HDR_BYTES, "stream header is 64 bytes");
#endif

/* ------------------------------------------------------------- instances
 * The instance generator is the one of the one-bit dense path of the
 * companion paper (apps/bit_dense_multi.cu of dSB_GPUs_3Million_N), copied
 * verbatim so that a seed denotes the same matrix there and here:
 *   random:  u_ij = fmix32(fmix32(hi ^ seed) + lo * 0x9e3779b9) & 1,  lo/hi = min/max(i,j)
 *   Mattis:  u_ij = NOT(xibit(i) XOR xibit(j)),  xibit(i) = fmix32(i ^ fmix32(seed + 0x6d2b79f5)) & 1
 *   bit 1 <=> J_ij = +1; the DIAGONAL IS STORED AS +1 (u_ii = 1) and removed in
 *   the sum: a_i = n - 2 popc(u_i XOR v) - s_i; columns >= n are 0.
 *   initial state (replica 0): x,y = (2u - 1) * 0.01 with
 *   u = (fmix32(fmix32(2i + which) ^ (seed * 0x27d4eb2d)) >> 8) * 2^-24, which = 0 (x), 1 (y);
 *   replica b > 0 uses seed + b for the initial state only.
 *   defaults: xi = 0.5/sqrt(n), dt = 1.1.
 */
static ST_HD inline uint32_t st_nblk(uint32_t n) { return (n + ST_COLS - 1u) / ST_COLS; }
static ST_HD inline uint32_t st_wtot(uint32_t n) { return st_nblk(n) * ST_WORDS; }   /* words per row / per sign bitmap */

/* 64-bit mixer, used only for the order-independent bitmap checksum */
static ST_HD inline uint64_t st_mix64(uint64_t z)
{
	z ^= z >> 29;
	z *= 0xBF58476D1CE4E5B9ull;
	z ^= z >> 32;
	z *= 0x94D049BB133111EBull;
	z ^= z >> 29;
	return z;
}

static ST_HD inline uint32_t st_fmix32(uint32_t h)
{
	h ^= h >> 16;
	h *= 0x85ebca6bu;
	h ^= h >> 13;
	h *= 0xc2b2ae35u;
	h ^= h >> 16;
	return h;
}

static ST_HD inline uint32_t st_xibit(uint32_t seed, uint32_t i)
{
	return st_fmix32(i ^ st_fmix32(seed + 0x6d2b79f5u)) & 1u;
}

/* u_ij (bit 1 <=> J = +1); diagonal 1; the caller masks columns >= n */
static ST_HD inline uint32_t st_jbit(uint32_t instance, uint32_t seed, uint32_t i, uint32_t j)
{
	if (i == j)
		return 1u;
	if (instance == ST_INST_MATTIS)
		return (st_xibit(seed, i) ^ st_xibit(seed, j)) ^ 1u;
	const uint32_t lo = i < j ? i : j, hi = i < j ? j : i;
	return st_fmix32(st_fmix32(hi ^ seed) + lo * 0x9e3779b9u) & 1u;
}

/* One word of the bit plane u for row i, columns j0..j0+31 (j0 multiple of 32). */
static ST_HD inline uint32_t st_uword(uint32_t instance, uint32_t seed, uint32_t i, uint32_t j0, uint32_t n)
{
	if (j0 >= n)
		return 0u;
	const uint32_t cnt = (n - j0) < 32u ? (n - j0) : 32u;
	uint32_t w = 0;
	if (instance == ST_INST_MATTIS) {
		/* one xi bit per column; the diagonal comes out as 1 by itself */
		for (uint32_t b = 0; b < cnt; ++b)
			w |= ((st_xibit(seed, i) ^ st_xibit(seed, j0 + b)) ^ 1u) << b;
		return w;
	}
	for (uint32_t b = 0; b < cnt; ++b)
		w |= st_jbit(instance, seed, i, j0 + b) << b;
	return w;
}

/* Initial state of the companion generator, replica 0 == bit_dense_multi. */
static ST_HD inline float st_init_u(uint32_t seed, uint32_t replica, uint32_t i, uint32_t which)
{
	const uint32_t sd = seed + replica;
	return (float)(st_fmix32(st_fmix32(i * 2u + which) ^ (sd * 0x27d4eb2du)) >> 8) * 5.9604644775390625e-8f;
}

/* Default coupling gain of the companion generator, xi = 0.5/sqrt(n) (float
 * of the double), and its default time step.  Computed on the host and passed
 * to every kernel, so all paths use the same values. */
static inline float st_default_xi(uint32_t n)
{
	return (float)(0.5 / sqrt((double)n));
}
#define ST_DEFAULT_DT 1.1f

#endif /* STREAM_PROTO_H_ */
