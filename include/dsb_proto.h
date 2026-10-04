/*
 * dsb_proto.h -- wire format of the dSB optimization service.
 *
 * Every message is one raw Ethernet frame with Ethertype DSB_ETHERTYPE.
 * The DOCA GPUNetIO server, the CPU+GPU baseline server and the client all
 * speak exactly this format, so the three configurations are compared on the
 * same bytes on the wire.
 *
 * Layout: a 64-byte header (Ethernet header + 2 pad bytes + 48 bytes of dSB
 * header) followed by a payload.  The 2 pad bytes put every dSB field on its
 * natural alignment relative to the frame start; the DOCA receive buffer
 * places frames on 64-byte boundaries, so the GPU can read them with aligned
 * loads.  Multi-byte dSB fields are little-endian (both GH200/Grace and x86
 * are little-endian); only the Ethertype is big-endian.
 *
 *   type                       payload
 *   ---------------------------------------------------------------------
 *   DSB_MSG_LOAD      (0x02)   one fragment of a packed coupling model
 *   DSB_MSG_SOLVE     (0x01)   linear field h[n] (int16), optional
 *   DSB_MSG_LOAD_ACK  (0x82)   none
 *   DSB_MSG_RESULT    (0x81)   best spin configuration, ceil(n/32) words
 *
 * The coupling matrix J is uploaded once per model (LOAD) and stays resident
 * in GPU memory; each SOLVE carries only the per-request linear field.  This
 * is the "slowly varying J, fast varying h" pattern of streaming portfolio or
 * scheduling problems: a covariance/conflict matrix changes rarely, the
 * returns/priorities change on every tick.
 */
#ifndef DSB_PROTO_H_
#define DSB_PROTO_H_

#include <stdint.h>

#if defined(__CUDACC__)
#define DSB_HD __host__ __device__
#else
#define DSB_HD
#endif

#define DSB_ETHERTYPE        0xD5B1
#define DSB_PROTO_VERSION    1
#define DSB_HDR_BYTES        64
#define DSB_MAX_FRAME        4096   /* server RX slot (max frame the server accepts) */
#define DSB_MAX_PAYLOAD      (DSB_MAX_FRAME - DSB_HDR_BYTES)
/* LOAD fragment payload: 64 + 1408 = 1472-byte frames, so a model upload fits a
 * plain 1500-byte MTU.  A SOLVE frame is 64 + 2n bytes (n <= 704 fits 1500);
 * only n > 704 needs a larger MTU on the client interface. */
#define DSB_LOAD_FRAG_BYTES  1408   /* multiple of 16 */
#define DSB_MIN_FRAME        64

#define DSB_MAX_N            2048   /* hard protocol limit; the GPU plan is tighter */
#define DSB_MAX_MODELS       64

enum dsb_msg_type {
	DSB_MSG_SOLVE    = 0x01,
	DSB_MSG_LOAD     = 0x02,
	DSB_MSG_RESULT   = 0x81,
	DSB_MSG_LOAD_ACK = 0x82,
};

/* Coupling storage formats (see dsb_model.h for the exact byte layout). */
enum dsb_format {
	DSB_FMT_PM1     = 1,  /* dense, every off-diagonal J_ij = +-1: one bit plane   */
	DSB_FMT_TERNARY = 2,  /* J_ij in {-1,0,+1}: two bit planes + row constants    */
	DSB_FMT_INT8    = 3,  /* J_ij in [-127,127]: dense int8 rows                   */
};

enum dsb_status {
	DSB_OK              = 0,
	DSB_ERR_NO_MODEL    = 1,  /* model id not loaded (or still loading)    */
	DSB_ERR_SHAPE       = 2,  /* n / replicas / steps outside the GPU plan  */
	DSB_ERR_BAD_MSG     = 3,
	DSB_ERR_LOAD_RANGE  = 4,  /* fragment outside the model slot          */
};

#pragma pack(push, 1)

/* Common first 32 bytes of every message. */
struct dsb_hdr_common {
	uint8_t  d_mac[6];        /*  0 */
	uint8_t  s_mac[6];        /*  6 */
	uint16_t ethertype;       /* 12  big-endian DSB_ETHERTYPE */
	uint8_t  pad0[2];         /* 14 */
	uint8_t  version;         /* 16 */
	uint8_t  type;            /* 17  enum dsb_msg_type */
	uint16_t flags;           /* 18  echoed in the reply; clients send 0 */
	uint32_t req_id;          /* 20  echoed in the reply */
	uint64_t client_ts;       /* 24  opaque to the server, echoed */
};

struct dsb_solve_req {
	struct dsb_hdr_common c;  /*  0..31 */
	uint16_t model_id;        /* 32 */
	uint16_t n;               /* 34  must equal the model's n */
	uint16_t steps;           /* 36  K >= 2 */
	uint16_t replicas;        /* 38  B >= 1 */
	uint32_t seed;            /* 40 */
	float    xi;              /* 44  coupling gain; <= 0 -> model default */
	float    dt;              /* 48 */
	uint16_t h_format;        /* 50  0: no field, 1: int16[n] payload */
	uint16_t reserved0;       /* 52 */
	uint8_t  reserved1[8];    /* 56 */
	/* 64: int16 h[n] */
};

struct dsb_result {
	struct dsb_hdr_common c;  /*  0..31 */
	uint16_t model_id;        /* 32 */
	uint16_t n;               /* 34 */
	uint16_t status;          /* 36  enum dsb_status */
	uint16_t best_replica;    /* 38 */
	int64_t  energy_x2;       /* 40  2*E of the best replica, E = -1/2 s'Js - h's */
	uint32_t queue_ns;        /* 48  server: request seen -> solve start */
	uint32_t solve_ns;        /* 52  server: solve start -> solve end */
	uint32_t tx_ns;           /* 56  server: solve end -> reply handed to NIC */
	uint32_t worker;          /* 60  which worker CTA / stream served it */
	/* 64: uint32 spins[ceil(n/32)], bit (i%32) of word i/32 set <=> s_i = +1 */
};

struct dsb_load_frag {
	struct dsb_hdr_common c;  /*  0..31 */
	uint16_t model_id;        /* 32 */
	uint16_t n;               /* 34 */
	uint8_t  format;          /* 36  enum dsb_format */
	uint8_t  reserved0;       /* 37 */
	uint16_t frag_idx;        /* 38 */
	uint32_t total_bytes;     /* 40  size of the packed model */
	uint32_t offset;          /* 44  byte offset of this fragment */
	uint32_t frag_bytes;      /* 48  bytes of payload in this frame (multiple of 4) */
	uint32_t generation;      /* 52  new value starts a fresh upload of this id */
	float    xi_default;      /* 56  used when a SOLVE sends xi <= 0 */
	uint16_t frag_count;      /* 60 */
	uint16_t reserved1;       /* 62 */
	/* 64: payload */
};

struct dsb_load_ack {
	struct dsb_hdr_common c;  /*  0..31 */
	uint16_t model_id;        /* 32 */
	uint16_t n;               /* 34 */
	uint16_t status;          /* 36 */
	uint16_t reserved0;       /* 38 */
	uint32_t generation;      /* 40 */
	uint32_t total_bytes;     /* 44 */
	uint8_t  reserved1[16];   /* 48 */
};

#pragma pack(pop)

#if defined(__cplusplus)
#define DSB_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define DSB_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
DSB_STATIC_ASSERT(sizeof(struct dsb_hdr_common) == 32, "common header");
DSB_STATIC_ASSERT(sizeof(struct dsb_solve_req) == DSB_HDR_BYTES, "solve header");
DSB_STATIC_ASSERT(sizeof(struct dsb_result) == DSB_HDR_BYTES, "result header");
DSB_STATIC_ASSERT(sizeof(struct dsb_load_frag) == DSB_HDR_BYTES, "load header");
DSB_STATIC_ASSERT(sizeof(struct dsb_load_ack) == DSB_HDR_BYTES, "ack header");

/* Bytes of a SOLVE frame / RESULT frame for a given n. */
static DSB_HD inline uint32_t dsb_solve_frame_bytes(uint32_t n, uint32_t h_format)
{
	uint32_t b = DSB_HDR_BYTES + (h_format ? 2u * n : 0u);
	b = (b + 3u) & ~3u;
	return b < DSB_MIN_FRAME ? DSB_MIN_FRAME : b;
}

static DSB_HD inline uint32_t dsb_result_frame_bytes(uint32_t n)
{
	return DSB_HDR_BYTES + 4u * ((n + 31u) / 32u);
}

#endif /* DSB_PROTO_H_ */
