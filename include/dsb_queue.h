/*
 * dsb_queue.h -- the queues between the pipeline stages, shared by the host
 * code (C) and the kernels (CUDA).
 *
 *   producer --(work ring)--> worker CTAs --(response ring)--> consumer
 *
 *   producer  DOCA server: the GPU RX warp.   proxy baseline: a CPU thread.
 *   consumer  DOCA server: the GPU TX warp.   proxy baseline: a CPU thread.
 *
 * Work ring: a ticket queue.  The producer writes item[t % cap] and then its
 * `seq` = t + 1 (release).  A worker takes ticket t = atomicAdd(claim, 1),
 * waits for item[t % cap].seq == t + 1, and after publishing its reply sets
 * item.ack = t + 1.  The producer reuses a slot only when the previous ticket
 * of that slot has been acknowledged (dsb_slot_free), so neither the item nor
 * the frame it points to is overwritten while a worker may read it.  The DOCA
 * receiver copies every request out of the NIC's cyclic receive buffer into
 * frame_pool[t % cap] for the same reason: the NIC recycles its buffer at
 * line rate regardless of how far the workers are behind.
 *
 * Response ring: a worker that has finished takes r = atomicAdd(head, 1),
 * waits until r - released < cap, writes its reply into slot r % cap and then
 * ready[r % cap] = r + 1.  The consumer sends slots strictly in r order.
 * Slots are allocated after the solve, so a long solve never blocks the
 * replies of shorter ones behind it.
 */
#ifndef DSB_QUEUE_H_
#define DSB_QUEUE_H_

#include <stdint.h>
#include "dsb_model.h"

#define DSB_WORK_CAP     4096u    /* power of two */
#define DSB_RSP_CAP      4096u    /* power of two */
#define DSB_RSP_SLOT     512u     /* bytes per reply slot (max reply is 320 B) */
#define DSB_SMEM_RESERVE 1024u    /* request header copy + reply staging */

enum dsb_work_kind {
	DSB_WORK_SOLVE = 1,
	DSB_WORK_ACK   = 2,
};

struct dsb_work_item {
	unsigned long long seq;      /* t + 1 when valid */
	unsigned long long t_rx;     /* producer timestamp (GPU globaltimer or 0) */
	const uint8_t     *frame;    /* request frame (device or host-mapped) */
	uint32_t           bytes;
	uint32_t           kind;     /* enum dsb_work_kind */
	unsigned long long ack;      /* t + 1 once the worker holding ticket t has replied */
};

/* Slot t % DSB_WORK_CAP may be (re)written for ticket t only when the ticket
 * t - DSB_WORK_CAP that used it before has been answered. */
static DSB_HD inline int dsb_slot_free(unsigned long long ack, unsigned long long t)
{
	return ack == (t >= DSB_WORK_CAP ? t - DSB_WORK_CAP + 1ull : 0ull);
}

struct dsb_stats {
	unsigned long long rx_frames;
	unsigned long long rx_solve;
	unsigned long long rx_load;
	unsigned long long rx_bad;
	unsigned long long rx_dropped;   /* work ring full */
	unsigned long long solved;
	unsigned long long solve_errors;
	unsigned long long tx_frames;
	unsigned long long tx_errors;
	unsigned long long models_ready;
	unsigned long long tx_last_error; /* doca_error_t of the last failed TX completion poll */
	unsigned long long rx_acks;       /* DOCA server: ACK work items pushed by the RX warp */
	unsigned long long tx_posted;     /* DOCA server: replies handed to the send queue (before completion) */
};

/* Everything a kernel needs, passed by value. */
struct dsb_ctx {
	volatile uint32_t          *exit_flag;
	unsigned long long         *work_claim;    /* device */
	unsigned long long         *work_done;     /* device */
	struct dsb_work_item       *items;         /* [DSB_WORK_CAP] device or host-mapped */
	uint8_t                    *frame_pool;    /* [DSB_WORK_CAP][DSB_MAX_FRAME] (DOCA server only) */
	unsigned long long         *rsp_head;      /* device */
	unsigned long long         *rsp_released;  /* written by the consumer */
	unsigned long long         *rsp_ready;     /* [DSB_RSP_CAP] */
	unsigned long long         *rsp_tend;      /* [DSB_RSP_CAP] solve-end timestamp */
	uint8_t                    *rsp_buf;       /* [DSB_RSP_CAP][DSB_RSP_SLOT] */
	struct dsb_model_desc      *models;        /* [DSB_MAX_MODELS] device */
	uint8_t                    *model_pool;    /* [DSB_MAX_MODELS][DSB_MODEL_SLOT_BYTES] device */
	struct dsb_stats           *stats;
	uint32_t                    smem_bytes;    /* dynamic shared memory per worker CTA */
	uint32_t                    rsp_on_host;   /* 1: rsp ring is host-mapped (system fences) */
};

#endif /* DSB_QUEUE_H_ */
