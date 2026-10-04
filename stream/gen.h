/*
 * gen.h -- GPUNetIO streaming generator (source side, PU3 / GH200 GPU 1).
 *
 * Produces the bit plane of a dense +-1 coupling matrix (stream_proto.h,
 * the companion paper's generator) row block by row block in column-block-
 * major order and sends it as raw Ethertype frames through NQ GPU send
 * queues (queue set-up copied from server_doca/doca_host.c and
 * client_doca/cdoca_host.c = the TVG's tvg_host.c).  Nothing is stored: a
 * frame is synthesized into a ring slot of the registered send buffer by one
 * of NSB synthesis blocks, published through a ready word, posted by the
 * queue's sender block, and the slot is reused once the NIC has completed it.
 *
 * Send paths (-W):
 *   2 (default) the TVG transmitter's kernel_send_packets, per queue: a block
 *     of ST_GEN_BURST = 512 threads, one frame per thread,
 *     doca_gpu_dev_eth_txq_wait_send<SHARING_CTA, SYNC_CTA, EXEC_SCOPE_BLOCK>
 *     (one reservation + one doorbell for the 512 WQEs), NOTIFY only on the
 *     last thread's frame, thread 0 poll_completion<CQ_POLL_LAST>(1) before
 *     the next burst.  The tail of a pass (< 512 frames) and the PASS_END go
 *     out with client_doca's call (same SHARING_CTA/SYNC_CTA, EXEC_SCOPE_THREAD,
 *     thread 0), NOTIFY on the last.  Timestamp 0 (no wait-on-time delay);
 *     -L paces per burst with the GPU global timer.
 *   0 / 1 the previous single-thread sender (txq_send / wait_send per frame,
 *     NOTIFY per frame): ~3 us per frame per queue, ~39 Gb/s with 4 queues.
 *
 *   ONE kernel, gen_kernel <<<NQ + 1 + NSB, ST_GEN_THREADS>>>, roles by block index:
 *     blocks 0..NQ-1   sender (posts, polls completions, releases ring slots,
 *                      inserts PASS_END, waits for the PASS_ACK)
 *     block  NQ        ACK receive warp (TVG receive); its lane 0 also copies the
 *                      device counters into the host-mapped mirror every 0.5 ms
 *     blocks NQ+1..    synthesis: take per-queue tickets, synthesize frames
 *                      into free ring slots
 *
 * Memory (lesson of the receiver, x_random_1e5_b): nothing the kernel touches
 * while frames flow is managed memory.  Counters and flags shared between
 * blocks are device memory; the host reads only the host-mapped mirror, which
 * is written by one GPU thread; the host -> GPU exit flag is host-mapped.
 */
#ifndef STREAM_GEN_H_
#define STREAM_GEN_H_

#include <stdbool.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "stream_proto.h"

#define ST_GEN_SQ_DEPTH   4096u     /* = TVG SQ_DESCRIPTOR_DEPTH */
#define ST_GEN_RX_PKTS    1024u
#define ST_GEN_RX_BURST   16u
#define ST_GEN_THREADS    512u      /* every block of gen_kernel (the sender needs one thread per frame of a burst) */
#define ST_GEN_BURST      512u      /* = TVG PACKETS_PER_BURST: frames per BLOCK-scope wait_send */
#define ST_GEN_SYNTH_THREADS ST_GEN_THREADS

struct st_gen_qstats {
	unsigned long long sent;        /* data frames completed by the NIC (all passes) */
	unsigned long long posted;      /* data frames handed to the NIC */
	unsigned long long ends;        /* PASS_END frames sent */
	unsigned long long tx_errors;
	unsigned long long last_error;
	unsigned long long wait_ring_ns;   /* sender: time with nothing ready (synthesis behind) */
	unsigned long long wait_ack_ns;    /* sender: time waiting for PASS_ACK */
	unsigned long long finished;    /* 1 when all passes are acknowledged */
	unsigned long long synth;       /* frames published by the synthesis blocks of this queue */
	unsigned long long synth_wait;  /* synthesis blocks that waited for a ring slot */
	unsigned long long synth_blocks;   /* synthesis blocks that entered the kernel */
	unsigned long long synth_ticket;   /* highest ticket taken + 1 */
	unsigned long long synth_words;    /* frames whose payload was written (before publish) */
	unsigned long long bursts;      /* sender: full 512-frame BLOCK-scope bursts posted (-W 2) */
	unsigned long long wait_tx_ns;  /* sender: time in poll_completion (NIC still sending the burst) */
	unsigned long long pad[1];
};

/* host-mapped, written only by lane 0 of the ACK warp, read by the host */
struct st_gen_mirror {
	struct st_gen_qstats qs[ST_MAX_Q];
	unsigned long long ack_pass, ack_lost, ticks;
};

struct st_gen_args {
	uint32_t nq, n, nblk, wtot, instance, passes, ring, burst, nsb;
	uint32_t sleep_ns;              /* sender: pause after each batch (crude rate limit) */
	uint32_t pace_ns_x1024;         /* sender: ns per frame x 1024 at the target payload rate (0 = unpaced) */
	uint32_t seed;
	uint32_t tx_mode;               /* 2: TVG burst send (default); 0: txq_send per frame; 1: wait_send per frame */
	uint8_t *txbuf;                 /* registered: [nq][ring + 1][ST_SLOT] (last slot: PASS_END) */
	uint8_t *hdr_tmpl;              /* device: [nq][ST_HDR_BYTES] */
	unsigned long long *ready;      /* device: [nq][ring]  = ticket+1 when the slot holds frame `ticket` */
	unsigned long long *sent_ctr;   /* device: [nq] frames released (completed) */
	unsigned long long *ticket;     /* device: [nq] synthesis tickets */
	volatile unsigned long long *ack_pass;   /* device: [1] highest acknowledged pass + 1 */
	unsigned long long *ack_lost;   /* device: [1] lost count reported by the last ACK */
	struct st_gen_qstats *qs;       /* device: [nq] */
	struct st_gen_mirror *mir;      /* host-mapped (device pointer) */
	volatile uint32_t *exit_flag;   /* host-mapped */
};

struct doca_gpu_eth_rxq;
struct doca_gpu_eth_txq;

#ifdef __cplusplus
extern "C" {
#endif
int st_gen_launch(cudaStream_t s_io, cudaStream_t s_synth, struct st_gen_args a, struct doca_gpu_eth_txq **txqs_dev,
		  struct doca_gpu_eth_rxq *ack_rxq, uint32_t mkey);
int st_gen_max_blocks(void);   /* gen_kernel blocks of ST_GEN_THREADS that can be resident at once */
#ifdef __cplusplus
}
#endif

#endif
