/*
 * recv.h -- GPUNetIO streaming receiver (solver side, PU2).
 *
 * Transport = the request service's (= the TVG receiver's): DOCA Flow steers
 * Ethertype ST_ETHERTYPE to ONE cyclic GPU receive queue; ONE receive warp
 * calls the warp-scope receive in bursts of up to ST_RX_BURST frames.  What
 * differs from the request service is only what happens to a burst: instead
 * of copying each frame into a work slot, the receive warp publishes the
 * burst (first index, count) as one work item, and one of W worker blocks
 * (512 threads) consumes the frames IN PLACE in the cyclic buffer:
 * popc(u XOR v_b) against the sign tile of the frame's column block for all B
 * replicas (shared memory), one atomicAdd per replica into acc[b][row].  The
 * worker acknowledges the item when done (ring slot reuse rule of
 * dsb_queue.h).  A PASS_END frame bumps the global pass counters.
 *
 * As in the request service the host launches NOTHING on the GPU while the
 * persistent kernels run (a kernel or a copy issued then never gets scheduled
 * behind W resident worker blocks -- seen on PU2): it only reads the counters
 * (host-mapped, written by the receive warp) and writes requests into a
 * host-mapped control block (zero-copy, like the exit flag).  End of a pass: the host waits for all PASS_ENDs and
 * writes upd_req; the idle worker blocks run the update (update_word of
 * stream_kernels.cuh, one sign word per warp, each block its share of the
 * words) and count themselves in upd_done; the host reads the energies and
 * the checksum, writes the ACK fields and ack_req; the receive warp fills the
 * PASS_ACK frame (template in the registered send buffer) and sends it with
 * the request service's reply path (txq_send + poll_completion, or wait_send
 * with -W 1), then sets ack_done.
 */
#ifndef STREAM_RECV_H_
#define STREAM_RECV_H_

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include <doca_error.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_flow.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_eth_txq_gpu_data_path.h>

#include "stream_proto.h"

#define ST_RX_PKT_NUM   16384u     /* cyclic ring slots x 4096 B = 64 MiB: exactly the request service's queue (DSB_RX_PKT_NUM x DSB_MAX_FRAME) */
#define ST_RX_BURST     2048u      /* frames per receive call: the TVG receiver's MAX_RX_NUM_PKTS (the request service's 64 is a latency choice; at line rate 64 x poll rate cannot keep up and the queue overruns) */
#define ST_RX_TIMEOUT_NS 5000u     /* the request service's: the call returns after <= 5 us without frames.  NOT 0: with timeout 0 the call blocks until frames arrive (seen on PU2: heartbeat frozen, the ACK request never seen after a pass), the TVG's "0 = non-blocking" only works because its stream never stops */
#define ST_WORK_CAP     4096u      /* burst items (power of two) */
#define ST_WORKER_THREADS 512u
#define ST_TX_SQ_DEPTH  256u

/* one burst of frames in the cyclic buffer */
struct st_work_item {
	volatile unsigned long long seq;    /* ticket + 1 when valid */
	volatile unsigned long long ack;    /* ticket + 1 when consumed */
	unsigned long long first_idx;
	unsigned long long t_rx;
	uint32_t count;
	uint32_t pad;
};

/* GPU -> host counters: host-mapped PINNED memory (cudaHostAllocMapped, the same kind as st_ctl), NOT
 * managed memory.  Written ONLY by lane 0 of the receive warp (plain stores, posted PCIe writes: the
 * warp never waits for them), read by the host from its own DRAM (no page migration).  The receive
 * warp copies the worker counters (struct st_dev, device memory) into it once per ST_EXIT_POLL_NS,
 * so every field is a snapshot at most ~0.5 ms old; the drain rule (bursts_done == rx_bursts, stable
 * over 2 ms) is exact because both come from the same snapshot.
 * Why not managed: every host read of a managed page migrates it to the CPU and the next GPU access
 * faults; with the host polling and 372 worker blocks + the receive warp writing the same page the
 * faults (and the driver's thrashing throttling) stalled the receive warp long enough for the NIC to
 * lap the 16384-slot cyclic queue (x_random_1e5_b: frames lost in 6 of 21 passes, NIC counters 0). */
struct st_stats {
	unsigned long long rx_bursts, rx_frames;     /* receive warp (rx_frames counts every published frame, PASS_END included) */
	unsigned long long rx_dropped;               /* bursts dropped: work ring full */
	unsigned long long rx_err;                   /* receive call failed -> warp exited */
	unsigned long long t_recv_ns, t_pub_ns;      /* receive warp: in the receive call / publishing */
	/* workers (copied from struct st_dev) */
	unsigned long long frames;                   /* STREAM frames consumed (all passes) */
	unsigned long long ends;                     /* PASS_END frames seen (all passes) */
	unsigned long long end_frames;               /* sum of PASS_END counts (frames the generator sent) */
	unsigned long long gen_queues;               /* from the last PASS_END */
	unsigned long long bad;                      /* wrong magic/type/size/length */
	unsigned long long tiles;                    /* tile reloads */
	unsigned long long t_work_ns;                /* worker time in bursts (sum over workers) */
	unsigned long long rx_polls;                 /* receive warp: receive calls made (heartbeat) */
	unsigned long long worker_blocks;            /* worker blocks that entered the kernel */
	unsigned long long bursts_done;              /* bursts consumed and acknowledged by the workers (== rx_bursts when the ring is drained) */
	unsigned long long upd_done;                 /* worker blocks that finished an update (cumulative: W per pass) */
	unsigned long long ack_done;                 /* receive warp: pass + 1 of the last PASS_ACK sent */
	unsigned long long tx_err;                   /* receive warp: doca_error_t of a failed ACK send (0 = none) */
	/* receive-warp diagnostics.  "pass" fields are maxima since the last PASS_ACK was sent (the host
	 * reads them after the drain, before it requests the ACK); the others are cumulative. */
	unsigned long long gap_max_ns;               /* pass: longest receive-loop iteration (a stall of the warp) */
	unsigned long long gaps_1ms, gaps_5ms;       /* cumulative: iterations longer than 1 ms / 5 ms */
	unsigned long long call_max;                 /* pass: most frames returned by one receive call */
	unsigned long long full_calls;               /* cumulative: calls that returned ST_RX_BURST frames (the warp was behind) */
	unsigned long long backlog_max;              /* pass: most frames published but not yet consumed by the workers */
	unsigned long long slots_done;               /* cumulative: frames (all types) consumed by the workers */
};

/* worker counters and the host requests the workers need: DEVICE memory (atomics in L2, no faults,
 * no PCIe).  The receive warp mirrors exit_flag / upd_req from st_ctl into it, so the 372 worker
 * blocks never read host memory. */
struct st_dev {
	unsigned long long frames, ends, end_frames, gen_queues, bad, tiles, t_work_ns;
	unsigned long long worker_blocks, bursts_done, upd_done, slots_done;
	uint32_t exit_flag, upd_req;
};

/* host -> GPU requests: host-mapped pinned memory (zero-copy), written by the host only */
struct st_ctl {
	volatile uint32_t exit_flag;
	volatile uint32_t upd_req;      /* pass + 1: run the update of pass upd_req - 1 */
	volatile uint32_t ack_req;      /* pass + 1: send the PASS_ACK of pass ack_req - 1 (fields below written first) */
	volatile uint32_t ack_lost;
	volatile uint32_t ack_frames;
	uint32_t pad[11];
};

struct st_rx_args {
	uint32_t n, B, wtot, nblk, workers;
	uint32_t K;                 /* steps: pass k < K updates the state, pass K only evaluates */
	float xi, dt;
	uint32_t *signs;            /* device [B][wtot] */
	int32_t *acc;               /* device [B][n] */
	float *x, *y;               /* device [B][npad] */
	long long *e2;              /* managed [B] */
	unsigned long long *chk;    /* managed [1] */
	struct st_work_item *items; /* device [ST_WORK_CAP] */
	unsigned long long *work_claim;   /* device */
	struct st_stats *st;        /* host-mapped (device pointer); written by receive-warp lane 0 only */
	struct st_dev *dv;          /* device */
	struct st_ctl *ctl;         /* host-mapped (device pointer) */
	/* PASS_ACK send (receive warp) */
	struct doca_gpu_eth_txq *txq;
	uint8_t *tx_frame;          /* registered GPU buffer holding the ACK template (ST_HDR_BYTES used) */
	uint32_t tx_mkey;
	uint32_t tx_mode;           /* 0 txq_send, 1 wait_send */
};

#ifdef __cplusplus
extern "C" {
#endif
int st_rx_launch(cudaStream_t s_rx, cudaStream_t s_work, struct st_rx_args a, struct doca_gpu_eth_rxq *rxq);
int st_worker_max_blocks(uint32_t smem_bytes);
#ifdef __cplusplus
}
#endif

#endif
