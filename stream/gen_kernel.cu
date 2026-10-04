/*
 * gen_kernel.cu -- synthesis blocks, sender threads and ACK receive warp of
 * the streaming generator.  See gen.h.
 *
 * Send path -W 2 (default) = TVG generator/device/tvg_kernel.cu
 * kernel_send_packets: 512 threads, wait_send<SHARING_CTA, SYNC_CTA,
 * EXEC_SCOPE_BLOCK>(ts), NOTIFY on the last thread only, thread 0
 * poll_completion<CQ_POLL_LAST>(1) after every burst.  Remainders and the
 * PASS_END: client_doca/cdoca_kernel.cu's wait_send<SHARING_CTA, SYNC_CTA,
 * EXEC_SCOPE_THREAD>(ts = 0) by thread 0.  -W 0/1: the previous per-frame
 * sender.  Receive path (ACK) = the TVG receiver's warp-scope receive.
 */
#include <stdio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "gen.h"

#define ST_EXIT_POLL_NS 500000ull

/* The sender is one thread of a 32-thread block; the other lanes idle in a
 * loop instead of exiting, so the CTA-scope send/poll calls see a full block
 * as in client_doca/cdoca_kernel.cu. */

__device__ __forceinline__ unsigned long long gtimer()
{
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

/* frames of queue q per pass: #{f in [0, F) : f % nq == q} */
__host__ __device__ inline unsigned long long st_count_q(unsigned long long F, uint32_t nq, uint32_t q)
{
	return F > q ? (F - q + nq - 1ull) / nq : 0ull;
}

/* ------------------------------------------------------------------ synthesis */
/* One synthesis block (all ST_GEN_THREADS threads); `sb` is its index among the synthesis blocks.
 * Its counters (qs) are device memory: per-frame atomics stay in L2. */
__device__ static void synth_block(struct st_gen_args a, uint32_t sb)
{
	__shared__ unsigned long long t_sh;
	__shared__ int stop;
	const uint32_t q = sb % a.nq;
	const uint32_t tid = threadIdx.x;
	const unsigned long long F = (unsigned long long)a.nblk * a.n;
	const unsigned long long count_q = st_count_q(F, a.nq, q);
	const unsigned long long total_q = count_q * a.passes;
	volatile unsigned long long *sent = a.sent_ctr + q;
	unsigned long long next_check = 0ull;
	const size_t qstride = (size_t)(a.ring + 1u) * ST_SLOT;
	if (tid == 0)
		atomicAdd(&a.qs[q].synth_blocks, 1ull);

	for (;;) {
		if (tid == 0) {
			stop = 0;
			const unsigned long long t = atomicAdd(a.ticket + q, 1ull);
			atomicMax(&a.qs[q].synth_ticket, t + 1ull);
			if (t >= total_q) {
				stop = 1;
			} else {
				if (*sent + a.ring <= t)
					atomicAdd(&a.qs[q].synth_wait, 1ull);
				while (*sent + a.ring <= t) {   /* wait for the ring slot */
					if (gtimer() >= next_check) {
						next_check = gtimer() + ST_EXIT_POLL_NS;
						if (*a.exit_flag) { stop = 1; break; }
					}
					__nanosleep(500);
				}
			}
			t_sh = t;
		}
		__syncthreads();
		if (stop)
			break;
		const unsigned long long t = t_sh;
		const uint32_t pass = (uint32_t)(t / count_q);
		const unsigned long long seq = t - (unsigned long long)pass * count_q;
		const unsigned long long f = seq * a.nq + q;            /* column-block-major frame index */
		const uint32_t cblk = (uint32_t)(f / a.n), row = (uint32_t)(f - (unsigned long long)cblk * a.n);
		const uint32_t slot = (uint32_t)(t % a.ring);
		uint8_t *fr = a.txbuf + (size_t)q * qstride + (size_t)slot * ST_SLOT;
		uint32_t *pl = (uint32_t *)(fr + ST_HDR_BYTES);
		const uint32_t j0_base = cblk * ST_COLS;
		for (uint32_t w = tid; w < ST_WORDS; w += ST_GEN_SYNTH_THREADS)
			pl[w] = st_uword(a.instance, a.seed, row, j0_base + 32u * w, a.n);
		if (tid < 16) {   /* 64-byte header template of this queue */
			const uint32_t *src = (const uint32_t *)(a.hdr_tmpl + (size_t)q * ST_HDR_BYTES);
			((uint32_t *)fr)[tid] = src[tid];
		}
		__syncthreads();
		if (tid == 0) {
			atomicAdd(&a.qs[q].synth_words, 1ull);
			struct st_hdr *h = (struct st_hdr *)fr;
			h->type = ST_STREAM;
			h->pass = pass;
			h->row = row;
			h->seq = (uint32_t)seq;
			h->cblk = (uint16_t)cblk;
			__threadfence();
			*(volatile unsigned long long *)(a.ready + (size_t)q * a.ring + slot) = t + 1ull;
			atomicAdd(&a.qs[q].synth, 1ull);
		}
		__syncthreads();
	}
}

/* --------------------------------------------------------------------- sender */
/* Posts k frames from `base` (stride ST_SLOT) with NOTIFY on every frame, then
 * poll_completion<CQ_POLL_LAST>(k).  TX_MODE 0 = doca_gpu_dev_eth_txq_send
 * (SHARING_GPU / SYNC_GPU / THREAD): exactly the request service's reply path
 * that ran the whole campaign (server log: "tx mode 0").  TX_MODE 1 =
 * wait_send(ts = 0) (client_doca / TVG transmitter call, needs wait-on-time). */
template <int TX_MODE>
__device__ static int post_batch(struct doca_gpu_eth_txq *txq, uint32_t mkey, const uint8_t *base, uint32_t k,
				 struct st_gen_qstats *qs)
{
	doca_gpu_dev_eth_ticket_t ticket;
	uint32_t completed = 0, retries = 0;
	doca_error_t st = DOCA_SUCCESS;
	__threadfence_system();
	for (uint32_t j = 0; j < k; ++j) {
		const uint8_t *fr = base + (size_t)j * ST_SLOT;
		const uint32_t len = (((const struct st_hdr *)fr)->type == ST_STREAM) ? ST_FRAME : ST_HDR_BYTES;
		if (TX_MODE == 0) {
			doca_gpu_dev_eth_txq_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
						  DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
				txq, (uint64_t)fr, mkey, len, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
		} else {
			doca_gpu_dev_eth_txq_wait_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
						       DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
				txq, 0, (uint64_t)fr, mkey, len, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
		}
	}
	qs->posted += k;
	do {
		st = doca_gpu_dev_eth_txq_poll_completion<DOCA_GPUNETIO_ETH_CQ_POLL_LAST>(txq, k, DOCA_GPUNETIO_ETH_WAIT_FLAG_B,
											&completed);
		++retries;
	} while (st == DOCA_ERROR_AGAIN && retries < 20000000u);
	if (st != DOCA_SUCCESS) {
		qs->tx_errors += 1;
		qs->last_error = (unsigned long long)st;
		return -1;
	}
	qs->sent += k;
	return 0;
}

template <int TX_MODE>
__device__ static void sender_thread(struct st_gen_args a, struct doca_gpu_eth_txq *txq, uint32_t mkey, uint32_t q)
{
	struct st_gen_qstats *qs = a.qs + q;
	const unsigned long long F = (unsigned long long)a.nblk * a.n;
	const unsigned long long count_q = st_count_q(F, a.nq, q);
	const unsigned long long total_q = count_q * a.passes;
	volatile unsigned long long *ready = a.ready + (size_t)q * a.ring;
	uint8_t *ring_base = a.txbuf + (size_t)q * (a.ring + 1u) * ST_SLOT;   /* [ring + 1] slots per queue */
	uint8_t *end_frame = ring_base + (size_t)a.ring * ST_SLOT;            /* the extra slot: PASS_END */
	unsigned long long next = 0ull, next_check = 0ull, t_next = 0ull;
	uint32_t cur_pass = 0;
	int end_pending = 0;

	/* the PASS_END frame of this queue: header template, type and constants once */
	{
		const uint32_t *src = (const uint32_t *)(a.hdr_tmpl + (size_t)q * ST_HDR_BYTES);
		uint32_t *dst = (uint32_t *)end_frame;
		for (int i = 0; i < 16; ++i) dst[i] = src[i];
		struct st_hdr *h = (struct st_hdr *)end_frame;
		h->type = ST_PASS_END;
		h->row = a.n;
		h->cblk = (uint16_t)a.nblk;
		h->gen_queues = (uint16_t)a.nq;
	}

	for (;;) {
		if (gtimer() >= next_check) {
			next_check = gtimer() + ST_EXIT_POLL_NS;
			if (*a.exit_flag)
				break;
		}
		/* pass boundary: all frames of cur_pass posted and completed -> PASS_END, wait for the ACK */
		if (next == (unsigned long long)(cur_pass + 1u) * count_q) {
			if (!end_pending) {
				struct st_hdr *h = (struct st_hdr *)end_frame;
				h->pass = cur_pass;
				h->seq = (uint32_t)count_q;
				if (post_batch<TX_MODE>(txq, mkey, end_frame, 1, qs))
					break;
				qs->sent -= 1;   /* not a data frame */
				qs->ends += 1;
				end_pending = 1;
			}
			const unsigned long long tw = gtimer();
			if (*a.ack_pass > cur_pass) {
				end_pending = 0;
				++cur_pass;
				if (next >= total_q) {
					qs->finished = 1;
					break;
				}
			} else {
				__nanosleep(2000);
				qs->wait_ack_ns += gtimer() - tw;
				continue;
			}
		}
		/* gather up to `burst` consecutive ready frames of cur_pass */
		uint32_t k = 0;
		const unsigned long long limit = (unsigned long long)(cur_pass + 1u) * count_q;
		while (k < a.burst && next + k < limit) {
			const uint32_t slot = (uint32_t)((next + k) % a.ring);
			if (ready[slot] != next + k + 1ull)
				break;
			++k;
		}
		if (k == 0) {
			const unsigned long long tw = gtimer();
			__nanosleep(200);
			qs->wait_ring_ns += gtimer() - tw;
			continue;
		}
		/* ring slots wrap: post in at most two contiguous pieces */
		const uint32_t s0 = (uint32_t)(next % a.ring);
		const uint32_t k1 = (s0 + k <= a.ring) ? k : a.ring - s0;
		if (post_batch<TX_MODE>(txq, mkey, ring_base + (size_t)s0 * ST_SLOT, k1, qs))
			break;
		if (k1 < k && post_batch<TX_MODE>(txq, mkey, ring_base, k - k1, qs))
			break;
		next += k;
		*(volatile unsigned long long *)(a.sent_ctr + q) = next;   /* releases the slots to synthesis */
		if (a.sleep_ns)
			__nanosleep(a.sleep_ns);
		if (a.pace_ns_x1024) {   /* token pacing: this queue may not run ahead of the target rate */
			const unsigned long long now = gtimer();
			if (t_next < now)
				t_next = now;
			t_next += ((unsigned long long)k * a.pace_ns_x1024) >> 10;
			while (gtimer() < t_next)
				__nanosleep(1000);
		}
	}
}

/* ------------------------------------------------- sender block, -W 2 (TVG) */
/* thread 0: client_doca's single-frame call (SHARING_CTA / SYNC_CTA / THREAD scope, ts = 0) for k frames
 * from ring position `pos`, NOTIFY on the last one only (the TVG burst flag pattern). */
__device__ static void post_thread_frames(struct doca_gpu_eth_txq *txq, uint32_t mkey, uint8_t *ring_base, uint32_t ring,
					  unsigned long long pos, uint32_t k, uint32_t len)
{
	doca_gpu_dev_eth_ticket_t ticket;
	for (uint32_t j = 0; j < k; ++j) {
		const uint8_t *fr = ring_base + (size_t)((pos + j) % ring) * ST_SLOT;
		doca_gpu_dev_eth_txq_wait_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA, DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA,
					       DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
			txq, 0, (uint64_t)fr, mkey, len,
			(j + 1u == k) ? DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY : DOCA_GPUNETIO_ETH_SEND_FLAG_NONE, &ticket);
	}
}

/* thread 0: completion of the last NOTIFY frame (TVG: CQ_POLL_LAST, 1, WAIT_FLAG_B, 20 M retries) */
__device__ static doca_error_t wait_last(struct doca_gpu_eth_txq *txq)
{
	uint32_t completed = 0, retries = 0;
	doca_error_t st;
	do {
		st = doca_gpu_dev_eth_txq_poll_completion<DOCA_GPUNETIO_ETH_CQ_POLL_LAST>(txq, 1, DOCA_GPUNETIO_ETH_WAIT_FLAG_B,
											&completed);
		++retries;
	} while (st == DOCA_ERROR_AGAIN && retries < 20000000u);
	return st;
}

__device__ static void sender_block_tvg(struct st_gen_args a, struct doca_gpu_eth_txq *txq, uint32_t mkey, uint32_t q)
{
	__shared__ int s_stop;   /* sticky: set once by thread 0, never cleared */
	const uint32_t tid = threadIdx.x;
	volatile struct st_gen_qstats *qs = a.qs + q;
	const unsigned long long F = (unsigned long long)a.nblk * a.n;
	const unsigned long long count_q = st_count_q(F, a.nq, q);
	const unsigned long long total_q = count_q * a.passes;
	volatile unsigned long long *ready = a.ready + (size_t)q * a.ring;
	uint8_t *ring_base = a.txbuf + (size_t)q * (a.ring + 1u) * ST_SLOT;   /* [ring + 1] slots per queue */
	uint8_t *end_frame = ring_base + (size_t)a.ring * ST_SLOT;            /* the extra slot: PASS_END */
	unsigned long long next = 0ull;                                       /* uniform over the block */
	uint32_t cur_pass = 0;                                                /* uniform over the block */
	unsigned long long next_check = 0ull, t_next = 0ull;                  /* thread 0 */

	if (tid == 0) {
		s_stop = 0;
		const uint32_t *src = (const uint32_t *)(a.hdr_tmpl + (size_t)q * ST_HDR_BYTES);
		uint32_t *dst = (uint32_t *)end_frame;
		for (int i = 0; i < 16; ++i) dst[i] = src[i];
		struct st_hdr *h = (struct st_hdr *)end_frame;
		h->type = ST_PASS_END;
		h->row = a.n;
		h->cblk = (uint16_t)a.nblk;
		h->gen_queues = (uint16_t)a.nq;
	}
	__syncthreads();

	for (;;) {
		if (tid == 0 && gtimer() >= next_check) {
			next_check = gtimer() + ST_EXIT_POLL_NS;
			if (*a.exit_flag)
				s_stop = 1;
		}
		__syncthreads();
		if (s_stop)
			break;
		const unsigned long long limit = (unsigned long long)(cur_pass + 1u) * count_q;

		/* pass boundary: every frame of cur_pass completed -> PASS_END, wait for the ACK */
		if (next == limit) {
			if (tid == 0) {
				struct st_hdr *h = (struct st_hdr *)end_frame;
				h->pass = cur_pass;
				h->seq = (uint32_t)count_q;
				__threadfence_system();
				post_thread_frames(txq, mkey, end_frame, 1u, 0ull, 1u, ST_HDR_BYTES);
				const doca_error_t st = wait_last(txq);
				if (st != DOCA_SUCCESS) {
					qs->tx_errors += 1;
					qs->last_error = (unsigned long long)st;
					s_stop = 1;
				} else {
					qs->ends += 1;
					const unsigned long long tw = gtimer();
					while (*a.ack_pass <= (unsigned long long)cur_pass) {
						if (gtimer() >= next_check) {
							next_check = gtimer() + ST_EXIT_POLL_NS;
							if (*a.exit_flag) { s_stop = 1; break; }
						}
						__nanosleep(2000);
					}
					qs->wait_ack_ns += gtimer() - tw;
				}
			}
			__syncthreads();
			if (s_stop)
				break;
			++cur_pass;
			if (next >= total_q) {
				if (tid == 0)
					qs->finished = 1ull;
				break;
			}
			continue;
		}

		/* the next k frames of this pass must all be synthesized (one ready word per thread) */
		const uint32_t k = (limit - next < ST_GEN_BURST) ? (uint32_t)(limit - next) : ST_GEN_BURST;
		const unsigned long long my = next + tid;
		const int ok = (tid < k) ? (ready[my % a.ring] == my + 1ull) : 1;
		if (!__syncthreads_and(ok)) {
			if (tid == 0) {
				const unsigned long long tw = gtimer();
				__nanosleep(200);
				qs->wait_ring_ns += gtimer() - tw;
			}
			continue;
		}

		/* -L: token pacing per burst on the GPU global timer (the burst itself leaves at line rate) */
		if (tid == 0 && a.pace_ns_x1024) {
			const unsigned long long now = gtimer();
			if (t_next < now)
				t_next = now;
			t_next += ((unsigned long long)k * a.pace_ns_x1024) >> 10;
			while (gtimer() < t_next)
				__nanosleep(1000);
		}
		__threadfence_system();   /* payloads (written by the synthesis blocks) visible to the NIC */
		__syncthreads();

		if (k == ST_GEN_BURST) {
			/* TVG kernel_send_packets: one frame per thread, NOTIFY on the last thread */
			doca_gpu_dev_eth_ticket_t ticket;
			const uint8_t *fr = ring_base + (size_t)(my % a.ring) * ST_SLOT;
			doca_gpu_dev_eth_txq_wait_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA, DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA,
						       DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK>(
				txq, 0, (uint64_t)fr, mkey, ST_FRAME,
				(tid == ST_GEN_BURST - 1u) ? DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY : DOCA_GPUNETIO_ETH_SEND_FLAG_NONE,
				&ticket);
			__syncthreads();
		} else if (tid == 0) {
			post_thread_frames(txq, mkey, ring_base, a.ring, next, k, ST_FRAME);   /* tail of the pass */
		}
		if (tid == 0) {
			qs->posted += k;
			const unsigned long long tw = gtimer();
			const doca_error_t st = wait_last(txq);
			qs->wait_tx_ns += gtimer() - tw;
			if (st != DOCA_SUCCESS) {
				qs->tx_errors += 1;
				qs->last_error = (unsigned long long)st;
				s_stop = 1;
			} else {
				qs->sent += k;
				if (k == ST_GEN_BURST)
					qs->bursts += 1;
				__threadfence();
				*(volatile unsigned long long *)(a.sent_ctr + q) = next + k;   /* releases the slots to synthesis */
			}
		}
		__syncthreads();
		if (s_stop)
			break;
		next += k;
		if (a.sleep_ns && tid == 0)
			__nanosleep(a.sleep_ns);
	}
}

/* ----------------------------------------------------------------- ACK warp */
/* lane 0 of the ACK warp: device counters -> host-mapped mirror (posted writes, the warp does not wait) */
__device__ static void publish_mirror(const struct st_gen_args &a, unsigned long long ticks)
{
	volatile struct st_gen_mirror *m = a.mir;
	for (uint32_t q = 0; q < a.nq; ++q) {
		const volatile unsigned long long *src = (const volatile unsigned long long *)(a.qs + q);
		volatile unsigned long long *dst = (volatile unsigned long long *)&m->qs[q];
		for (uint32_t i = 0; i < sizeof(struct st_gen_qstats) / 8u; ++i)
			dst[i] = src[i];
	}
	m->ack_pass = *a.ack_pass;
	m->ack_lost = *(volatile unsigned long long *)a.ack_lost;
	m->ticks = ticks;   /* heartbeat of the ACK warp */
	__threadfence_system();
}

__device__ static void ack_warp(struct st_gen_args a, struct doca_gpu_eth_rxq *rxq)
{
	__shared__ uint64_t first_idx;
	__shared__ uint32_t num;
	__shared__ struct doca_gpu_dev_eth_rxq_attr attr[ST_GEN_RX_BURST];
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long next_check = 0ull, ticks = 0ull;
	for (;;) {
		int quit = 0;
		if (lane == 0 && gtimer() >= next_check) {
			next_check = gtimer() + ST_EXIT_POLL_NS;
			publish_mirror(a, ++ticks);
			quit = *a.exit_flag != 0u;
		}
		if (__shfl_sync(0xFFFFFFFFu, quit, 0))
			break;
		doca_error_t ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP, DOCA_GPUNETIO_ETH_MCST_AUTO,
							     DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
			rxq, ST_GEN_RX_BURST, 5000, &first_idx, &num, attr);
		if (ret != DOCA_SUCCESS) {
			if (lane == 0) {
				publish_mirror(a, ++ticks);
				a.mir->ack_lost = ~0ull;   /* marker: the ACK receive call failed */
				__threadfence_system();
			}
			break;
		}
		__syncwarp();
		const uint32_t count = num;
		if (lane == 0) {
			for (uint32_t i = 0; i < count; ++i) {
				const uint32_t *w32 = (const uint32_t *)doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, first_idx + i);
				const uint32_t w3 = __ldcg(w32 + 3), w4 = __ldcg(w32 + 4);
				if ((w3 >> 16) != ST_MAGIC || (w4 & 0xFFu) != ST_PASS_ACK)
					continue;
				const uint32_t pass = __ldcg(w32 + 5), lost = __ldcg(w32 + 6);
				*(volatile unsigned long long *)a.ack_lost = lost;
				__threadfence();
				if (*a.ack_pass < (unsigned long long)pass + 1ull)
					*a.ack_pass = (unsigned long long)pass + 1ull;
				__threadfence();
			}
		}
		__syncwarp();
	}
}

/* ONE kernel, all roles by block index, all blocks resident at once:
 *   blocks 0..nq-1      sender (thread 0; the other threads idle until it is done)
 *   block  nq           ACK receive warp (warp 0; the other warps idle)
 *   blocks nq+1..       synthesis blocks (all threads)
 * No second kernel, no second stream: nothing can be left waiting behind a
 * persistent kernel on a shared hardware queue. */
template <int TX_MODE>
__global__ void __launch_bounds__(ST_GEN_THREADS) gen_kernel(struct st_gen_args a, struct doca_gpu_eth_txq **txqs,
							     struct doca_gpu_eth_rxq *ack_rxq, uint32_t mkey)
{
	__shared__ int done;
	if (blockIdx.x < a.nq && TX_MODE == 2) {
		sender_block_tvg(a, txqs[blockIdx.x], mkey, blockIdx.x);
	} else if (blockIdx.x < a.nq) {
		if (threadIdx.x == 0) {
			done = 0;
			sender_thread<TX_MODE == 1 ? 1 : 0>(a, txqs[blockIdx.x], mkey, blockIdx.x);
			done = 1;
			__threadfence_block();
		} else {
			while (!*(volatile int *)&done)
				__nanosleep(100000);
		}
	} else if (blockIdx.x == a.nq) {
		if (threadIdx.x < 32) {
			if (threadIdx.x == 0)
				done = 0;
			__syncwarp();
			ack_warp(a, ack_rxq);
			if (threadIdx.x == 0) {
				done = 1;
				__threadfence_block();
			}
		} else {
			while (!*(volatile int *)&done)
				__nanosleep(100000);
		}
	} else {
		synth_block(a, blockIdx.x - a.nq - 1u);
	}
}

extern "C" int st_gen_launch(cudaStream_t s_io, cudaStream_t s_synth, struct st_gen_args a,
			     struct doca_gpu_eth_txq **txqs_dev, struct doca_gpu_eth_rxq *ack_rxq, uint32_t mkey)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) {
		fprintf(stderr, "[gen] CUDA error before launch: %s\n", cudaGetErrorString(e));
		return -1;
	}
	(void)s_synth;
	const unsigned blocks = a.nq + 1u + a.nsb;
	if (a.tx_mode == 2)
		gen_kernel<2><<<blocks, ST_GEN_THREADS, 0, s_io>>>(a, txqs_dev, ack_rxq, mkey);
	else if (a.tx_mode == 1)
		gen_kernel<1><<<blocks, ST_GEN_THREADS, 0, s_io>>>(a, txqs_dev, ack_rxq, mkey);
	else
		gen_kernel<0><<<blocks, ST_GEN_THREADS, 0, s_io>>>(a, txqs_dev, ack_rxq, mkey);
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		fprintf(stderr, "[gen] kernel launch (%u blocks x %u threads): %s\n", blocks, ST_GEN_THREADS,
			cudaGetErrorString(e));
		return -1;
	}
	return 0;
}

/* all roles spin-wait on each other: every block must be resident at once */
extern "C" int st_gen_max_blocks(void)
{
	int dev = 0, per_sm = 0, sms = 0, worst = 1 << 30;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
	const void *k[3] = {(const void *)gen_kernel<0>, (const void *)gen_kernel<1>, (const void *)gen_kernel<2>};
	for (int i = 0; i < 3; ++i) {
		if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k[i], ST_GEN_THREADS, 0) != cudaSuccess)
			return 0;
		if (per_sm < worst)
			worst = per_sm;
	}
	return worst * sms;
}
