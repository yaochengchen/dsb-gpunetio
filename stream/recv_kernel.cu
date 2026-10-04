/*
 * recv_kernel.cu -- receive warp and worker blocks of the streaming receiver.
 * See recv.h.
 *
 * The receive call is the request service's (server_doca/doca_kernel.cu,
 * i.e. the TVG receiver's); the work ring is the request service's ticket
 * ring (dsb_queue.h rules); the PASS_ACK send is the request service's reply
 * path (mode 0 txq_send + poll_completion as NVIDIA's gpunetio_simple_send,
 * mode 1 wait_send with timestamp 0 as the TVG transmitter), issued by the
 * receive warp on the host's request.  The per-pass update is run by the
 * worker blocks on the host's request (update_word of stream_kernels.cuh):
 * the host launches nothing while these kernels run.
 */
#include <stdio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

#include "recv.h"
#include "stream_kernels.cuh"

using namespace st;

#define ST_EXIT_POLL_NS 500000ull

enum { SEG_STREAM = 1, SEG_END = 2, SEG_BAD = 3 };
enum { WAIT_BURST = 0, WAIT_UPDATE = 1, WAIT_QUIT = 2 };

/* The control block lives in host memory and is read ONLY by lane 0 of the
 * receive warp, at most once per ST_EXIT_POLL_NS (never once per receive
 * call); it mirrors exit_flag / upd_req into device memory (st_dev) for the
 * worker blocks. */
__device__ __forceinline__ bool should_exit(const struct st_rx_args &a, unsigned long long &next_check)
{
	const unsigned long long now = gtimer();
	if (now < next_check)
		return false;
	next_check = now + ST_EXIT_POLL_NS;
	return a.ctl->exit_flag != 0u;
}

__device__ __forceinline__ int slot_free(unsigned long long ack, unsigned long long t)
{
	return ack == (t >= ST_WORK_CAP ? t - ST_WORK_CAP + 1ull : 0ull);
}

/* ---- PASS_ACK: lane 0 of the receive warp fills the template and posts it ---- */
__device__ static void send_ack(const struct st_rx_args &a, uint32_t req)
{
	__threadfence();                               /* ack_req was read: the fields were written before it by the host */
	uint32_t *h = (uint32_t *)a.tx_frame;          /* 64-byte header, 4-byte aligned fields */
	h[5] = req - 1u;                               /* pass */
	h[6] = a.ctl->ack_lost;                        /* row = frames lost */
	h[7] = a.ctl->ack_frames;                      /* seq = frames received */
	__threadfence_system();                        /* frame bytes visible to the NIC before the doorbell */
	doca_gpu_dev_eth_ticket_t ticket;
	doca_error_t st = DOCA_SUCCESS;
	uint32_t completed = 0, retries = 0;
	if (a.tx_mode == 1) {
		doca_gpu_dev_eth_txq_wait_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
					       DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
			a.txq, 0, (uint64_t)a.tx_frame, a.tx_mkey, ST_HDR_BYTES, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
	} else {
		doca_gpu_dev_eth_txq_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
					  DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
			a.txq, (uint64_t)a.tx_frame, a.tx_mkey, ST_HDR_BYTES, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
	}
	do {
		st = doca_gpu_dev_eth_txq_poll_completion<DOCA_GPUNETIO_ETH_CQ_POLL_LAST>(a.txq, 1, DOCA_GPUNETIO_ETH_WAIT_FLAG_B,
											&completed);
		++retries;
	} while (st == DOCA_ERROR_AGAIN && retries < 20000000u);
	__threadfence();
	if (st == DOCA_SUCCESS)
		*(volatile unsigned long long *)&a.st->ack_done = req;
	else
		*(volatile unsigned long long *)&a.st->tx_err = (unsigned long long)st;
}

/* ------------------------------------------------------------ receive warp */
#define VLD(x) (*(volatile unsigned long long *)&(x))

/* lane 0: snapshot of the worker counters (device) + this warp's own into the host-mapped stats.
 * Plain stores to pinned host memory are posted PCIe writes: the warp does not wait for them. */
__device__ static void publish_stats(const struct st_rx_args &a, unsigned long long polls, unsigned long long bursts,
				     unsigned long long frames, unsigned long long dropped, unsigned long long t_recv,
				     unsigned long long t_pub, unsigned long long gap_max, unsigned long long gaps1,
				     unsigned long long gaps5, unsigned long long call_max, unsigned long long full_calls,
				     unsigned long long backlog_max)
{
	volatile struct st_stats *o = a.st;
	const struct st_dev *d = a.dv;
	/* worker side first, bursts_done before frames etc. are irrelevant here: the host uses them only
	 * when the traffic has stopped (drain), and rx_bursts below is this warp's exact value */
	const unsigned long long bdone = VLD(d->bursts_done);
	__threadfence();
	o->bursts_done = bdone;
	o->frames = VLD(d->frames);
	o->ends = VLD(d->ends);
	o->end_frames = VLD(d->end_frames);
	o->gen_queues = VLD(d->gen_queues);
	o->bad = VLD(d->bad);
	o->tiles = VLD(d->tiles);
	o->t_work_ns = VLD(d->t_work_ns);
	o->worker_blocks = VLD(d->worker_blocks);
	o->upd_done = VLD(d->upd_done);
	o->slots_done = VLD(d->slots_done);
	o->rx_polls = polls;
	o->rx_bursts = bursts;
	o->rx_frames = frames;
	o->rx_dropped = dropped;
	o->t_recv_ns = t_recv;
	o->t_pub_ns = t_pub;
	o->gap_max_ns = gap_max;
	o->gaps_1ms = gaps1;
	o->gaps_5ms = gaps5;
	o->call_max = call_max;
	o->full_calls = full_calls;
	o->backlog_max = backlog_max;
	__threadfence_system();
}

__global__ void rx_kernel(struct st_rx_args a, struct doca_gpu_eth_rxq *rxq)
{
	__shared__ uint64_t first_idx;
	__shared__ uint32_t num;
	__shared__ struct doca_gpu_dev_eth_rxq_attr attr[ST_RX_BURST];
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long tail = 0, next_check = 0ull;
	unsigned long long t_recv = 0, t_pub = 0, polls = 0, bursts = 0, frames = 0, dropped = 0;
	/* lane 0 diagnostics: see struct st_stats */
	unsigned long long t_prev = 0, gap_max = 0, gaps1 = 0, gaps5 = 0, call_max = 0, full_calls = 0, backlog_max = 0;
	uint32_t acks = 0;
	volatile struct st_dev *dv = a.dv;

	for (;;) {
		int quit = 0;
		if (lane == 0) {
			/* iteration length = how long this warp was away from the queue (tick, receive call, publish) */
			const unsigned long long now = gtimer();
			if (t_prev) {
				const unsigned long long g = now - t_prev;
				if (g > gap_max) gap_max = g;
				if (g > 1000000ull) ++gaps1;
				if (g > 5000000ull) ++gaps5;
			}
			t_prev = now;
			if (now >= next_check) {   /* throttled: counters out, host requests in (3 reads of pinned host memory) */
				next_check = now + ST_EXIT_POLL_NS;
				publish_stats(a, polls, bursts, frames, dropped, t_recv, t_pub, gap_max, gaps1, gaps5, call_max,
					      full_calls, backlog_max);
				quit = a.ctl->exit_flag != 0u;
				dv->upd_req = a.ctl->upd_req;     /* mirrored for the worker blocks (device memory) */
				if (quit)
					dv->exit_flag = 1u;
				__threadfence();
				const uint32_t req = a.ctl->ack_req;
				if (!quit && req > acks) {
					/* the host read the per-pass maxima after the drain; they restart with the next pass */
					gap_max = 0; call_max = 0; backlog_max = 0;
					publish_stats(a, polls, bursts, frames, dropped, t_recv, t_pub, gap_max, gaps1, gaps5,
						      call_max, full_calls, backlog_max);
					send_ack(a, req);
					acks = req;
					t_prev = gtimer();   /* the ACK send is not a receive stall */
				}
			}
		}
		if (__shfl_sync(0xFFFFFFFFu, quit, 0))
			break;
		const unsigned long long t0 = (lane == 0) ? gtimer() : 0ull;
		doca_error_t ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP, DOCA_GPUNETIO_ETH_MCST_AUTO,
							     DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO, DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
			rxq, ST_RX_BURST, ST_RX_TIMEOUT_NS, &first_idx, &num, attr);
		if (ret != DOCA_SUCCESS) {
			if (lane == 0) {
				publish_stats(a, polls, bursts, frames, dropped, t_recv, t_pub, gap_max, gaps1, gaps5, call_max,
					      full_calls, backlog_max);
				a.st->rx_err += 1;
				dv->exit_flag = 1u;   /* the workers must not wait for a warp that is gone */
				__threadfence_system();
			}
			break;
		}
		__syncwarp();
		const uint32_t count = num;
		if (lane == 0) {
			const unsigned long long t1 = gtimer();
			t_recv += t1 - t0;
			++polls;
			if (count) {
				if (count > call_max) call_max = count;
				if (count >= ST_RX_BURST) ++full_calls;
				volatile struct st_work_item *it = a.items + (tail & (ST_WORK_CAP - 1u));
				if (slot_free(it->ack, tail)) {
					it->first_idx = first_idx;
					it->count = count;
					it->t_rx = t1;
					__threadfence();
					it->seq = tail + 1ull;
					++tail;
					++bursts;
					frames += count;
					/* frames the workers still have to read in place: > ST_RX_PKT_NUM means the NIC can
					 * overwrite slots before they are read (one device-memory read per burst) */
					const unsigned long long done = dv->slots_done;
					const unsigned long long bl = frames > done ? frames - done : 0ull;
					if (bl > backlog_max) backlog_max = bl;
				} else {
					++dropped;   /* workers are more than ST_WORK_CAP bursts behind */
				}
				t_pub += gtimer() - t1;
			}
		}
		__syncwarp();
	}
	if (lane == 0) {
		dv->exit_flag = 1u;
		__threadfence();
	}
}

/* ------------------------------------------------------------- worker block */
template <int MAXB>
__global__ void __launch_bounds__(ST_WORKER_THREADS) worker_kernel(struct st_rx_args a, struct doca_gpu_eth_rxq *rxq)
{
	extern __shared__ uint32_t tile[];                 /* [B][ST_WORDS] */
	__shared__ unsigned long long first_idx, ticket;
	__shared__ uint32_t count;
	__shared__ uint32_t h_type[ST_WORKER_THREADS], h_row[ST_WORKER_THREADS], h_cblk[ST_WORKER_THREADS], h_pass[ST_WORKER_THREADS];
	__shared__ uint32_t seg_start, seg_end, seg_kind, seg_cblk, seg_pass;
	__shared__ uint32_t tile_cblk, tile_pass;
	__shared__ int mode;
	__shared__ uint32_t upd_k;

	const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, nwarps = blockDim.x >> 5;
	unsigned long long next_check = 0ull;
	uint32_t upd_seen = 0;                                /* tid 0: last update request served */
	int have_ticket = 0;                                  /* tid 0: a claimed ticket is pending */
	unsigned long long t = 0;                             /* tid 0: the pending ticket */
	/* tid 0: counters of the current burst, added to the DEVICE-memory counters (a.dv) when the burst
	 * is done -- device atomics, no page faults, no PCIe */
	unsigned long long l_frames = 0, l_bad = 0;
	volatile struct st_dev *dv = a.dv;
	state S;
	S.n = a.n; S.B = a.B; S.wtot = a.wtot; S.nblk = a.nblk;
	S.x = a.x; S.y = a.y; S.signs = a.signs; S.acc = a.acc; S.e2 = a.e2; S.chk = a.chk;

	if (tid == 0) {
		tile_cblk = 0xFFFFFFFFu;
		tile_pass = 0xFFFFFFFFu;
		atomicAdd(&a.dv->worker_blocks, 1ull);
	}
	__syncthreads();

	for (;;) {
		/* ---- claim a burst (ticket ring of the request service); while waiting
		 * for it, serve an update request of the host ------------------------ */
		if (tid == 0) {
			if (!have_ticket) {
				t = atomicAdd(a.work_claim, 1ull);
				have_ticket = 1;
			}
			volatile struct st_work_item *w = a.items + (t & (ST_WORK_CAP - 1u));
			mode = WAIT_BURST;
			ticket = t;
			while (w->seq != t + 1ull) {
				const unsigned long long now = gtimer();
				if (now >= next_check) {
					next_check = now + ST_EXIT_POLL_NS;
					/* host requests as mirrored by the receive warp (device memory, not the host page) */
					if (dv->exit_flag) {
						mode = WAIT_QUIT;
						break;
					}
					const uint32_t req = dv->upd_req;
					if (req > upd_seen) {
						upd_seen = req;
						upd_k = req - 1u;
						mode = WAIT_UPDATE;
						break;
					}
				}
			}
			if (mode == WAIT_BURST) {
				__threadfence();
				first_idx = w->first_idx;
				count = w->count;
				have_ticket = 0;
			}
		}
		__syncthreads();
		if (mode == WAIT_QUIT)
			break;
		if (mode == WAIT_UPDATE) {
			/* end of pass upd_k: this block's share of the B x wtot sign words, one word per warp */
			const uint32_t k = upd_k;
			const int do_update = k < a.K;
			const unsigned long long words = (unsigned long long)a.B * a.wtot;
			for (unsigned long long i = (unsigned long long)blockIdx.x * nwarps + warp; i < words;
			     i += (unsigned long long)gridDim.x * nwarps) {
				const uint32_t b = (uint32_t)(i / a.wtot), w = (uint32_t)(i - (unsigned long long)b * a.wtot);
				update_word(S, b, w, lane, k, a.K, a.xi, a.dt, do_update);
			}
			__threadfence();
			__syncthreads();
			if (tid == 0)
				atomicAdd(&a.dv->upd_done, 1ull);
			__syncthreads();
			continue;
		}
		const unsigned long long tw0 = (tid == 0) ? gtimer() : 0ull;
		const uint32_t total = count;

		/* a burst holds up to ST_RX_BURST frames; take it in chunks of one frame per thread */
		for (uint32_t base = 0; base < total; base += ST_WORKER_THREADS) {
			const uint32_t cnt = (total - base < ST_WORKER_THREADS) ? total - base : ST_WORKER_THREADS;
			const unsigned long long chunk_idx = first_idx + base;

			/* ---- headers of the chunk, in parallel (all fields 4-byte aligned) */
			if (tid < cnt) {
				const uint8_t *f = (const uint8_t *)doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, chunk_idx + tid);
				const uint32_t *w32 = (const uint32_t *)f;
				const uint32_t w3 = __ldcg(w32 + 3);   /* bytes 12..15: ethertype (BE) | magic (LE) */
				const uint32_t w4 = __ldcg(w32 + 4);   /* bytes 16..19: type, queue, version, instance */
				const uint32_t mg = w3 >> 16;
				const uint32_t type = w4 & 0xFFu;
				const uint32_t pass = __ldcg(w32 + 5);
				const uint32_t row = __ldcg(w32 + 6);
				const uint32_t seq = __ldcg(w32 + 7);
				const uint32_t w8 = __ldcg(w32 + 8);   /* cblk | gen_queues << 16 */
				uint32_t kind = SEG_BAD;
				if (mg == ST_MAGIC) {
					if (type == ST_STREAM)
						kind = SEG_STREAM;
					else if (type == ST_PASS_END)
						kind = SEG_END;
				}
				h_type[tid] = kind;
				h_pass[tid] = pass;
				h_row[tid] = row;
				h_cblk[tid] = w8 & 0xFFFFu;
				if (kind == SEG_END) {   /* PASS_END: count and queue number into the global stats */
					dv->gen_queues = w8 >> 16;
					atomicAdd(&a.dv->end_frames, (unsigned long long)seq);
					__threadfence();
					atomicAdd(&a.dv->ends, 1ull);
				}
			}
			__syncthreads();

			/* ---- segments of equal (pass, column block) ----------------------- */
			uint32_t i = 0;
			while (i < cnt) {
				if (tid == 0) {
					const uint32_t kind = h_type[i];
					uint32_t j = i + 1;
					if (kind == SEG_STREAM)
						while (j < cnt && h_type[j] == SEG_STREAM && h_cblk[j] == h_cblk[i] && h_pass[j] == h_pass[i])
							++j;
					seg_start = i;
					seg_end = j;
					seg_kind = kind;
					seg_cblk = h_cblk[i];
					seg_pass = h_pass[i];
				}
				__syncthreads();
				const uint32_t kind = seg_kind, s0 = seg_start, s1 = seg_end;
				if (kind == SEG_STREAM) {
					if (tile_cblk != seg_cblk || tile_pass != seg_pass) {
						load_tile(tile, a.signs, a.wtot, a.B, seg_cblk, tid, blockDim.x);
						__syncthreads();
						if (tid == 0) {
							tile_cblk = seg_cblk;
							tile_pass = seg_pass;
							atomicAdd(&a.dv->tiles, 1ull);
						}
						__syncthreads();
					}
					for (uint32_t f = s0 + warp; f < s1; f += nwarps) {
						const uint8_t *fr = (const uint8_t *)doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, chunk_idx + f);
						const uint32_t row = h_row[f];
						if (row < a.n)
							accumulate_frame<MAXB>((const uint32_t *)(fr + ST_HDR_BYTES), tile, a.acc, a.n, a.B,
									       row, lane);
					}
					if (tid == 0)
						l_frames += s1 - s0;
				} else if (kind == SEG_BAD) {
					if (tid == 0)
						++l_bad;
				}
				__syncthreads();
				i = s1;
			}
		}
		/* ---- done: every thread's atomics visible, then acknowledge ------- */
		__threadfence();
		__syncthreads();
		if (tid == 0) {
			atomicAdd(&a.dv->frames, l_frames);
			if (l_bad) atomicAdd(&a.dv->bad, l_bad);
			atomicAdd(&a.dv->t_work_ns, gtimer() - tw0);
			atomicAdd(&a.dv->slots_done, (unsigned long long)total);
			__threadfence();   /* the counters before bursts_done: drained => counted */
			atomicAdd(&a.dv->bursts_done, 1ull);
			l_frames = l_bad = 0;
			volatile struct st_work_item *w = a.items + (ticket & (ST_WORK_CAP - 1u));
			w->ack = ticket + 1ull;
		}
	}
}

extern "C" int st_worker_max_blocks(uint32_t smem_bytes)
{
	int dev = 0, per_sm = 0, sms = 0;
	cudaGetDevice(&dev);
	const void *k = (const void *)worker_kernel<ST_MAX_B>;
	if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes) != cudaSuccess)
		return 0;
	if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, ST_WORKER_THREADS, smem_bytes) != cudaSuccess)
		return 0;
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
	return per_sm * sms;
}

extern "C" int st_rx_launch(cudaStream_t s_rx, cudaStream_t s_work, struct st_rx_args a, struct doca_gpu_eth_rxq *rxq)
{
	const uint32_t smem = a.B * ST_PAYLOAD;
	cudaError_t e;
	if (a.B > ST_MAX_B)
		return -1;
	e = cudaFuncSetAttribute((const void *)worker_kernel<ST_MAX_B>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
	if (e != cudaSuccess) {
		fprintf(stderr, "[recv] smem %u B per block not available: %s\n", smem, cudaGetErrorString(e));
		return -1;
	}
	/* receive warp first, so it is resident before the workers take the SMs */
	rx_kernel<<<1, 32, 0, s_rx>>>(a, rxq);
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		fprintf(stderr, "[recv] rx kernel launch: %s\n", cudaGetErrorString(e));
		return -1;
	}
	worker_kernel<ST_MAX_B><<<a.workers, ST_WORKER_THREADS, smem, s_work>>>(a, rxq);
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		fprintf(stderr, "[recv] worker kernel launch: %s\n", cudaGetErrorString(e));
		return -1;
	}
	return 0;
}
