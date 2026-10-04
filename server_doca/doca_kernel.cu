/*
 * doca_kernel.cu -- the two persistent kernels of the server, each on its
 * own CUDA stream, the small RX/TX one launched first so it is resident
 * before the workers fill the GPU:
 *
 *   dsb_io_kernel  <<<2, 32>>>   block 0: DOCA GPUNetIO receive, warp scope
 *                                (the TVG receiver kernel + a copy into the work ring)
 *                                block 1, thread 0: DOCA GPUNetIO send + completion poll
 *                                (the official gpunetio_simple_send THREAD-scope sequence)
 *   worker_kernel  <<<W, 256>>>  dSB workers (dsb_solve.cuh; the kernel dsb_bench
 *                                "persistent" mode runs on this GPU)
 * Two kernels on two streams, exactly the shape of dsb_bench "persistent".
 *
 * No cooperative launch: the kernels only communicate through the rings in
 * dsb_queue.h, so none of them needs the others to be co-resident.
 */
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_log.h>

#include "doca_server.h"
#include "dsb_solve.cuh"

DOCA_LOG_REGISTER(DSB::KERNEL);

/* Receive timeout of the warp-scope recv call.  The call returns when it has
 * DSB_RX_BURST frames or when the timeout has elapsed, so a lone request waits
 * for the rest of the window (the residual of a 5000 ns timeout was 0-5 us,
 * uniformly distributed, on top of a 12.5 us floor).  It must not be 0: on PU2
 * a timeout of 0 blocks until frames arrive (see stream/recv.h), and the RX
 * warp would then never see the exit flag on an idle link.  Override with
 * -DDSB_RX_TIMEOUT_NS=<ns> to compare. */
#ifndef DSB_RX_TIMEOUT_NS
#define DSB_RX_TIMEOUT_NS 500u
#endif

using namespace dsbgpu;

/* ---------------------------------------------------------------------------
 * RX: one warp.  Frames are handled one after another by the whole warp;
 * SOLVE frames cost one work item (lane 0), LOAD fragments a warp-wide copy.
 * ------------------------------------------------------------------------- */
/* Whole warp.  Copies the frame out of the NIC's cyclic buffer into the work
 * slot's own frame buffer and publishes the item; drops it when the slot's
 * previous ticket has not been answered yet (ring full).  `tail` is kept
 * identical in all lanes. */
__device__ static void push_item(const dsb_ctx &ctx, unsigned long long &tail, const uint8_t *frame,
				 uint32_t bytes, uint32_t kind, unsigned long long t_rx, uint32_t lane)
{
	volatile dsb_work_item *it = ctx.items + (tail & (DSB_WORK_CAP - 1u));
	int ok = 0;
	if (lane == 0)
		ok = dsb_slot_free(it->ack, tail);
	ok = __shfl_sync(0xFFFFFFFFu, ok, 0);
	if (!ok) {
		if (lane == 0)
			atomicAdd(&ctx.stats->rx_dropped, 1ull);
		return;
	}
	uint8_t *dst = ctx.frame_pool + (size_t)(tail & (DSB_WORK_CAP - 1u)) * DSB_MAX_FRAME;
	const uint32_t q = (bytes + 15u) >> 4;
	for (uint32_t i = lane; i < q; i += 32u)
		reinterpret_cast<uint4 *>(dst)[i] = __ldcg(reinterpret_cast<const uint4 *>(frame) + i);
	__syncwarp();
	if (lane == 0) {
		__threadfence();
		it->t_rx = t_rx;
		it->frame = dst;
		it->bytes = bytes;
		it->kind = kind;
		__threadfence();
		it->seq = tail + 1ull;
	}
	__syncwarp();
	++tail;
}

__device__ static void rx_role(const dsb_ctx ctx, struct doca_gpu_eth_rxq *rxq)
{
	__shared__ uint64_t first_idx;
	__shared__ uint32_t num;
	__shared__ struct doca_gpu_dev_eth_rxq_attr attr[DSB_RX_BURST];
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long tail = 0; /* identical in all lanes */
	unsigned long long next_check = 0ull;

	for (;;) {
		/* one reader of the (host-mapped, throttled) exit flag, so the warp leaves together */
		int quit = (lane == 0) ? (int)should_exit(ctx, next_check) : 0;
		if (__shfl_sync(0xFFFFFFFFu, quit, 0))
			break;
		doca_error_t ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP,
							     DOCA_GPUNETIO_ETH_MCST_AUTO,
							     DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
							     DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
			rxq, DSB_RX_BURST, DSB_RX_TIMEOUT_NS, &first_idx, &num, attr);
		if (ret != DOCA_SUCCESS) {
			if (lane == 0) {
				atomicAdd(&ctx.stats->rx_bad, 1ull);
				DOCA_GPUNETIO_VOLATILE(*ctx.exit_flag) = 1;
			}
			break;
		}
		__syncwarp();
		const uint32_t count = num;
		if (count == 0)
			continue;
		const unsigned long long t_rx = gtimer();
		if (lane == 0)
			atomicAdd(&ctx.stats->rx_frames, (unsigned long long)count);

		for (uint32_t i = 0; i < count; ++i) {
			const uint8_t *f = (const uint8_t *)doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, first_idx + i);
			const uint32_t bytes = attr[i].bytes;
			/* every lane reads the 64-byte header (broadcast, L1 bypass) */
			uint4 hq[4];
#pragma unroll
			for (int q = 0; q < 4; ++q)
				hq[q] = __ldcg(reinterpret_cast<const uint4 *>(f) + q);
			const uint8_t *hb = reinterpret_cast<const uint8_t *>(hq);
			const dsb_hdr_common *c = reinterpret_cast<const dsb_hdr_common *>(hb);
			const bool ok = bytes >= DSB_HDR_BYTES && bytes <= DSB_MAX_FRAME &&
					hb[12] == (DSB_ETHERTYPE >> 8) && hb[13] == (DSB_ETHERTYPE & 0xFF) &&
					c->version == DSB_PROTO_VERSION;
			if (!ok) {
				if (lane == 0)
					atomicAdd(&ctx.stats->rx_bad, 1ull);
				continue;
			}

			if (c->type == DSB_MSG_SOLVE) {
				if (lane == 0)
					atomicAdd(&ctx.stats->rx_solve, 1ull);
				push_item(ctx, tail, f, bytes, DSB_WORK_SOLVE, t_rx, lane);
			} else if (c->type == DSB_MSG_LOAD) {
				const dsb_load_frag *h = reinterpret_cast<const dsb_load_frag *>(hb);
				const uint32_t id = h->model_id;
				const bool range = id < DSB_MAX_MODELS && h->generation != 0 &&
						   h->total_bytes <= DSB_MODEL_SLOT_BYTES &&
						   h->total_bytes >= dsb_model_bytes(h->n, h->format) &&
						   dsb_model_bytes(h->n, h->format) != 0 &&
						   h->frag_count >= 1 && h->frag_count <= DSB_MAX_FRAGS &&
						   h->frag_idx < h->frag_count &&
						   (h->frag_bytes & 3u) == 0 && (h->offset & 3u) == 0 &&
						   h->offset + h->frag_bytes <= ((h->total_bytes + 3u) & ~3u) &&
						   DSB_HDR_BYTES + h->frag_bytes <= bytes;
				if (!range) {
					if (lane == 0)
						atomicAdd(&ctx.stats->rx_bad, 1ull);
					continue;
				}
				volatile dsb_model_desc *m = ctx.models + id;
				/* lane 0 decides: 0 = ignore, 1 = new fragment, 2 = duplicate (re-ACK if ready) */
				int act = 0;
				if (lane == 0) {
					atomicAdd(&ctx.stats->rx_load, 1ull);
					if (m->generation != h->generation) { /* a new upload of this id */
						m->ready_gen = 0;
						__threadfence();
						m->generation = h->generation;
						m->frags_seen = 0;
						m->total_bytes = h->total_bytes;
						m->frag_count = h->frag_count;
						m->n = h->n;
						m->format = h->format;
						m->xi_default = h->xi_default;
						for (uint32_t k = 0; k < DSB_MAX_FRAGS / 32; ++k)
							m->frag_bits[k] = 0;
					}
					if (m->total_bytes != h->total_bytes || m->frag_count != h->frag_count ||
					    m->n != h->n || m->format != h->format)
						act = 0; /* inconsistent with the generation's first fragment */
					else if ((m->frag_bits[h->frag_idx >> 5] >> (h->frag_idx & 31)) & 1u)
						act = 2;
					else
						act = 1;
				}
				act = __shfl_sync(0xFFFFFFFFu, act, 0);
				if (act == 1) {
					uint32_t *dst = reinterpret_cast<uint32_t *>(
						ctx.model_pool + (size_t)id * DSB_MODEL_SLOT_BYTES + h->offset);
					const uint32_t *src = reinterpret_cast<const uint32_t *>(f + DSB_HDR_BYTES);
					for (uint32_t w = lane; w < (h->frag_bytes >> 2); w += 32u)
						dst[w] = __ldcg(src + w);
					__syncwarp();
				}
				int ack = 0;
				if (lane == 0) {
					if (act == 1) {
						__threadfence();
						m->frag_bits[h->frag_idx >> 5] |= 1u << (h->frag_idx & 31);
						const uint32_t seen = m->frags_seen + 1u;
						m->frags_seen = seen;
						if (seen == h->frag_count) {
							__threadfence();
							m->ready_gen = h->generation;
							atomicAdd(&ctx.stats->models_ready, 1ull);
							ack = 1;
						}
					} else if (act == 2 && m->ready_gen == h->generation) {
						ack = 1; /* the client retried: the ACK was lost */
					}
				}
				if (__shfl_sync(0xFFFFFFFFu, ack, 0)) {
					if (lane == 0)
						atomicAdd(&ctx.stats->rx_acks, 1ull);
					push_item(ctx, tail, f, DSB_HDR_BYTES, DSB_WORK_ACK, t_rx, lane);
				}
			} else {
				if (lane == 0)
					atomicAdd(&ctx.stats->rx_bad, 1ull);
			}
			__syncwarp();
		}
	}
}

/* ---------------------------------------------------------------------------
 * TX: one thread.  Replies are sent strictly in reply-ring order, up to
 * DSB_TX_BATCH per completion poll.  Three send paths, chosen with -W:
 *
 *   mode 0  the official DOCA sample gpunetio_simple_send, THREAD exec scope
 *           (send_packets_shared_qp_enabled<EXEC_SCOPE_THREAD>):
 *           doca_gpu_dev_eth_txq_send<SHARING_GPU, SYNC_GPU, AUTO, THREAD>(..., NOTIFY)
 *           for every reply, then poll_completion<CQ_POLL_LAST>(txq, n, WAIT_FLAG_B).
 *   mode 1  the TVG transmitter's call, wait_send with timestamp 0, same
 *           NOTIFY/poll pattern (needs wait-on-time offload: set on the host).
 *   mode 2  the same sample with shared_qp disabled
 *           (send_packets_shared_qp_disabled): get_wqe_ptr + wqe_prepare_send
 *           for every reply (NOTIFY on the last), one submit, then
 *           poll_completion_at<SHARING_GPU, SYNC_CTA>(txq, cqe_idx, WAIT_FLAG_B).
 * ------------------------------------------------------------------------- */
template <int TX_MODE>
__device__ static void tx_role(const dsb_ctx ctx, struct doca_gpu_eth_txq *txq, uint32_t mkey)
{
	unsigned long long next = 0;
	unsigned long long next_check = 0ull;
	uint64_t wqe_idx = 0, cqe_idx = 0; /* mode 2 */
	doca_gpu_dev_eth_ticket_t ticket;

	for (;;) {
		if (should_exit(ctx, next_check))
			break;
		uint32_t k = 0;
		while (k < DSB_TX_BATCH) {
			const uint32_t slot = (uint32_t)((next + k) & (DSB_RSP_CAP - 1u));
			const unsigned long long w = *(volatile unsigned long long *)(ctx.rsp_ready + slot);
			if (dsb_ready_seq(w) != next + k + 1ull)
				break;
			++k;
		}
		if (k == 0)
			continue;
		__threadfence();
		const unsigned long long now = gtimer();
		for (uint32_t j = 0; j < k; ++j) {
			const uint32_t slot = (uint32_t)((next + j) & (DSB_RSP_CAP - 1u));
			uint8_t *buf = ctx.rsp_buf + (size_t)slot * DSB_RSP_SLOT;
			if (buf[17] == DSB_MSG_RESULT) {
				const unsigned long long tend = *(volatile unsigned long long *)(ctx.rsp_tend + slot);
				reinterpret_cast<dsb_result *>(buf)->tx_ns = (uint32_t)(now > tend ? now - tend : 0ull);
			}
		}
		__threadfence_system(); /* reply bytes visible to the NIC before the doorbell */
		atomicAdd(&ctx.stats->tx_posted, (unsigned long long)k); /* counted before the send call: a hang inside it shows as posted > tx_frames */

		for (uint32_t j = 0; j < k; ++j) {
			const uint32_t slot = (uint32_t)((next + j) & (DSB_RSP_CAP - 1u));
			const unsigned long long w = *(volatile unsigned long long *)(ctx.rsp_ready + slot);
			const size_t len = dsb_ready_len(w);
			const uint64_t addr = (uint64_t)(ctx.rsp_buf + (size_t)slot * DSB_RSP_SLOT);
#ifndef DSB_TX_WAIT_SEND_ONLY
			if (TX_MODE == 0) {
				doca_gpu_dev_eth_txq_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
							  DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
							  DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
							  DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
					txq, addr, mkey, len, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
			} else if (TX_MODE == 2) {
				const enum doca_gpu_eth_send_flags flags = (j + 1 == k) ? DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY
										      : DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;
				struct doca_gpu_dev_eth_txq_wqe *wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);
				doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, (uint16_t)wqe_idx, addr, mkey, (uint32_t)len,
								      flags);
				++wqe_idx;
			} else
#endif
			{
				doca_gpu_dev_eth_txq_wait_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
							       DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
							       DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
							       DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
					txq, 0, addr, mkey, len, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
			}
		}

		uint32_t completed = 0;
		uint32_t retries = 0;
		doca_error_t status = DOCA_SUCCESS;
#ifndef DSB_TX_WAIT_SEND_ONLY
		if (TX_MODE == 2) {
			doca_gpu_dev_eth_txq_submit(txq, wqe_idx);
			do {
				status = doca_gpu_dev_eth_txq_poll_completion_at<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
										 DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>(
					txq, cqe_idx, DOCA_GPUNETIO_ETH_WAIT_FLAG_B);
				++retries;
			} while (status == DOCA_ERROR_AGAIN && retries < 20000000u);
			++cqe_idx;
		} else
#endif
		{
			do {
				status = doca_gpu_dev_eth_txq_poll_completion<DOCA_GPUNETIO_ETH_CQ_POLL_LAST>(
					txq, k, DOCA_GPUNETIO_ETH_WAIT_FLAG_B, &completed);
				++retries;
			} while (status == DOCA_ERROR_AGAIN && retries < 20000000u);
		}
		if (status != DOCA_SUCCESS) {
			atomicAdd(&ctx.stats->tx_errors, 1ull);
			ctx.stats->tx_last_error = (unsigned long long)status;
			DOCA_GPUNETIO_VOLATILE(*ctx.exit_flag) = 1;
			break;
		}
		next += k;
		*(volatile unsigned long long *)ctx.rsp_released = next;
		atomicAdd(&ctx.stats->tx_frames, (unsigned long long)k);
	}
}

/* RX and TX together in one small kernel of two 32-thread blocks, launched
 * before the workers on its own stream: the same two-kernel / two-stream
 * shape as dsb_bench "persistent" (gen_consume_kernel first, worker_kernel
 * second), which runs on this GPU. */
template <int TX_MODE>
__global__ void dsb_io_kernel(dsb_ctx ctx, struct doca_gpu_eth_rxq *rxq, struct doca_gpu_eth_txq *txq,
			      uint32_t mkey)
{
	if (blockIdx.x == 0) {
		rx_role(ctx, rxq);
	} else if (threadIdx.x == 0) {
		tx_role<TX_MODE>(ctx, txq, mkey);
	}
}

extern "C" {

int dsb_doca_max_blocks(uint32_t smem_bytes)
{
	int dev = 0, per_sm = 0, sms = 0;
	cudaGetDevice(&dev);
	const void *k = (const void *)worker_kernel;
	if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes) != cudaSuccess)
		return 0;
	if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, DSB_THREADS, smem_bytes) != cudaSuccess)
		return 0;
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
	return per_sm * sms;
}

doca_error_t dsb_doca_launch(cudaStream_t s_io, cudaStream_t s_work, struct dsb_ctx ctx,
			     struct doca_gpu_eth_rxq *rxq, struct doca_gpu_eth_txq *txq, uint32_t tx_mkey,
			     uint32_t workers, uint32_t tx_mode)
{
	cudaError_t r = cudaGetLastError();
	if (r != cudaSuccess) {
		DOCA_LOG_ERR("CUDA error before server launch: %s", cudaGetErrorString(r));
		return DOCA_ERROR_BAD_STATE;
	}
#ifdef DSB_TX_WAIT_SEND_ONLY
	if (tx_mode != 1) {
		DOCA_LOG_ERR("built with tx_wait_send_only: start the server with -W 1");
		return DOCA_ERROR_INVALID_VALUE;
	}
#endif
	/* RX + TX first, so they are resident before the workers take the rest of the GPU */
	if (tx_mode == 1)
		dsb_io_kernel<1><<<2, 32, 0, s_io>>>(ctx, rxq, txq, tx_mkey);
	else if (tx_mode == 2)
		dsb_io_kernel<2><<<2, 32, 0, s_io>>>(ctx, rxq, txq, tx_mkey);
	else
		dsb_io_kernel<0><<<2, 32, 0, s_io>>>(ctx, rxq, txq, tx_mkey);
	r = cudaGetLastError();
	if (r != cudaSuccess) {
		DOCA_LOG_ERR("RX/TX kernel launch failed: %s", cudaGetErrorString(r));
		return DOCA_ERROR_BAD_STATE;
	}
	worker_kernel<<<workers, DSB_THREADS, ctx.smem_bytes, s_work>>>(ctx);
	r = cudaGetLastError();
	if (r != cudaSuccess) {
		DOCA_LOG_ERR("worker kernel launch failed: %s", cudaGetErrorString(r));
		return DOCA_ERROR_BAD_STATE;
	}
	return DOCA_SUCCESS;
}

} /* extern "C" */
