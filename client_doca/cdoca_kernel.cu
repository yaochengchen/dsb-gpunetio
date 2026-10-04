#include <stdio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include "cdoca_int.h"

/* RX: one warp, same recv call as TVG receiver/device/recv_kernel.cu */
__device__ void cd_rx_kernel(struct doca_gpu_eth_rxq *rxq, struct cd_shared *sh, uint8_t *rxd)
{
	__shared__ uint64_t first;
	__shared__ uint32_t num;
	__shared__ struct doca_gpu_dev_eth_rxq_attr attr[CD_RX_BURST];
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long tail = 0;
	for (;;) {
		int quit = (lane == 0) ? (int)sh->exit_flag : 0;
		if (__shfl_sync(0xFFFFFFFFu, quit, 0))
			break;
		doca_error_t ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP,
							     DOCA_GPUNETIO_ETH_MCST_AUTO,
							     DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
							     DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(rxq, CD_RX_BURST, 5000, &first, &num, attr);
		if (ret != DOCA_SUCCESS) {
			if (lane == 0)
				sh->exit_flag = 2;
			break;
		}
		__syncwarp();
		const uint32_t cnt = num;
		if (cnt == 0)
			continue;
		for (uint32_t i = 0; i < cnt; ++i) {
			const uint4 *src = (const uint4 *)doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, first + i);
			uint32_t bytes = attr[i].bytes;
			if (bytes > CD_SLOT)
				bytes = CD_SLOT;
			int room = 0;
			if (lane == 0)
				room = (tail - sh->rx_cons) < CD_RXR;
			if (!__shfl_sync(0xFFFFFFFFu, room, 0)) {
				if (lane == 0)
					sh->rx_dropped += 1;
				continue;
			}
			const uint32_t slot = (uint32_t)(tail & (CD_RXR - 1u));
			uint4 *dst = (uint4 *)(rxd + (size_t)slot * CD_SLOT);
			for (uint32_t q = lane; q < ((bytes + 15u) >> 4); q += 32u)
				dst[q] = __ldcg(src + q);
			__threadfence_system();
			__syncwarp();
			if (lane == 0) {
				sh->rx_len[slot] = bytes;
				__threadfence_system();
				sh->rx_seq[slot] = tail + 1ull;
			}
			++tail;
		}
		if (lane == 0)
			sh->rx_frames += cnt;
		__syncwarp();
	}
}

/* TX: one warp copies the frame into the registered GPU buffer, lane 0 sends it
 * with wait_send(ts = 0) + NOTIFY + poll_completion, exactly as TVG's sender. */
__device__ void cd_tx_kernel(struct doca_gpu_eth_txq *txq, uint32_t mkey, uint8_t *gbuf, struct cd_shared *sh,
			     const uint8_t *txd)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long next = 0;
	doca_gpu_dev_eth_ticket_t ticket;
	for (;;) {
		const uint32_t slot = (uint32_t)(next & (CD_TXR - 1u));
		int go = 0;
		uint32_t len = 0;
		if (lane == 0) {
			for (;;) {
				if (sh->exit_flag) { go = -1; break; }
				if (sh->tx_seq[slot] == next + 1ull) {
					__threadfence_system();
					len = sh->tx_len[slot];
					sh->pad = 1;
					go = 1;
					break;
				}
			}
		}
		go = __shfl_sync(0xFFFFFFFFu, go, 0);
		if (go < 0)
			break;
		len = __shfl_sync(0xFFFFFFFFu, len, 0);
		if (len > CD_SLOT)
			len = CD_SLOT;
		const uint4 *src = (const uint4 *)(txd + (size_t)slot * CD_SLOT);
		uint4 *dst = (uint4 *)(gbuf + (size_t)slot * CD_SLOT);
		for (uint32_t q = lane; q < ((len + 15u) >> 4); q += 32u)
			dst[q] = __ldcv(src + q);
		__threadfence_system();
		__syncwarp();
		int err = 0;
		if (lane == 0) {
			doca_gpu_dev_eth_txq_wait_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA,
						       DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA,
						       DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
						       DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
				txq, 0, (uint64_t)dst, mkey, len, DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY, &ticket);
			sh->pad = 2;
			uint32_t done = 0, retries = 0;
			doca_error_t st;
			do {
				st = doca_gpu_dev_eth_txq_poll_completion<DOCA_GPUNETIO_ETH_CQ_POLL_LAST>(
					txq, 1, DOCA_GPUNETIO_ETH_WAIT_FLAG_B, &done);
				++retries;
			} while (st == DOCA_ERROR_AGAIN && retries < 20000000u);
			if (st != DOCA_SUCCESS) {
				sh->tx_errors += 1;
				sh->exit_flag = 3;
				err = 1;
			} else {
				sh->tx_frames += 1;
				sh->tx_done = next + 1ull;
			}
		}
		if (__shfl_sync(0xFFFFFFFFu, err, 0))
			break;
		++next;
	}
}

__global__ void cd_kernel(struct doca_gpu_eth_rxq *rxq, struct doca_gpu_eth_txq *txq, uint32_t mkey, uint8_t *gbuf,
			  struct cd_shared *sh, uint8_t *txd, uint8_t *rxd)
{
	if (blockIdx.x == 0)
		cd_rx_kernel(rxq, sh, rxd);
	else
		cd_tx_kernel(txq, mkey, gbuf, sh, txd);
}

extern "C" int cd_launch(cudaStream_t srx, cudaStream_t stx, struct doca_gpu_eth_rxq *rxq,
			 struct doca_gpu_eth_txq *txq, uint32_t mkey, uint8_t *gbuf, struct cd_shared *sh,
			 uint8_t *txd, uint8_t *rxd)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) {
		fprintf(stderr, "[cdoca] CUDA error before launch: %s\n", cudaGetErrorString(e));
		return -1;
	}
	(void)stx;
	cd_kernel<<<2, 32, 0, srx>>>(rxq, txq, mkey, gbuf, sh, txd, rxd);
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		fprintf(stderr, "[cdoca] launch: %s\n", cudaGetErrorString(e));
		return -1;
	}
	return 0;
}
