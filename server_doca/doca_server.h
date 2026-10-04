/*
 * doca_server.h -- dSB optimization service on NVIDIA DOCA GPUNetIO.
 *
 * Data path (no CPU after start-up):
 *
 *   ConnectX-7 --GPUDirect RDMA--> RX cyclic buffer (GPU memory)
 *        RX warp   (block 0): polls the CQ, steers LOAD fragments into the
 *                             model store, turns SOLVE frames into work items
 *        workers   (blocks 2..): one request per CTA, solved in shared memory
 *        TX thread (block 1): posts finished replies from the reply ring to
 *                             the send queue, polls completions, recycles slots
 *   reply ring (GPU memory, registered with the NIC) --> ConnectX-7
 *
 * The roles are two persistent kernels on two streams (see
 * doca_kernel.cu).  The RX set-up (DOCA Flow steering by Ethertype, cyclic
 * RX queue, dmabuf-registered buffer) follows the TVG receiver in
 * TVG-suite-two-servers; the TX set-up follows the official DOCA sample
 * gpunetio_simple_send.
 */
#ifndef DOCA_SERVER_H_
#define DOCA_SERVER_H_

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_error.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_buf_array.h>
#include <doca_gpunetio_eth_def.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_argp.h>

#include "dsb_proto.h"
#include "dsb_queue.h"

#define MAX_PCI_ADDRESS_LEN 32U
#define DSB_RX_PKT_NUM      16384u     /* RX cyclic ring depth (>> DSB_WORK_CAP) */
#define DSB_RX_BURST        64u        /* max frames per recv call (small: latency) */
#define DSB_TX_SQ_DEPTH     4096u
#define DSB_TX_BATCH        32u        /* replies per doorbell / completion poll */

extern volatile sig_atomic_t force_quit;

struct dsb_doca_cfg {
	char gpu_pcie_addr[MAX_PCI_ADDRESS_LEN];
	char nic_pcie_addr[MAX_PCI_ADDRESS_LEN];
	int cuda_id;
	uint32_t smem_kb;       /* dynamic shared memory per worker CTA (0 = 48) */
	uint32_t workers;       /* 0 = as many as can be resident */
	uint32_t tx_mode;       /* 0: txq_send (official sample, THREAD scope)
	                         * 1: wait_send(ts=0) (TVG transmitter call)
	                         * 2: wqe_prepare_send + submit + poll_completion_at (sample, shared_qp=off) */
	uint32_t report_ms;
};

struct dsb_doca_io {
	struct doca_gpu *gpu_dev;
	struct doca_dev *ddev;
	struct doca_flow_port *port;

	/* RX */
	struct doca_ctx *rxq_ctx;
	struct doca_eth_rxq *rxq_cpu;
	struct doca_gpu_eth_rxq *rxq_gpu;
	struct doca_mmap *rx_mmap;
	uint8_t *rx_buf;
	int rx_dmabuf_fd;
	struct doca_flow_pipe *rx_pipe;
	struct doca_flow_pipe_entry *rx_entry;

	/* TX */
	struct doca_ctx *txq_ctx;
	struct doca_eth_txq *txq_cpu;
	struct doca_gpu_eth_txq *txq_gpu;
	struct doca_mmap *tx_mmap;
	uint8_t *tx_buf;          /* == ctx.rsp_buf */
	int tx_dmabuf_fd;
	uint32_t tx_mkey;         /* big-endian, as the GPU send wants it */
};

doca_error_t dsb_doca_run(struct dsb_doca_cfg *cfg);

#if __cplusplus
extern "C" {
#endif

/* Number of worker CTAs that can be resident at once with smem_bytes each. */
int dsb_doca_max_blocks(uint32_t smem_bytes);

/* Launches the RX/TX kernel (2 blocks x 32 threads) on s_io, then `workers`
 * worker CTAs on s_work. */
doca_error_t dsb_doca_launch(cudaStream_t s_io, cudaStream_t s_work, struct dsb_ctx ctx,
			     struct doca_gpu_eth_rxq *rxq, struct doca_gpu_eth_txq *txq, uint32_t tx_mkey,
			     uint32_t workers, uint32_t tx_mode);

#if __cplusplus
}
#endif

#endif /* DOCA_SERVER_H_ */
