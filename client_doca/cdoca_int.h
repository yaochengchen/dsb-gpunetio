#ifndef CDOCA_INT_H_
#define CDOCA_INT_H_
#include <stdint.h>
#include <cuda_runtime.h>
#define CD_SLOT        4096u
#define CD_TXR         1024u
#define CD_RXR         4096u
#define CD_RX_PKT_NUM  16384u
#define CD_RX_BURST    64u
#define CD_SQ_DEPTH    4096u
struct cd_shared {                         /* host-mapped, shared CPU <-> GPU */
	volatile uint32_t exit_flag;           /* 1 host stop, 2 rx error, 3 tx error */
	volatile uint32_t pad;
	volatile unsigned long long tx_seq[CD_TXR];
	volatile uint32_t tx_len[CD_TXR];
	volatile unsigned long long tx_done;
	volatile unsigned long long rx_seq[CD_RXR];
	volatile uint32_t rx_len[CD_RXR];
	volatile unsigned long long rx_cons;
	volatile unsigned long long rx_frames, rx_dropped, tx_frames, tx_errors;
};
struct doca_gpu_eth_rxq;
struct doca_gpu_eth_txq;
#ifdef __cplusplus
extern "C" {
#endif
int cd_launch(cudaStream_t srx, cudaStream_t stx, struct doca_gpu_eth_rxq *rxq, struct doca_gpu_eth_txq *txq,
	      uint32_t mkey, uint8_t *gbuf, struct cd_shared *sh, uint8_t *txd, uint8_t *rxd);
#ifdef __cplusplus
}
#endif
#endif
