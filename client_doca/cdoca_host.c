/* Host set-up copied from TVG: receiver/host/recv_host.c (flow + cyclic RXQ)
 * and generator/host/tvg_host.c (GPU TXQ with wait-on-time, dmabuf buffer). */
#include <arpa/inet.h>
#include <endian.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_flow.h>
#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_mmap.h>
#include <doca_log.h>
#include "common.h"
#include "dsb_proto.h"
#include "cdoca.h"
#include "cdoca_int.h"

#define FLOW_NB_COUNTERS 524288
#define ALIGN_SIZE(size, align) (size = (((size) + (align) - 1) / (align)) * (align))
#define CK(x, msg) do { doca_error_t r_ = (x); if (r_ != DOCA_SUCCESS) { \
	fprintf(stderr, "[cdoca] %s: %s\n", msg, doca_error_get_descr(r_)); return -1; } } while (0)

static struct {
	struct doca_dev *ddev;
	struct doca_gpu *gpu;
	struct doca_flow_port *port;
	struct doca_eth_rxq *rxq; struct doca_ctx *rxq_ctx; struct doca_gpu_eth_rxq *rxq_gpu;
	struct doca_mmap *rx_mmap; uint8_t *rx_buf; int rx_fd;
	struct doca_flow_pipe *pipe; struct doca_flow_pipe_entry *entry;
	struct doca_eth_txq *txq; struct doca_ctx *txq_ctx; struct doca_gpu_eth_txq *txq_gpu;
	struct doca_mmap *tx_mmap; uint8_t *tx_buf; int tx_fd; uint32_t mkey;
	struct cd_shared *sh, *sh_dev;
	uint8_t *txd, *txd_dev, *rxd, *rxd_dev;
	cudaStream_t s_rx, s_tx;
	unsigned long long tx_head, rx_cons;
} C;

static size_t pg(void) { long r = sysconf(_SC_PAGESIZE); return r == -1 ? 4096 : (size_t)r; }

static int reg_gpu(size_t bytes, uint8_t **addr, int *fd, struct doca_mmap **mm)
{
	CK(doca_mmap_create(mm), "mmap create");
	CK(doca_mmap_add_dev(*mm, C.ddev), "mmap add dev");
	CK(doca_gpu_mem_alloc(C.gpu, bytes, pg(), DOCA_GPU_MEM_TYPE_GPU, (void **)addr, NULL), "gpu mem alloc");
	if (doca_gpu_dmabuf_fd(C.gpu, *addr, bytes, fd) != DOCA_SUCCESS) {
		*fd = -1;
		CK(doca_mmap_set_memrange(*mm, *addr, bytes), "mmap memrange");
	} else {
		CK(doca_mmap_set_dmabuf_memrange(*mm, *fd, *addr, 0, bytes), "mmap dmabuf memrange");
	}
	CK(doca_mmap_set_permissions(*mm, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE), "mmap permissions");
	CK(doca_mmap_start(*mm), "mmap start");
	return 0;
}

static int init_flow(void)
{
	struct doca_flow_cfg *fc;
	struct doca_flow_port_cfg *pc;
	CK(doca_flow_cfg_create(&fc), "flow cfg");
	CK(doca_flow_cfg_set_pipe_queues(fc, 1), "flow queues");
	CK(doca_flow_cfg_set_mode_args(fc, "vnf,hws=off"), "flow mode");
	CK(doca_flow_cfg_set_nr_counters(fc, FLOW_NB_COUNTERS), "flow counters");
	CK(doca_flow_init(fc), "flow init");
	doca_flow_cfg_destroy(fc);
	CK(doca_flow_port_cfg_create(&pc), "port cfg");
	CK(doca_flow_port_cfg_set_port_id(pc, 0), "port id");
	CK(doca_flow_port_cfg_set_dev(pc, C.ddev), "port dev");
	CK(doca_flow_port_start(pc, &C.port), "port start");
	doca_flow_port_cfg_destroy(pc);
	return 0;
}

static int create_rx(int cuda_id)
{
	uint32_t bytes = 0;
	struct cudaDeviceProp prop;
	union doca_data ud;
	CK(doca_eth_rxq_create(C.ddev, CD_RX_PKT_NUM, CD_SLOT, &C.rxq), "rxq create");
	CK(doca_eth_rxq_set_type(C.rxq, DOCA_ETH_RXQ_TYPE_CYCLIC), "rxq type");
	CK(doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, CD_SLOT, CD_RX_PKT_NUM, 0, 0, 0, &bytes),
	   "rxq estimate");
	ALIGN_SIZE(bytes, pg());
	if (reg_gpu(bytes, &C.rx_buf, &C.rx_fd, &C.rx_mmap))
		return -1;
	CK(doca_eth_rxq_set_pkt_buf(C.rxq, C.rx_mmap, 0, bytes), "rxq pkt buf");
	cudaGetDeviceProperties(&prop, cuda_id);
	if (prop.major < 9)
		CK(doca_eth_rxq_gpu_enable_mcst_qp(C.rxq), "mcst qp");
	C.rxq_ctx = doca_eth_rxq_as_doca_ctx(C.rxq);
	CK(doca_ctx_set_datapath_on_gpu(C.rxq_ctx, C.gpu), "rxq datapath");
	ud.ptr = &C;
	CK(doca_ctx_set_user_data(C.rxq_ctx, ud), "rxq user data");
	CK(doca_ctx_start(C.rxq_ctx), "rxq start");
	CK(doca_eth_rxq_get_gpu_handle(C.rxq, &C.rxq_gpu), "rxq gpu handle");

	/* root pipe: Ethertype -> RSS queue 0 (as recv_host.c) */
	struct doca_flow_match match = {0}, mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_pipe_cfg *pcfg;
	uint16_t q[1] = {0};
	struct doca_flow_monitor mon = {.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED};
	doca_eth_rxq_apply_queue_id(C.rxq, 0);
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array = q;
	fwd.rss.outer_flags = 0;
	fwd.rss.nr_queues = 1;
	mask.outer.eth.type = 0xFFFF;
	CK(doca_flow_pipe_cfg_create(&pcfg, C.port), "pipe cfg");
	CK(doca_flow_pipe_cfg_set_name(pcfg, "DSB_CLIENT_RX"), "pipe name");
	CK(doca_flow_pipe_cfg_set_type(pcfg, DOCA_FLOW_PIPE_BASIC), "pipe type");
	CK(doca_flow_pipe_cfg_set_is_root(pcfg, true), "pipe root");
	CK(doca_flow_pipe_cfg_set_match(pcfg, NULL, &mask), "pipe match");
	CK(doca_flow_pipe_cfg_set_monitor(pcfg, &mon), "pipe monitor");
	CK(doca_flow_pipe_create(pcfg, &fwd, NULL, &C.pipe), "pipe create");
	doca_flow_pipe_cfg_destroy(pcfg);
	match.outer.eth.type = htons(DSB_ETHERTYPE);
	CK(doca_flow_pipe_basic_add_entry(0, C.pipe, &match, 0, NULL, NULL, NULL, DOCA_FLOW_ENTRY_FLAGS_NO_WAIT, NULL,
					  &C.entry), "pipe entry");
	CK(doca_flow_entries_process(C.port, 0, 0, 0), "entries process");
	return 0;
}

static int create_tx(void)
{
	enum doca_eth_wait_on_time_type mode;
	CK(doca_eth_txq_cap_get_wait_on_time_offload_supported(doca_dev_as_devinfo(C.ddev), &mode), "wait-on-time cap");
	if (mode != DOCA_ETH_WAIT_ON_TIME_TYPE_NATIVE) {
		fprintf(stderr, "[cdoca] NIC has no native wait-on-time\n");
		return -1;
	}
	CK(doca_eth_txq_create(C.ddev, CD_SQ_DEPTH, &C.txq), "txq create");
	CK(doca_eth_txq_set_l3_chksum_offload(C.txq, 0), "txq l3");
	CK(doca_eth_txq_set_l4_chksum_offload(C.txq, 0), "txq l4");
	CK(doca_eth_txq_gpu_set_completion_on_gpu(C.txq), "txq completion on gpu");
	C.txq_ctx = doca_eth_txq_as_doca_ctx(C.txq);
	CK(doca_ctx_set_datapath_on_gpu(C.txq_ctx, C.gpu), "txq datapath");
	CK(doca_eth_txq_set_wait_on_time_offload(C.txq), "txq wait-on-time");
	CK(doca_ctx_start(C.txq_ctx), "txq start");
	CK(doca_eth_txq_apply_queue_id(C.txq, 0), "txq queue id");
	CK(doca_eth_txq_get_gpu_handle(C.txq, &C.txq_gpu), "txq gpu handle");
	if (reg_gpu((size_t)CD_TXR * CD_SLOT, &C.tx_buf, &C.tx_fd, &C.tx_mmap))
		return -1;
	CK(doca_mmap_get_mkey(C.tx_mmap, C.ddev, &C.mkey), "mkey");
	C.mkey = htobe32(C.mkey);
	return 0;
}

int cdoca_open(const char *gpu_pci, const char *nic_pci, uint8_t mac[6])
{
	struct doca_log_backend *sdk_log;
	int cuda_id = 0;
	memset(&C, 0, sizeof(C));
	doca_log_backend_create_standard();
	if (doca_log_backend_create_with_file_sdk(stderr, &sdk_log) == DOCA_SUCCESS)
		doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (cudaDeviceGetByPCIBusId(&cuda_id, gpu_pci) != cudaSuccess || cudaSetDevice(cuda_id) != cudaSuccess ||
	    cudaFree(0) != cudaSuccess) {
		fprintf(stderr, "[cdoca] bad GPU %s\n", gpu_pci);
		return -1;
	}
	CK(open_doca_device_with_pci((char *)nic_pci, NULL, &C.ddev), "open NIC");
	CK(doca_gpu_create(gpu_pci, &C.gpu), "doca_gpu_create");
	if (init_flow() || create_rx(cuda_id) || create_tx())
		return -1;
	CK(doca_devinfo_get_mac_addr(doca_dev_as_devinfo(C.ddev), mac, 6), "get MAC");

	if (cudaHostAlloc((void **)&C.sh, sizeof(struct cd_shared), cudaHostAllocMapped) != cudaSuccess ||
	    cudaHostAlloc((void **)&C.txd, (size_t)CD_TXR * CD_SLOT, cudaHostAllocMapped) != cudaSuccess ||
	    cudaHostAlloc((void **)&C.rxd, (size_t)CD_RXR * CD_SLOT, cudaHostAllocMapped) != cudaSuccess) {
		fprintf(stderr, "[cdoca] cudaHostAlloc failed\n");
		return -1;
	}
	memset((void *)C.sh, 0, sizeof(struct cd_shared));
	cudaHostGetDevicePointer((void **)&C.sh_dev, (void *)C.sh, 0);
	cudaHostGetDevicePointer((void **)&C.txd_dev, C.txd, 0);
	cudaHostGetDevicePointer((void **)&C.rxd_dev, C.rxd, 0);
	cudaStreamCreateWithFlags(&C.s_rx, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&C.s_tx, cudaStreamNonBlocking);
	if (cd_launch(C.s_rx, C.s_tx, C.rxq_gpu, C.txq_gpu, C.mkey, C.tx_buf, C.sh_dev, C.txd_dev, C.rxd_dev))
		return -1;
	usleep(100000);
	fprintf(stderr, "[cdoca] GPUNetIO client up: GPU %s NIC %s\n", gpu_pci, nic_pci);
	return 0;
}

int cdoca_send(const uint8_t *f, uint32_t len)
{
	if (len > CD_SLOT)
		return 0;
	while (C.tx_head - C.sh->tx_done >= CD_TXR)
		if (C.sh->exit_flag)
			return 0;
	if (C.sh->exit_flag)
		return 0;
	const uint32_t slot = (uint32_t)(C.tx_head & (CD_TXR - 1u));
	memcpy(C.txd + (size_t)slot * CD_SLOT, f, len);
	C.sh->tx_len[slot] = len;
	__sync_synchronize();
	C.sh->tx_seq[slot] = C.tx_head + 1ull;
	C.tx_head++;
	return 1;
}

int cdoca_recv(uint8_t *b, uint32_t cap)
{
	const uint32_t slot = (uint32_t)(C.rx_cons & (CD_RXR - 1u));
	if (C.sh->rx_seq[slot] != C.rx_cons + 1ull)
		return 0;
	__sync_synchronize();
	const uint32_t len = C.sh->rx_len[slot];
	memcpy(b, C.rxd + (size_t)slot * CD_SLOT, len < cap ? len : cap);
	C.rx_cons++;
	__sync_synchronize();
	C.sh->rx_cons = C.rx_cons;
	return (int)len;
}

void cdoca_close(void)
{
	if (!C.sh)
		return;
	fprintf(stderr, "[cdoca] rx_frames %llu rx_dropped %llu tx_frames %llu tx_errors %llu exit_flag %u\n",
		C.sh->rx_frames, C.sh->rx_dropped, C.sh->tx_frames, C.sh->tx_errors, C.sh->exit_flag);
	fprintf(stderr, "[cdoca] host queued %llu frames, GPU TX stage %u (0=never saw a frame, 1=saw frame, 2=wait_send returned)\n",
		C.tx_head, C.sh->pad);
	C.sh->exit_flag = 1;
	cudaStreamSynchronize(C.s_rx);
	cudaStreamSynchronize(C.s_tx);
	fprintf(stderr, "[cdoca] after stop: tx_frames %llu tx_errors %llu stage %u\n", C.sh->tx_frames, C.sh->tx_errors, C.sh->pad);
	if (C.pipe) doca_flow_pipe_destroy(C.pipe);
	if (C.port) doca_flow_port_stop(C.port);
	doca_flow_destroy();
	if (C.rxq_ctx) doca_ctx_stop(C.rxq_ctx);
	if (C.rxq) doca_eth_rxq_destroy(C.rxq);
	if (C.rx_mmap) doca_mmap_destroy(C.rx_mmap);
	if (C.rx_buf) doca_gpu_mem_free(C.gpu, C.rx_buf);
	if (C.txq_ctx) doca_ctx_stop(C.txq_ctx);
	if (C.txq) doca_eth_txq_destroy(C.txq);
	if (C.tx_mmap) doca_mmap_destroy(C.tx_mmap);
	if (C.tx_buf) doca_gpu_mem_free(C.gpu, C.tx_buf);
	if (C.gpu) doca_gpu_destroy(C.gpu);
	if (C.ddev) doca_dev_close(C.ddev);
	cudaFreeHost((void *)C.sh);
	cudaFreeHost(C.txd);
	cudaFreeHost(C.rxd);
	C.sh = NULL;
}
