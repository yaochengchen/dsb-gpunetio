/*
 * gen_host.c -- host side of the streaming generator (PU3, GH200 GPU 1).
 *
 *   sudo ./bin/aarch64/st_gen -g <GPU 1 PCIe> -n 0002:01:00.0 -m 58:a2:e1:db:db:80 \
 *        -N 10000000 -K 20 -q 4 -i mattis -s 42 -L 380 -o results/stream/gen.csv
 *
 * Default send path -W 2 = the TVG transmitter's 512-frame BLOCK-scope burst (gen.h).
 *
 * Set-up is client_doca/cdoca_host.c (copied from the TVG's tvg_host.c and
 * recv_host.c): NQ GPU send queues with wait-on-time offload and completions
 * on the GPU, one dmabuf registered send buffer; one cyclic GPU receive queue
 * steered by DOCA Flow on Ethertype ST_ETHERTYPE for the PASS_ACK.  The host
 * only prints counters, read from the host-mapped mirror (never managed memory).
 */
#include <getopt.h>
#include <endian.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
#include "gen.h"
#include "st_net.h"

DOCA_LOG_REGISTER(ST::GEN);

#define ALIGN_SIZE(size, align) (size = (((size) + (align) - 1) / (align)) * (align))
#define FLOW_NB_COUNTERS 524288
#define CK(x, msg)                                                                                 \
	do {                                                                                       \
		doca_error_t r_ = (x);                                                             \
		if (r_ != DOCA_SUCCESS) {                                                          \
			DOCA_LOG_ERR("%s: %s", msg, doca_error_get_descr(r_));                     \
			return r_;                                                                 \
		}                                                                                  \
	} while (0)

static volatile sig_atomic_t force_quit;
static void on_signal(int s) { (void)s; force_quit = 1; }
static size_t pg(void) { long r = sysconf(_SC_PAGESIZE); return r == -1 ? 4096 : (size_t)r; }

struct txq_set {
	struct doca_ctx *ctx;
	struct doca_eth_txq *q;
	struct doca_gpu_eth_txq *gpu;
};

static struct {
	struct doca_dev *ddev;
	struct doca_gpu *gpu;
	struct doca_flow_port *port;
	struct doca_flow_pipe *pipe;
	struct doca_flow_pipe_entry *entry;
	struct txq_set tx[ST_MAX_Q];
	struct doca_mmap *tx_mmap;
	uint8_t *tx_buf;
	int tx_fd;
	uint32_t mkey;
	struct doca_ctx *rxq_ctx;
	struct doca_eth_rxq *rxq;
	struct doca_gpu_eth_rxq *rxq_gpu;
	struct doca_mmap *rx_mmap;
	uint8_t *rx_buf;
	int rx_fd;
	uint8_t my_mac[6];
} D;

static doca_error_t reg_gpu(size_t bytes, uint8_t **addr, int *fd, struct doca_mmap **mm)
{
	ALIGN_SIZE(bytes, pg());   /* doca_gpu_mem_alloc wants a multiple of the host page size (64 KiB on GH200) */
	CK(doca_mmap_create(mm), "mmap create");
	CK(doca_mmap_add_dev(*mm, D.ddev), "mmap add dev");
	CK(doca_gpu_mem_alloc(D.gpu, bytes, pg(), DOCA_GPU_MEM_TYPE_GPU, (void **)addr, NULL), "gpu mem alloc");
	if (doca_gpu_dmabuf_fd(D.gpu, *addr, bytes, fd) != DOCA_SUCCESS) {
		*fd = -1;
		CK(doca_mmap_set_memrange(*mm, *addr, bytes), "mmap memrange");
	} else {
		CK(doca_mmap_set_dmabuf_memrange(*mm, *fd, *addr, 0, bytes), "mmap dmabuf memrange");
	}
	CK(doca_mmap_set_permissions(*mm, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE), "mmap permissions");
	CK(doca_mmap_start(*mm), "mmap start");
	return DOCA_SUCCESS;
}

static doca_error_t init_flow(void)
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
	CK(doca_flow_port_cfg_set_dev(pc, D.ddev), "port dev");
	CK(doca_flow_port_start(pc, &D.port), "port start");
	doca_flow_port_cfg_destroy(pc);
	return DOCA_SUCCESS;
}

/* ACK receive queue: cdoca_host.c create_rx (= TVG recv_host.c): Ethertype -> RSS queue 0 */
static doca_error_t create_ack_rxq(int cuda_id)
{
	uint32_t bytes = 0;
	struct cudaDeviceProp prop;
	union doca_data ud;
	CK(doca_eth_rxq_create(D.ddev, ST_GEN_RX_PKTS, 256, &D.rxq), "rxq create");
	CK(doca_eth_rxq_set_type(D.rxq, DOCA_ETH_RXQ_TYPE_CYCLIC), "rxq type");
	CK(doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, 256, ST_GEN_RX_PKTS, 0, 0, 0, &bytes),
	   "rxq estimate");
	ALIGN_SIZE(bytes, pg());
	CK(reg_gpu(bytes, &D.rx_buf, &D.rx_fd, &D.rx_mmap), "rx buffer");
	CK(doca_eth_rxq_set_pkt_buf(D.rxq, D.rx_mmap, 0, bytes), "rxq pkt buf");
	cudaGetDeviceProperties(&prop, cuda_id);
	if (prop.major < 9)
		CK(doca_eth_rxq_gpu_enable_mcst_qp(D.rxq), "mcst qp");
	D.rxq_ctx = doca_eth_rxq_as_doca_ctx(D.rxq);
	CK(doca_ctx_set_datapath_on_gpu(D.rxq_ctx, D.gpu), "rxq datapath");
	ud.ptr = &D;
	CK(doca_ctx_set_user_data(D.rxq_ctx, ud), "rxq user data");
	CK(doca_ctx_start(D.rxq_ctx), "rxq start");
	CK(doca_eth_rxq_get_gpu_handle(D.rxq, &D.rxq_gpu), "rxq gpu handle");

	struct doca_flow_match match = {0}, mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_pipe_cfg *pcfg;
	uint16_t q[1] = {0};
	struct doca_flow_monitor mon = {.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED};
	doca_eth_rxq_apply_queue_id(D.rxq, 0);
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array = q;
	fwd.rss.outer_flags = 0;
	fwd.rss.nr_queues = 1;
	mask.outer.eth.type = 0xFFFF;
	CK(doca_flow_pipe_cfg_create(&pcfg, D.port), "pipe cfg");
	CK(doca_flow_pipe_cfg_set_name(pcfg, "ST_GEN_ACK_RX"), "pipe name");
	CK(doca_flow_pipe_cfg_set_type(pcfg, DOCA_FLOW_PIPE_BASIC), "pipe type");
	CK(doca_flow_pipe_cfg_set_is_root(pcfg, true), "pipe root");
	CK(doca_flow_pipe_cfg_set_match(pcfg, NULL, &mask), "pipe match");
	CK(doca_flow_pipe_cfg_set_monitor(pcfg, &mon), "pipe monitor");
	CK(doca_flow_pipe_create(pcfg, &fwd, NULL, &D.pipe), "pipe create");
	doca_flow_pipe_cfg_destroy(pcfg);
	match.outer.eth.type = htons(ST_ETHERTYPE);
	CK(doca_flow_pipe_basic_add_entry(0, D.pipe, &match, 0, NULL, NULL, NULL, DOCA_FLOW_ENTRY_FLAGS_NO_WAIT, NULL, &D.entry),
	   "pipe entry");
	CK(doca_flow_entries_process(D.port, 0, 0, 0), "entries process");
	return DOCA_SUCCESS;
}

/* send queues: cdoca_host.c create_tx (= TVG tvg_host.c), NQ of them */
static doca_error_t create_txqs(uint32_t nq, size_t tx_bytes, uint32_t tx_mode)
{
	if (tx_mode >= 1) {   /* wait_send (modes 1 and 2) runs on wait-on-time queues, as in the TVG and client_doca */
		enum doca_eth_wait_on_time_type mode;
		CK(doca_eth_txq_cap_get_wait_on_time_offload_supported(doca_dev_as_devinfo(D.ddev), &mode), "wait-on-time cap");
		if (mode != DOCA_ETH_WAIT_ON_TIME_TYPE_NATIVE) {
			DOCA_LOG_ERR("NIC has no native wait-on-time (-W 1 and -W 2 need it)");
			return DOCA_ERROR_NOT_SUPPORTED;
		}
	}
	for (uint32_t q = 0; q < nq; ++q) {
		struct txq_set *T = &D.tx[q];
		CK(doca_eth_txq_create(D.ddev, ST_GEN_SQ_DEPTH, &T->q), "txq create");
		CK(doca_eth_txq_set_l3_chksum_offload(T->q, 0), "txq l3");
		CK(doca_eth_txq_set_l4_chksum_offload(T->q, 0), "txq l4");
		CK(doca_eth_txq_gpu_set_completion_on_gpu(T->q), "txq completion on gpu");
		T->ctx = doca_eth_txq_as_doca_ctx(T->q);
		if (!T->ctx)
			return DOCA_ERROR_BAD_STATE;
		CK(doca_ctx_set_datapath_on_gpu(T->ctx, D.gpu), "txq datapath");
		if (tx_mode >= 1)
			CK(doca_eth_txq_set_wait_on_time_offload(T->q), "txq wait-on-time");
		CK(doca_ctx_start(T->ctx), "txq start");
		CK(doca_eth_txq_apply_queue_id(T->q, (uint16_t)q), "txq queue id");
		CK(doca_eth_txq_get_gpu_handle(T->q, &T->gpu), "txq gpu handle");
	}
	CK(reg_gpu(tx_bytes, &D.tx_buf, &D.tx_fd, &D.tx_mmap), "tx buffer");
	CK(doca_mmap_get_mkey(D.tx_mmap, D.ddev, &D.mkey), "mkey");
	D.mkey = htobe32(D.mkey);
	return DOCA_SUCCESS;
}

static void destroy_all(uint32_t nq)
{
	if (D.pipe) doca_flow_pipe_destroy(D.pipe);
	if (D.port) doca_flow_port_stop(D.port);
	doca_flow_destroy();
	if (D.rxq_ctx) doca_ctx_stop(D.rxq_ctx);
	if (D.rxq) doca_eth_rxq_destroy(D.rxq);
	if (D.rx_mmap) doca_mmap_destroy(D.rx_mmap);
	if (D.rx_buf) doca_gpu_mem_free(D.gpu, D.rx_buf);
	for (uint32_t q = 0; q < nq; ++q) {
		if (D.tx[q].ctx) doca_ctx_stop(D.tx[q].ctx);
		if (D.tx[q].q) doca_eth_txq_destroy(D.tx[q].q);
	}
	if (D.tx_mmap) doca_mmap_destroy(D.tx_mmap);
	if (D.tx_buf) doca_gpu_mem_free(D.gpu, D.tx_buf);
	if (D.gpu) doca_gpu_destroy(D.gpu);
	if (D.ddev) doca_dev_close(D.ddev);
}

static void usage(void)
{
	fprintf(stderr,
		"st_gen: streaming generator (GPUNetIO, TVG transmitter path)\n"
		"  -g GPU PCIe  -n NIC PCIe (0002:01:00.0)  -m receiver MAC (58:a2:e1:db:db:80)\n"
		"  -N n  -K steps (sends K+1 passes)  -q send queues (1..16, default 4)  -i random|mattis  -s seed\n"
		"  -S synthesis blocks (default 96)  -R ring slots per queue (default 2048)  -b burst for -W 0/1 (default 64)\n"
		"  -L target payload rate in Gb/s (token pacing per queue and burst; 0 = unpaced; e.g. 380 leaves ~4%% under the link)\n"
		"  -W send path: 2 TVG 512-frame BLOCK-scope bursts (default); 0 txq_send per frame; 1 wait_send per frame\n"
		"  -z sleep ns after each batch (crude rate limit, default 0)  -r report ms  -o CSV\n");
}

int main(int argc, char **argv)
{
	char gpu[32] = "0000:00:00.0", nic[32] = "0002:01:00.0";
	uint8_t dst_mac[6] = {0x58, 0xa2, 0xe1, 0xdb, 0xdb, 0x80};   /* PU2 ens7np0 */
	setvbuf(stdout, NULL, _IOLBF, 0);   /* before ANY output (DOCA logs to stdout): line-buffered through pipes/tee */
	struct st_gen_args a;
	memset(&a, 0, sizeof(a));
	a.n = 10000; a.passes = 11; a.nq = 4; a.instance = ST_INST_MATTIS; a.seed = 42; a.nsb = 96; a.ring = 2048; a.burst = 64;
	a.tx_mode = 2;
	uint32_t report_ms = 1000;
	const char *csv = NULL;
	volatile uint32_t *exit_host = NULL;
	struct st_gen_mirror *mir = NULL;   /* host view of the mirror */
	struct doca_gpu_eth_txq **txqs_dev = NULL;
	cudaStream_t s_io = NULL, s_synth = NULL;
	FILE *fcsv = NULL;
	double rate_gbps = 0.0;
	int opt;
	while ((opt = getopt(argc, argv, "g:n:m:N:K:q:i:s:S:R:b:z:L:W:r:o:h")) != -1) {
		switch (opt) {
		case 'g': snprintf(gpu, sizeof(gpu), "%s", optarg); break;
		case 'n': snprintf(nic, sizeof(nic), "%s", optarg); break;
		case 'm': if (st_parse_mac(optarg, dst_mac)) { usage(); return 1; } break;
		case 'N': a.n = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'K': a.passes = (uint32_t)strtoul(optarg, NULL, 10) + 1u; break;
		case 'q': a.nq = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'i': a.instance = strcmp(optarg, "random") == 0 ? ST_INST_RANDOM : ST_INST_MATTIS; break;
		case 's': a.seed = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'S': a.nsb = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'R': a.ring = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'b': a.burst = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'z': a.sleep_ns = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'L': rate_gbps = strtod(optarg, NULL); break;
		case 'W': a.tx_mode = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'r': report_ms = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'o': csv = optarg; break;
		default: usage(); return 1;
		}
	}
	if (a.tx_mode == 2)
		a.burst = ST_GEN_BURST;   /* one frame per thread of the sender block */
	if (a.nq < 1 || a.nq > ST_MAX_Q || a.tx_mode > 2 || a.ring < 2 * a.burst || a.burst > 1024) { usage(); return 1; }
	a.nblk = st_nblk(a.n);
	a.wtot = st_wtot(a.n);
	if (rate_gbps > 0.0) {
		const double ns = 8.0 * ST_PAYLOAD * a.nq / rate_gbps;   /* ns per frame on one queue (Gb/s == bit/ns) */
		a.pace_ns_x1024 = (uint32_t)(ns * 1024.0 + 0.5);
	}
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

	struct doca_log_backend *sdk_log;
	doca_log_backend_create_standard();
	if (doca_log_backend_create_with_file_sdk(stderr, &sdk_log) == DOCA_SUCCESS)
		doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	int cuda_id = 0;
	if (cudaDeviceGetByPCIBusId(&cuda_id, gpu) != cudaSuccess || cudaSetDevice(cuda_id) != cudaSuccess ||
	    cudaFree(0) != cudaSuccess) {
		fprintf(stderr, "[gen] bad GPU %s (use the PCIe address of GH200 GPU 1)\n", gpu);
		return 1;
	}
	int rc = 1;
	{   /* port MTU: ST_FRAME-byte frames need MTU 9000 on both ends (stream/README.md) */
		char ifn[64] = "?";
		const int mtu = st_nic_ensure_mtu(nic, (int)ST_FRAME, ifn, sizeof(ifn));
		if (mtu < (int)ST_FRAME) {
			fprintf(stderr, "[gen] NIC %s (%s) MTU %d < %u: run  sudo ip link set %s mtu 9000  and retry\n", nic, ifn, mtu,
				(unsigned)ST_FRAME, ifn);
			return 1;
		}
		printf("# NIC %s = %s, MTU %d\n", nic, ifn, mtu);
	}
	doca_error_t r = open_doca_device_with_pci(nic, NULL, &D.ddev);
	if (r != DOCA_SUCCESS) { DOCA_LOG_ERR("open NIC %s: %s", nic, doca_error_get_descr(r)); return 1; }
	if ((r = doca_gpu_create(gpu, &D.gpu)) != DOCA_SUCCESS) { DOCA_LOG_ERR("doca_gpu_create: %s", doca_error_get_descr(r)); goto out; }
	const size_t tx_bytes = (size_t)a.nq * (a.ring + 1u) * ST_SLOT;
	if (init_flow() != DOCA_SUCCESS || create_ack_rxq(cuda_id) != DOCA_SUCCESS || create_txqs(a.nq, tx_bytes, a.tx_mode) != DOCA_SUCCESS)
		goto out;
	doca_devinfo_get_mac_addr(doca_dev_as_devinfo(D.ddev), D.my_mac, 6);

	/* header templates (one per queue: the queue byte) */
	uint8_t tmpl[ST_MAX_Q][ST_HDR_BYTES];
	for (uint32_t q = 0; q < a.nq; ++q)
		st_build_hdr((struct st_hdr *)tmpl[q], dst_mac, D.my_mac, ST_STREAM, (uint8_t)q, 0, 0, 0, 0, (uint16_t)a.nq,
			     (uint8_t)a.instance);
	a.txbuf = D.tx_buf;
	if (cudaMalloc((void **)&a.hdr_tmpl, sizeof(tmpl)) != cudaSuccess ||
	    cudaMemcpy(a.hdr_tmpl, tmpl, sizeof(tmpl), cudaMemcpyHostToDevice) != cudaSuccess ||
	    cudaMalloc((void **)&a.ready, sizeof(unsigned long long) * a.nq * a.ring) != cudaSuccess ||
	    cudaMemset(a.ready, 0, sizeof(unsigned long long) * a.nq * a.ring) != cudaSuccess ||
	    cudaMalloc((void **)&a.ticket, sizeof(unsigned long long) * a.nq) != cudaSuccess ||
	    cudaMemset(a.ticket, 0, sizeof(unsigned long long) * a.nq) != cudaSuccess ||
	    cudaMalloc((void **)&a.sent_ctr, sizeof(unsigned long long) * a.nq) != cudaSuccess ||
	    cudaMemset(a.sent_ctr, 0, sizeof(unsigned long long) * a.nq) != cudaSuccess ||
	    cudaMalloc((void **)&a.ack_pass, sizeof(unsigned long long)) != cudaSuccess ||
	    cudaMemset((void *)a.ack_pass, 0, sizeof(unsigned long long)) != cudaSuccess ||
	    cudaMalloc((void **)&a.ack_lost, sizeof(unsigned long long)) != cudaSuccess ||
	    cudaMemset(a.ack_lost, 0, sizeof(unsigned long long)) != cudaSuccess ||
	    cudaMalloc((void **)&a.qs, sizeof(struct st_gen_qstats) * a.nq) != cudaSuccess ||
	    cudaMemset(a.qs, 0, sizeof(struct st_gen_qstats) * a.nq) != cudaSuccess ||
	    cudaHostAlloc((void **)&mir, sizeof(*mir), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
	    cudaHostGetDevicePointer((void **)&a.mir, (void *)mir, 0) != cudaSuccess ||
	    cudaHostAlloc((void **)&a.exit_flag, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
	    cudaMalloc((void **)&txqs_dev, sizeof(void *) * a.nq) != cudaSuccess) {
		fprintf(stderr, "[gen] CUDA allocation failed\n");
		goto out;
	}
	memset(mir, 0, sizeof(*mir));
	exit_host = a.exit_flag;
	*exit_host = 0;
	cudaHostGetDevicePointer((void **)&a.exit_flag, (void *)exit_host, 0);
	{
		struct doca_gpu_eth_txq *h[ST_MAX_Q];
		for (uint32_t q = 0; q < a.nq; ++q) h[q] = D.tx[q].gpu;
		cudaMemcpy(txqs_dev, h, sizeof(void *) * a.nq, cudaMemcpyHostToDevice);
	}
	cudaMemset(D.tx_buf, 0, tx_bytes);
	cudaDeviceSynchronize();

	{   /* the roles wait on each other: all nq + 1 + nsb blocks must be resident at once */
		const int maxb = st_gen_max_blocks();
		if (maxb < (int)(a.nq + 1u + a.nsb)) {
			fprintf(stderr, "[gen] %u blocks of %u threads requested (-q %u, -S %u) but only %d fit at once: lower -S\n",
				a.nq + 1u + a.nsb, ST_GEN_THREADS, a.nq, a.nsb, maxb);
			goto out;
		}
		printf("# gen_kernel: %u blocks x %u threads (%d fit at once)\n", a.nq + 1u + a.nsb, ST_GEN_THREADS, maxb);
	}
	cudaStreamCreateWithFlags(&s_io, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s_synth, cudaStreamNonBlocking);
	if (st_gen_launch(s_io, s_synth, a, txqs_dev, D.rxq_gpu, D.mkey))
		goto out;
	const unsigned long long F = (unsigned long long)a.nblk * a.n;
	DOCA_LOG_INFO("generator up: n=%u (%u column blocks, %llu frames/pass, %.3f TB/pass) passes=%u queues=%u synth blocks=%u ring=%u burst=%u instance=%s seed=%u pace=%s%.0f Gb/s tx mode %u -> %02x:%02x:%02x:%02x:%02x:%02x Ethertype 0x%04x",
		      a.n, a.nblk, F, (double)F * ST_PAYLOAD / 1e12, a.passes, a.nq, a.nsb, a.ring, a.burst,
		      a.instance == ST_INST_MATTIS ? "mattis" : "random", a.seed, rate_gbps > 0 ? "" : "none ", rate_gbps, a.tx_mode,
		      dst_mac[0], dst_mac[1], dst_mac[2], dst_mac[3], dst_mac[4], dst_mac[5], ST_ETHERTYPE);
	fcsv = csv ? fopen(csv, "a") : NULL;
	if (fcsv) fprintf(fcsv, "t_s,pass_acked,frames_sent,payload_gbps,wait_ring_frac,wait_ack_frac,tx_errors\n");
	/* ring% = sender waiting for synthesis, tx% = sender waiting for the NIC to complete a burst,
	 * ack% = waiting for PASS_ACK (all as fractions of the queues' time) */
	printf("%8s | %6s %12s %12s %12s %10s | %8s %8s %8s %6s %8s %9s\n", "t(s)", "acked", "synth", "posted", "frames", "Gb/s",
	       "ring%", "tx%", "ack%", "txerr", "slotwait", "bursts");
	double t0 = st_now(), tp = t0;
	unsigned long long prev = 0, prev_ring = 0, prev_ack = 0, prev_tx = 0;
	volatile struct st_gen_mirror *M = mir;
	for (;;) {
		usleep(1000u * report_ms);
		unsigned long long sent = 0, posted = 0, synth = 0, swait = 0, err = 0, fin = 0, wr = 0, wa = 0, wt = 0, bursts = 0;
		unsigned long long sblk = 0, stk = 0, swords = 0, lasterr = 0;
		for (uint32_t q = 0; q < a.nq; ++q) {
			sent += M->qs[q].sent; posted += M->qs[q].posted; err += M->qs[q].tx_errors; fin += M->qs[q].finished;
			synth += M->qs[q].synth; swait += M->qs[q].synth_wait;
			sblk += M->qs[q].synth_blocks; stk += M->qs[q].synth_ticket; swords += M->qs[q].synth_words;
			wr += M->qs[q].wait_ring_ns; wa += M->qs[q].wait_ack_ns; wt += M->qs[q].wait_tx_ns; bursts += M->qs[q].bursts;
			if (M->qs[q].last_error) lasterr = M->qs[q].last_error;
		}
		const unsigned long long ack_pass = M->ack_pass, ack_lost = M->ack_lost;
		const double t = st_now();
		const double gbps = (double)(sent - prev) * ST_PAYLOAD * 8.0 / (t - tp) / 1e9;
		const double ring_pct = 100.0 * (double)(wr - prev_ring) / ((t - tp) * 1e9 * a.nq);
		const double tx_pct = 100.0 * (double)(wt - prev_tx) / ((t - tp) * 1e9 * a.nq);
		const double ack_pct = 100.0 * (double)(wa - prev_ack) / ((t - tp) * 1e9 * a.nq);
		printf("%8.1f | %6llu %12llu %12llu %12llu %10.1f | %7.1f%% %7.1f%% %7.1f%% %6llu %8llu %9llu\n", t - t0, ack_pass, synth,
		       posted, sent, gbps, ring_pct, tx_pct, ack_pct, err, swait, bursts);
		printf("         synth probes: blocks entered %llu, tickets %llu, payloads written %llu  (mirror tick %llu)\n", sblk, stk,
		       swords, M->ticks);
		(void)s_synth;
		if (fcsv) fprintf(fcsv, "%.3f,%llu,%llu,%.3f,%.4f,%.4f,%llu\n", t - t0, ack_pass, sent, gbps, ring_pct / 100.0,
				  ack_pct / 100.0, err);
		fflush(stdout);
		prev = sent; prev_ring = wr; prev_ack = wa; prev_tx = wt; tp = t;
		if (err) {
			fprintf(stderr, "[gen] send error on a queue: %s\n", doca_error_get_descr((doca_error_t)lasterr));
			break;
		}
		if (ack_lost == ~0ull) {
			fprintf(stderr, "[gen] the ACK receive call failed on the GPU\n");
			break;
		}
		if (fin == a.nq) {
			printf("# all %u passes acknowledged; %llu data frames sent; last ACK reported %llu lost\n", a.passes, sent,
			       ack_lost);
			rc = 0;
			break;
		}
		if (force_quit) break;
		if (cudaStreamQuery(s_io) != cudaErrorNotReady) {
			fprintf(stderr, "[gen] kernel ended unexpectedly: %s\n", cudaGetErrorString(cudaGetLastError()));
			break;
		}
	}
	if (fcsv) fclose(fcsv);
out:
	if (exit_host)
		__atomic_store_n((uint32_t *)exit_host, 1u, __ATOMIC_SEQ_CST);
	cudaDeviceSynchronize();
	destroy_all(a.nq);
	return rc;
}
