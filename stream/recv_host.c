/*
 * recv_host.c -- host side of the GPUNetIO streaming receiver (PU2).
 *
 *   sudo ./bin/x86_64/st_recv -N 10000000 -B 8 -K 10 -i mattis -s 42 -o results/stream/run.csv -c run.chk
 *
 * DOCA set-up is server_doca/doca_host.c line by line (the TVG receiver's):
 * one cyclic GPU receive queue, a DOCA Flow root pipe steering Ethertype
 * ST_ETHERTYPE to it, one GPU send queue with completions on the GPU for the
 * PASS_ACK, dmabuf registered buffers.  The CPU never touches a frame; between
 * passes it launches the update kernel and posts one 64-byte acknowledgement.
 */
#include <getopt.h>
#include <endian.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <doca_log.h>

#include "common.h"          /* open_doca_device_with_pci (DOCA samples) */
#include "recv.h"
#include "st_net.h"

DOCA_LOG_REGISTER(ST::RECV);

#define ALIGN_SIZE(size, align) (size = (((size) + (align) - 1) / (align)) * (align))
#define FLOW_NB_COUNTERS 524288

static volatile sig_atomic_t force_quit;
static void on_signal(int s)
{
	(void)s;
	if (force_quit)   /* second Ctrl+C: the host is stuck in a CUDA call that never returns */
		_exit(130);
	force_quit = 1;
}
static size_t pg(void) { long r = sysconf(_SC_PAGESIZE); return r == -1 ? 4096 : (size_t)r; }

struct cfg {
	char gpu[32], nic[32];
	int cuda_id;
	uint32_t n, B, K, workers, seed, report_ms, tx_mode, instance;
	float xi, dt;
	uint8_t gen_mac[6];
	const char *csv, *chkfile;
};

static struct {
	struct doca_dev *ddev;
	struct doca_gpu *gpu;
	struct doca_flow_port *port;
	/* RX (one queue, as the request service) */
	struct doca_ctx *rxq_ctx;
	struct doca_eth_rxq *rxq_cpu;
	struct doca_gpu_eth_rxq *rxq_gpu;
	struct doca_mmap *rx_mmap;
	uint8_t *rx_buf;
	int rx_fd;
	struct doca_flow_pipe *rx_pipe;
	struct doca_flow_pipe_entry *rx_entry;
	/* TX (PASS_ACK) */
	struct doca_ctx *txq_ctx;
	struct doca_eth_txq *txq;
	struct doca_gpu_eth_txq *txq_gpu;
	struct doca_mmap *tx_mmap;
	uint8_t *tx_buf;
	int tx_fd;
	uint32_t tx_mkey;
	uint8_t my_mac[6];
} D;

#define CK(x, msg)                                                                                 \
	do {                                                                                       \
		doca_error_t r_ = (x);                                                             \
		if (r_ != DOCA_SUCCESS) {                                                          \
			DOCA_LOG_ERR("%s: %s", msg, doca_error_get_descr(r_));                     \
			return r_;                                                                 \
		}                                                                                  \
	} while (0)

/* ---------------------------------------------------------------- DOCA Flow (doca_host.c) */
static doca_error_t init_doca_flow(void)
{
	struct doca_flow_cfg *flow_cfg;
	struct doca_flow_port_cfg *port_cfg;
	CK(doca_flow_cfg_create(&flow_cfg), "flow cfg");
	CK(doca_flow_cfg_set_pipe_queues(flow_cfg, 1), "flow queues");
	CK(doca_flow_cfg_set_mode_args(flow_cfg, "vnf,hws=off"), "flow mode");
	CK(doca_flow_cfg_set_nr_counters(flow_cfg, FLOW_NB_COUNTERS), "flow counters");
	CK(doca_flow_init(flow_cfg), "flow init");
	doca_flow_cfg_destroy(flow_cfg);
	CK(doca_flow_port_cfg_create(&port_cfg), "port cfg");
	CK(doca_flow_port_cfg_set_port_id(port_cfg, 0), "port id");
	CK(doca_flow_port_cfg_set_dev(port_cfg, D.ddev), "port dev");
	CK(doca_flow_port_start(port_cfg, &D.port), "port start");
	doca_flow_port_cfg_destroy(port_cfg);
	return DOCA_SUCCESS;
}

/* Root pipe: Ethertype ST_ETHERTYPE -> RSS to our single GPU RX queue (doca_host.c create_rx_pipe). */
static doca_error_t create_rx_pipe(void)
{
	struct doca_flow_match match = {0};
	struct doca_flow_match match_mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_pipe_cfg *pipe_cfg;
	uint16_t rss_queues[1];
	struct doca_flow_monitor monitor = {.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED};

	doca_eth_rxq_apply_queue_id(D.rxq_cpu, 0);
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array = rss_queues;
	fwd.rss.outer_flags = 0;
	fwd.rss.nr_queues = 1;
	match_mask.outer.eth.type = 0xFFFF;

	CK(doca_flow_pipe_cfg_create(&pipe_cfg, D.port), "pipe cfg");
	CK(doca_flow_pipe_cfg_set_name(pipe_cfg, "ST_RX_ROOT_PIPE"), "pipe name");
	CK(doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC), "pipe type");
	CK(doca_flow_pipe_cfg_set_is_root(pipe_cfg, true), "pipe root");
	CK(doca_flow_pipe_cfg_set_match(pipe_cfg, NULL, &match_mask), "pipe match");
	CK(doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor), "pipe monitor");
	CK(doca_flow_pipe_create(pipe_cfg, &fwd, NULL, &D.rx_pipe), "pipe create");
	doca_flow_pipe_cfg_destroy(pipe_cfg);

	match.outer.eth.type = htons(ST_ETHERTYPE);
	CK(doca_flow_pipe_basic_add_entry(0, D.rx_pipe, &match, 0, NULL, NULL, NULL, DOCA_FLOW_ENTRY_FLAGS_NO_WAIT, NULL,
					  &D.rx_entry), "pipe entry");
	return doca_flow_entries_process(D.port, 0, 0, 0);
}

static doca_error_t register_gpu_buffer(size_t bytes, uint8_t **addr, int *dmabuf_fd, struct doca_mmap **mmap)
{
	ALIGN_SIZE(bytes, pg());
	CK(doca_mmap_create(mmap), "mmap create");
	CK(doca_mmap_add_dev(*mmap, D.ddev), "mmap add dev");
	CK(doca_gpu_mem_alloc(D.gpu, bytes, pg(), DOCA_GPU_MEM_TYPE_GPU, (void **)addr, NULL), "gpu mem alloc");
	if (doca_gpu_dmabuf_fd(D.gpu, *addr, bytes, dmabuf_fd) != DOCA_SUCCESS) {
		*dmabuf_fd = -1;
		CK(doca_mmap_set_memrange(*mmap, *addr, bytes), "mmap memrange");
	} else {
		CK(doca_mmap_set_dmabuf_memrange(*mmap, *dmabuf_fd, *addr, 0, bytes), "mmap dmabuf memrange");
	}
	CK(doca_mmap_set_permissions(*mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE), "mmap permissions");
	CK(doca_mmap_start(*mmap), "mmap start");
	return DOCA_SUCCESS;
}

static doca_error_t create_rxq(struct cfg *c)
{
	uint32_t cyclic_bytes = 0;
	struct cudaDeviceProp prop;
	union doca_data user_data;
	CK(doca_eth_rxq_create(D.ddev, ST_RX_PKT_NUM, ST_SLOT, &D.rxq_cpu), "rxq create");
	CK(doca_eth_rxq_set_type(D.rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC), "rxq type");
	CK(doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, ST_SLOT, ST_RX_PKT_NUM, 0, 0, 0,
						 &cyclic_bytes), "rxq estimate");
	ALIGN_SIZE(cyclic_bytes, pg());
	CK(register_gpu_buffer(cyclic_bytes, &D.rx_buf, &D.rx_fd, &D.rx_mmap), "rx buffer");
	CK(doca_eth_rxq_set_pkt_buf(D.rxq_cpu, D.rx_mmap, 0, cyclic_bytes), "rxq pkt buf");
	cudaGetDeviceProperties(&prop, c->cuda_id);
	if (prop.major < 9)
		CK(doca_eth_rxq_gpu_enable_mcst_qp(D.rxq_cpu), "mcst qp");
	D.rxq_ctx = doca_eth_rxq_as_doca_ctx(D.rxq_cpu);
	if (D.rxq_ctx == NULL)
		return DOCA_ERROR_BAD_STATE;
	CK(doca_ctx_set_datapath_on_gpu(D.rxq_ctx, D.gpu), "rxq datapath");
	user_data.ptr = &D;
	CK(doca_ctx_set_user_data(D.rxq_ctx, user_data), "rxq user data");
	CK(doca_ctx_start(D.rxq_ctx), "rxq start");
	CK(doca_eth_rxq_get_gpu_handle(D.rxq_cpu, &D.rxq_gpu), "rxq gpu handle");
	DOCA_LOG_INFO("RX queue: %u slots x %u B, %.1f MB cyclic buffer in GPU memory", ST_RX_PKT_NUM, ST_SLOT,
		      (double)cyclic_bytes / 1e6);
	return create_rx_pipe();
}

static doca_error_t create_txq(struct cfg *c)
{
	CK(doca_eth_txq_create(D.ddev, ST_TX_SQ_DEPTH, &D.txq), "txq create");
	CK(doca_eth_txq_set_l3_chksum_offload(D.txq, 0), "txq l3");
	CK(doca_eth_txq_set_l4_chksum_offload(D.txq, 0), "txq l4");
	CK(doca_eth_txq_gpu_set_completion_on_gpu(D.txq), "txq completion on gpu");
	D.txq_ctx = doca_eth_txq_as_doca_ctx(D.txq);
	if (!D.txq_ctx)
		return DOCA_ERROR_BAD_STATE;
	CK(doca_ctx_set_datapath_on_gpu(D.txq_ctx, D.gpu), "txq datapath");
	if (c->tx_mode == 1) {
		enum doca_eth_wait_on_time_type mode;
		CK(doca_eth_txq_cap_get_wait_on_time_offload_supported(doca_dev_as_devinfo(D.ddev), &mode), "wait cap");
		if (mode != DOCA_ETH_WAIT_ON_TIME_TYPE_NATIVE)
			return DOCA_ERROR_NOT_SUPPORTED;
		CK(doca_eth_txq_set_wait_on_time_offload(D.txq), "txq wait-on-time");
	}
	CK(doca_ctx_start(D.txq_ctx), "txq start");
	CK(doca_eth_txq_apply_queue_id(D.txq, 0), "txq queue id");
	CK(doca_eth_txq_get_gpu_handle(D.txq, &D.txq_gpu), "txq gpu handle");
	CK(register_gpu_buffer(ST_SLOT, &D.tx_buf, &D.tx_fd, &D.tx_mmap), "tx buffer");
	CK(doca_mmap_get_mkey(D.tx_mmap, D.ddev, &D.tx_mkey), "mkey");
	D.tx_mkey = htobe32(D.tx_mkey);
	return DOCA_SUCCESS;
}

static void destroy_all(void)
{
	if (D.rx_pipe) doca_flow_pipe_destroy(D.rx_pipe);
	if (D.port) doca_flow_port_stop(D.port);
	doca_flow_destroy();
	if (D.rxq_ctx) doca_ctx_stop(D.rxq_ctx);
	if (D.rxq_cpu) doca_eth_rxq_destroy(D.rxq_cpu);
	if (D.rx_mmap) doca_mmap_destroy(D.rx_mmap);
	if (D.rx_buf) doca_gpu_mem_free(D.gpu, D.rx_buf);
	if (D.txq_ctx) doca_ctx_stop(D.txq_ctx);
	if (D.txq) doca_eth_txq_destroy(D.txq);
	if (D.tx_mmap) doca_mmap_destroy(D.tx_mmap);
	if (D.tx_buf) doca_gpu_mem_free(D.gpu, D.tx_buf);
	if (D.gpu) doca_gpu_destroy(D.gpu);
	if (D.ddev) doca_dev_close(D.ddev);
}

static void *dev_zero(size_t bytes)
{
	void *p = NULL;
	if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
	if (cudaMemset(p, 0, bytes) != cudaSuccess) return NULL;
	return p;
}

extern int st_state_alloc(uint32_t n, uint32_t B, void **state_out, uint32_t **signs, int32_t **acc, long long **e2,
			  unsigned long long **chk);
extern int st_state_init(void *state, uint32_t seed, cudaStream_t s);
extern void st_state_xy(void *state, float **x, float **y);

/* Waits until *ctr >= target (host-mapped counter written by the receive warp); prints a heartbeat every second.
 * Returns 0, or -1 on force_quit / receive-warp error. */
static int wait_counter(volatile unsigned long long *ctr, unsigned long long target, struct st_stats *st, const char *what)
{
	double t0 = st_now(), t_rep = t0;
	while (*ctr < target) {
		if (force_quit || st->rx_err || st->tx_err)
			return -1;
		const double now = st_now();
		if (now - t_rep >= 1.0) {
			printf("# waiting for %s: %llu/%llu after %.0f s\n", what, *ctr, target, now - t0);
			t_rep = now;
		}
		usleep(2000);   /* the counters are refreshed by the receive warp every ~0.5 ms */
	}
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
		"st_recv: GPUNetIO streaming receiver (one Ethertype-steered GPU queue, as the request service)\n"
		"  -g GPU PCIe (0000:ab:00.0)  -n NIC PCIe (0000:bd:00.0)  -m generator MAC (08:20:e7:47:10:8e)\n"
		"  -N n  -B replicas (<=8)  -K steps (updates; one extra energy pass follows)\n"
		"  -w worker blocks (default: all that fit minus 4)\n"
		"  -i random|mattis  -s seed  -x xi (default 0.5/sqrt n)  -d dt (default 1.1, as bit_dense_multi)\n"
		"  -W tx mode for the ACK (0 txq_send, 1 wait_send)  -o CSV output  -c per-step checksum file  -r report ms\n");
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);   /* before ANY output (DOCA logs to stdout): line-buffered through pipes/tee */
	struct cfg c;
	memset(&c, 0, sizeof(c));
	snprintf(c.gpu, sizeof(c.gpu), "0000:ab:00.0");
	snprintf(c.nic, sizeof(c.nic), "0000:bd:00.0");
	{ uint8_t m[6] = {0x08, 0x20, 0xe7, 0x47, 0x10, 0x8e}; memcpy(c.gen_mac, m, 6); }   /* PU3 enP2s2np0 */
	c.n = 10000; c.B = 8; c.K = 10; c.seed = 42; c.report_ms = 1000;
	c.instance = ST_INST_MATTIS; c.dt = ST_DEFAULT_DT; c.xi = 0.f;
	int opt;
	while ((opt = getopt(argc, argv, "g:n:m:N:B:K:w:i:s:x:d:W:o:c:r:h")) != -1) {
		switch (opt) {
		case 'g': snprintf(c.gpu, sizeof(c.gpu), "%s", optarg); break;
		case 'n': snprintf(c.nic, sizeof(c.nic), "%s", optarg); break;
		case 'm': if (st_parse_mac(optarg, c.gen_mac)) { usage(); return 1; } break;
		case 'N': c.n = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'B': c.B = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'K': c.K = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'w': c.workers = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'i': c.instance = strcmp(optarg, "random") == 0 ? ST_INST_RANDOM : ST_INST_MATTIS; break;
		case 's': c.seed = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'x': c.xi = strtof(optarg, NULL); break;
		case 'd': c.dt = strtof(optarg, NULL); break;
		case 'W': c.tx_mode = (uint32_t)strtoul(optarg, NULL, 10); break;
		case 'o': c.csv = optarg; break;
		case 'c': c.chkfile = optarg; break;
		case 'r': c.report_ms = (uint32_t)strtoul(optarg, NULL, 10); break;
		default: usage(); return 1;
		}
	}
	if (c.B < 1 || c.B > ST_MAX_B || c.K < 1) { usage(); return 1; }
	if (c.xi <= 0.f) c.xi = st_default_xi(c.n);
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);

	struct doca_log_backend *sdk_log;
	doca_log_backend_create_standard();
	if (doca_log_backend_create_with_file_sdk(stderr, &sdk_log) == DOCA_SUCCESS)
		doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);

	if (cudaDeviceGetByPCIBusId(&c.cuda_id, c.gpu) != cudaSuccess || cudaSetDevice(c.cuda_id) != cudaSuccess ||
	    cudaFree(0) != cudaSuccess) {
		fprintf(stderr, "[recv] bad GPU %s\n", c.gpu);
		return 1;
	}
	int rc = 1;
	void *S = NULL;
	uint32_t *signs = NULL; int32_t *acc = NULL; long long *e2 = NULL; unsigned long long *chk = NULL;
	struct st_stats *st = NULL, *st_d = NULL;     /* host-mapped: written by the receive warp, read here (no page migration) */
	struct st_dev *dv = NULL;                     /* device: worker counters */
	struct st_ctl *ctl = NULL, *ctl_dev = NULL;   /* host-mapped: the only host -> GPU channel while the kernels run */
	struct st_rx_args a;
	cudaStream_t s_rx = NULL, s_work = NULL, s_upd = NULL;
	FILE *csv = NULL, *chkf = NULL;
	memset(&a, 0, sizeof(a));

	{   /* port MTU: ST_FRAME-byte frames need MTU 9000 on both ends (stream/README.md) */
		char ifn[64] = "?";
		const int mtu = st_nic_ensure_mtu(c.nic, (int)ST_FRAME, ifn, sizeof(ifn));
		if (mtu < (int)ST_FRAME) {
			fprintf(stderr, "[recv] NIC %s (%s) MTU %d < %u: run  sudo ip link set %s mtu 9000  and retry\n", c.nic, ifn, mtu,
				(unsigned)ST_FRAME, ifn);
			return 1;
		}
		printf("# NIC %s = %s, MTU %d\n", c.nic, ifn, mtu);
	}
	doca_error_t r = open_doca_device_with_pci(c.nic, NULL, &D.ddev);
	if (r != DOCA_SUCCESS) { DOCA_LOG_ERR("open NIC %s: %s", c.nic, doca_error_get_descr(r)); return 1; }
	if ((r = doca_gpu_create(c.gpu, &D.gpu)) != DOCA_SUCCESS) { DOCA_LOG_ERR("doca_gpu_create: %s", doca_error_get_descr(r)); goto out; }
	if (init_doca_flow() != DOCA_SUCCESS || create_rxq(&c) != DOCA_SUCCESS || create_txq(&c) != DOCA_SUCCESS)
		goto out;
	doca_devinfo_get_mac_addr(doca_dev_as_devinfo(D.ddev), D.my_mac, 6);
	printf("# my MAC %02x:%02x:%02x:%02x:%02x:%02x (the generator must send to this address)\n", D.my_mac[0], D.my_mac[1],
	       D.my_mac[2], D.my_mac[3], D.my_mac[4], D.my_mac[5]);

	/* ---- state, rings, counters ---- */
	if (st_state_alloc(c.n, c.B, &S, &signs, &acc, &e2, &chk)) { fprintf(stderr, "[recv] state alloc failed\n"); goto out; }
	if (cudaHostAlloc((void **)&st, sizeof(*st), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
	    cudaHostGetDevicePointer((void **)&st_d, (void *)st, 0) != cudaSuccess ||
	    cudaHostAlloc((void **)&ctl, sizeof(*ctl), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
	    cudaHostGetDevicePointer((void **)&ctl_dev, (void *)ctl, 0) != cudaSuccess)
		goto out;
	memset(st, 0, sizeof(*st));
	memset(ctl, 0, sizeof(*ctl));
	a.items = dev_zero(sizeof(struct st_work_item) * ST_WORK_CAP);
	a.work_claim = dev_zero(sizeof(unsigned long long));
	dv = (struct st_dev *)dev_zero(sizeof(struct st_dev));
	if (!a.items || !a.work_claim || !dv) goto out;
	a.n = c.n; a.B = c.B; a.wtot = st_wtot(c.n); a.nblk = st_nblk(c.n);
	a.signs = signs; a.acc = acc; a.e2 = e2; a.chk = chk; a.st = st_d; a.dv = dv; a.ctl = ctl_dev;
	a.K = c.K; a.xi = c.xi; a.dt = c.dt;
	st_state_xy(S, &a.x, &a.y);
	a.txq = D.txq_gpu; a.tx_frame = D.tx_buf; a.tx_mkey = D.tx_mkey; a.tx_mode = c.tx_mode;
	{   /* PASS_ACK template into the registered send buffer, BEFORE the persistent kernels start
	     * (the receive warp fills pass/lost/frames and posts it; the host never touches the GPU again) */
		uint8_t ack[ST_HDR_BYTES];
		st_build_hdr((struct st_hdr *)ack, c.gen_mac, D.my_mac, ST_PASS_ACK, 0, 0, 0, 0, (uint16_t)a.nblk, 0, (uint8_t)c.instance);
		if (cudaMemcpy(D.tx_buf, ack, ST_HDR_BYTES, cudaMemcpyHostToDevice) != cudaSuccess) goto out;
	}
	{
		int maxb = st_worker_max_blocks(c.B * ST_PAYLOAD);
		if (maxb < 6) { fprintf(stderr, "[recv] only %d resident worker blocks with %u B smem\n", maxb, c.B * ST_PAYLOAD); goto out; }
		a.workers = (uint32_t)maxb - 4u;   /* leave slots free, as the request service does */
		if (c.workers && c.workers < a.workers) a.workers = c.workers;
	}
	cudaStreamCreateWithFlags(&s_rx, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s_work, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&s_upd, cudaStreamNonBlocking);
	if (st_state_init(S, c.seed, s_upd)) goto out;
	cudaStreamSynchronize(s_upd);
	const unsigned long long chk0 = *chk;
	if (st_rx_launch(s_rx, s_work, a, D.rxq_gpu)) goto out;
	DOCA_LOG_INFO("receiver up: n=%u B=%u K=%u workers=%u instance=%s xi=%.8g dt=%g; Ethertype 0x%04x; waiting for pass 0",
		      c.n, c.B, c.K, a.workers, c.instance == ST_INST_MATTIS ? "mattis" : "random", c.xi, c.dt, ST_ETHERTYPE);
	printf("# n %u B %u K %u workers %u instance %s seed %u xi %.9g dt %g frames/pass %llu payload/pass %.3f TB\n", c.n, c.B,
	       c.K, a.workers, c.instance == ST_INST_MATTIS ? "mattis" : "random", c.seed, c.xi, c.dt,
	       (unsigned long long)a.nblk * c.n, (double)a.nblk * c.n * ST_PAYLOAD / 1e12);
	printf("# init chk 0x%016llx  (Mattis ground state 2E = %lld)\n", chk0, -(long long)c.n * ((long long)c.n - 1));
	csv = c.csv ? fopen(c.csv, "a") : NULL;
	if (csv) fprintf(csv, "pass,update,t_pass_s,payload_gbps,frames,expected,lost,bad,dropped_bursts,recv_frac,best_2E,chk\n");
	chkf = c.chkfile ? fopen(c.chkfile, "w") : NULL;
	if (chkf) fprintf(chkf, "init 0x%016llx\n", chk0);

	/* ---- passes ---- */
	double t_pass0 = st_now(), t_rep = t_pass0;
	unsigned long long rep_frames = 0, prev_frames = 0, prev_expect = 0, prev_bad = 0, prev_drop = 0;
	unsigned long long prev_recv = 0, prev_pub = 0;
	unsigned long long prev_rxf = 0, prev_g1 = 0, prev_g5 = 0, prev_full = 0;
	int bad_run = 0;
	for (uint32_t pass = 0; pass <= c.K && !force_quit; ++pass) {
		const int do_update = pass < c.K;
		*chk = 0ull;
		for (uint32_t b = 0; b < c.B; ++b) e2[b] = 0;
		/* wait for the PASS_END of every generator queue */
		for (;;) {
			const unsigned long long gq = st->gen_queues;
			if (gq && st->ends >= gq * (pass + 1ull)) break;
			if (st->rx_err) { fprintf(stderr, "[recv] receive error: the receive warp exited\n"); force_quit = 1; }
			if (force_quit) break;
			const double now = st_now();
			if (now - t_rep >= c.report_ms * 1e-3) {
				const unsigned long long fr = st->frames;
				printf("  t %8.1f s  pass %u  %.1f Gb/s payload  frames %llu  rx_frames %llu  bad %llu  dropped bursts %llu  (rx polls %llu, workers up %llu/%u)\n",
				       now - t_pass0, pass, (double)(fr - rep_frames) * ST_PAYLOAD * 8.0 / (now - t_rep) / 1e9, fr,
				       st->rx_frames, st->bad, st->rx_dropped, st->rx_polls, st->worker_blocks, a.workers);
				fflush(stdout);
				rep_frames = fr; t_rep = now;
			}
			usleep(2000);   /* host-mapped stats, refreshed by the receive warp every ~0.5 ms */
		}
		if (force_quit) break;
		const double t_end = st_now();
		/* drain: every burst the receive warp published has been consumed and acknowledged
		 * by a worker (exact: bursts_done == rx_bursts, re-checked after a short pause) */
		for (;;) {
			const unsigned long long pub = st->rx_bursts;
			if (wait_counter(&st->bursts_done, pub, st, "ring drain")) goto out;
			usleep(2000);
			if (st->rx_bursts == pub) break;
		}
		/* update: done by the idle worker blocks on request (no kernel launch from the host:
		 * behind the W resident blocks it would never be scheduled) */
		__sync_synchronize();
		ctl->upd_req = pass + 1u;
		__sync_synchronize();
		if (wait_counter(&st->upd_done, (unsigned long long)a.workers * (pass + 1ull), st, "update by the worker blocks"))
			goto out;
		const unsigned long long fr = st->frames - prev_frames, expect = st->end_frames - prev_expect;
		const unsigned long long bad = st->bad - prev_bad, drop = st->rx_dropped - prev_drop;
		const unsigned long long trecv = st->t_recv_ns - prev_recv, tpub = st->t_pub_ns - prev_pub;
		prev_frames = st->frames; prev_expect = st->end_frames; prev_bad = st->bad; prev_drop = st->rx_dropped;
		prev_recv = st->t_recv_ns; prev_pub = st->t_pub_ns;
		const long long lost = (long long)expect - (long long)fr;   /* > 0 lost, < 0 duplicated/overwritten */
		/* where frames went missing: rxw = frames the receive warp got from the queue this pass (PASS_ENDs
		 * included, so rxw == expect + queues means the queue delivered every frame and any loss is behind
		 * the warp); gap = the warp's longest absence from the queue; backlog = most frames waiting to be
		 * read in place (> ST_RX_PKT_NUM: the NIC can overwrite them first) */
		const unsigned long long rxw = st->rx_frames - prev_rxf, g1 = st->gaps_1ms - prev_g1, g5 = st->gaps_5ms - prev_g5;
		const unsigned long long fullc = st->full_calls - prev_full;
		const double gap_ms = (double)st->gap_max_ns * 1e-6;
		const unsigned long long call_max = st->call_max, backlog = st->backlog_max, gq_now = st->gen_queues;
		prev_rxf = st->rx_frames; prev_g1 = st->gaps_1ms; prev_g5 = st->gaps_5ms; prev_full = st->full_calls;
		long long best = e2[0];
		for (uint32_t b = 1; b < c.B; ++b) if (e2[b] < best) best = e2[b];
		const double tp = t_end - t_pass0, gbps = (double)fr * ST_PAYLOAD * 8.0 / tp / 1e9;
		printf("pass %3u %s  t %9.3f s  %7.1f Gb/s  frames %llu/%llu lost %lld bad %llu drop %llu  best 2E %lld  chk 0x%016llx  recv-wait %.1f%%",
		       pass, do_update ? "upd" : "fin", tp, gbps, fr, expect, lost, bad, drop, best, *chk,
		       trecv + tpub ? 100.0 * (double)trecv / (double)(trecv + tpub) : 0.0);
		for (uint32_t b = 0; b < c.B; ++b) printf("%s%lld", b ? "," : "  2E[", e2[b]);
		printf("]\n");
		printf("         rxw %llu (expect+queues %llu, diff %lld)  gap max %.3f ms  gaps>1ms %llu >5ms %llu  call max %llu full %llu  backlog max %llu/%u\n",
		       rxw, expect + gq_now, (long long)(expect + gq_now) - (long long)rxw, gap_ms, g1, g5, call_max, fullc, backlog,
		       ST_RX_PKT_NUM);
		if (csv) fprintf(csv, "%u,%d,%.6f,%.3f,%llu,%llu,%lld,%llu,%llu,%.4f,%lld,0x%016llx\n", pass, do_update, tp, gbps, fr,
				 expect, lost, bad, drop, trecv + tpub ? (double)trecv / (double)(trecv + tpub) : 0.0, best, *chk);
		if (chkf) fprintf(chkf, "pass %u 2E %lld chk 0x%016llx\n", pass, best, *chk);
		if (lost != 0 || bad || drop) bad_run = 1;
		/* acknowledge (the receive warp posts the frame): next pass may start */
		ctl->ack_lost = (uint32_t)(lost > 0 ? lost : 0);
		ctl->ack_frames = (uint32_t)fr;
		__sync_synchronize();
		ctl->ack_req = pass + 1u;
		__sync_synchronize();
		if (wait_counter(&st->ack_done, pass + 1ull, st, "PASS_ACK send")) {
			if (st->tx_err)
				fprintf(stderr, "[recv] PASS_ACK send failed: %s; try -W %u\n",
					doca_error_get_descr((doca_error_t)st->tx_err), c.tx_mode ? 0u : 1u);
			goto out;
		}
		t_pass0 = st_now();
	}
	printf("# %s\n", bad_run ? "RUN INVALID: frames lost/duplicated/bad or bursts dropped in at least one pass"
				: "RUN OK: every pass complete, no frame lost");
	rc = bad_run ? 2 : 0;
out:
	if (csv) fclose(csv);
	if (chkf) fclose(chkf);
	if (ctl) __atomic_store_n((uint32_t *)&ctl->exit_flag, 1u, __ATOMIC_SEQ_CST);
	{   /* bounded wait for the kernels; a stuck receive warp must not hang the exit */
		const double t_exit = st_now();
		while ((cudaStreamQuery(s_rx) == cudaErrorNotReady || cudaStreamQuery(s_work) == cudaErrorNotReady) &&
		       st_now() - t_exit < 3.0)
			usleep(10000);
		if (cudaStreamQuery(s_rx) == cudaErrorNotReady || cudaStreamQuery(s_work) == cudaErrorNotReady) {
			fprintf(stderr, "[recv] kernels did not exit within 3 s; exiting hard\n");
			fflush(stdout);
			_exit(rc);
		}
	}
	cudaDeviceSynchronize();
	destroy_all();
	return rc;
}
