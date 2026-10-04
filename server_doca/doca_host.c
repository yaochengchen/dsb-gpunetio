/*
 * doca_host.c -- start-up and monitoring of the DOCA GPUNetIO dSB server.
 *
 * The CPU only builds the queues, launches the persistent kernel and prints
 * counters; it never touches a request.  The RX set-up (DOCA Flow root pipe
 * steering our Ethertype to one cyclic GPU RX queue) and the TX set-up (GPU
 * send queue with completions on the GPU, dmabuf-registered GPU buffer)
 * follow receiver/host/recv_host.c and generator/host/tvg_host.c of the TVG
 * suite line by line where possible.
 */
#include <arpa/inet.h>
#include <stdio.h>
#include <endian.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <doca_flow.h>
#include <doca_log.h>

#include "common.h"
#include "doca_server.h"

DOCA_LOG_REGISTER(DSB::HOST);

#define FLOW_NB_COUNTERS 524288
#define ALIGN_SIZE(size, align) (size = (((size) + (align) - 1) / (align)) * (align))

volatile sig_atomic_t force_quit;
static struct doca_flow_port *df_port;

static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, stopping the server...", signum);
		force_quit = 1;
	}
}

static size_t host_page_size(void)
{
	long ret = sysconf(_SC_PAGESIZE);
	return ret == -1 ? 4096 : (size_t)ret;
}

/* ---------------------------------------------------------------- DOCA Flow */

static doca_error_t init_doca_flow(struct doca_dev *dev)
{
	struct doca_flow_cfg *flow_cfg;
	struct doca_flow_port_cfg *port_cfg;
	doca_error_t result;

	result = doca_flow_cfg_create(&flow_cfg);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_flow_cfg_set_pipe_queues(flow_cfg, 1);
	if (result == DOCA_SUCCESS)
		result = doca_flow_cfg_set_mode_args(flow_cfg, "vnf,hws=off");
	if (result == DOCA_SUCCESS)
		result = doca_flow_cfg_set_nr_counters(flow_cfg, FLOW_NB_COUNTERS);
	if (result == DOCA_SUCCESS)
		result = doca_flow_init(flow_cfg);
	doca_flow_cfg_destroy(flow_cfg);
	if (result != DOCA_SUCCESS)
		return result;

	result = doca_flow_port_cfg_create(&port_cfg);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_flow_port_cfg_set_port_id(port_cfg, 0);
	if (result == DOCA_SUCCESS)
		result = doca_flow_port_cfg_set_dev(port_cfg, dev);
	if (result == DOCA_SUCCESS)
		result = doca_flow_port_start(port_cfg, &df_port);
	doca_flow_port_cfg_destroy(port_cfg);
	return result;
}

/* Root pipe: Ethertype DSB_ETHERTYPE -> RSS to our single GPU RX queue. */
static doca_error_t create_rx_pipe(struct dsb_doca_io *io)
{
	doca_error_t result;
	struct doca_flow_match match = {0};
	struct doca_flow_match match_mask = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_pipe_cfg *pipe_cfg;
	uint16_t rss_queues[1];
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};

	doca_eth_rxq_apply_queue_id(io->rxq_cpu, 0);
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array = rss_queues;
	fwd.rss.outer_flags = 0;
	fwd.rss.nr_queues = 1;
	match_mask.outer.eth.type = 0xFFFF;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_flow_pipe_cfg_set_name(pipe_cfg, "DSB_RX_ROOT_PIPE");
	if (result == DOCA_SUCCESS)
		result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
	if (result == DOCA_SUCCESS)
		result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
	if (result == DOCA_SUCCESS)
		result = doca_flow_pipe_cfg_set_match(pipe_cfg, NULL, &match_mask);
	if (result == DOCA_SUCCESS)
		result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
	if (result == DOCA_SUCCESS)
		result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, &io->rx_pipe);
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	if (result != DOCA_SUCCESS)
		return result;

	match.outer.eth.type = htons(DSB_ETHERTYPE);
	result = doca_flow_pipe_basic_add_entry(0, io->rx_pipe, &match, 0, NULL, NULL, NULL,
						DOCA_FLOW_ENTRY_FLAGS_NO_WAIT, NULL, &io->rx_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add Ethertype 0x%04x steering entry: %s", DSB_ETHERTYPE,
			     doca_error_get_descr(result));
		return result;
	}
	return doca_flow_entries_process(df_port, 0, 0, 0);
}

/* Registers a GPU buffer with the NIC (dmabuf when available). */
static doca_error_t register_gpu_buffer(struct dsb_doca_io *io, size_t bytes, uint8_t **addr,
					int *dmabuf_fd, struct doca_mmap **mmap)
{
	doca_error_t result;

	result = doca_mmap_create(mmap);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_mmap_add_dev(*mmap, io->ddev);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_gpu_mem_alloc(io->gpu_dev, bytes, host_page_size(), DOCA_GPU_MEM_TYPE_GPU,
				    (void **)addr, NULL);
	if (result != DOCA_SUCCESS || *addr == NULL)
		return result != DOCA_SUCCESS ? result : DOCA_ERROR_NO_MEMORY;
	result = doca_gpu_dmabuf_fd(io->gpu_dev, *addr, bytes, dmabuf_fd);
	if (result != DOCA_SUCCESS) {
		*dmabuf_fd = -1;
		result = doca_mmap_set_memrange(*mmap, *addr, bytes);
	} else {
		result = doca_mmap_set_dmabuf_memrange(*mmap, *dmabuf_fd, *addr, 0, bytes);
	}
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_mmap_set_permissions(*mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
	if (result != DOCA_SUCCESS)
		return result;
	return doca_mmap_start(*mmap);
}

static doca_error_t create_rxq(struct dsb_doca_io *io, struct dsb_doca_cfg *cfg)
{
	doca_error_t result;
	uint32_t cyclic_bytes = 0;
	struct cudaDeviceProp prop;
	union doca_data user_data;

	result = doca_eth_rxq_create(io->ddev, DSB_RX_PKT_NUM, DSB_MAX_FRAME, &io->rxq_cpu);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_rxq_set_type(io->rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, DSB_MAX_FRAME,
						       DSB_RX_PKT_NUM, 0, 0, 0, &cyclic_bytes);
	if (result != DOCA_SUCCESS)
		return result;
	ALIGN_SIZE(cyclic_bytes, host_page_size());

	result = register_gpu_buffer(io, cyclic_bytes, &io->rx_buf, &io->rx_dmabuf_fd, &io->rx_mmap);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_rxq_set_pkt_buf(io->rxq_cpu, io->rx_mmap, 0, cyclic_bytes);
	if (result != DOCA_SUCCESS)
		return result;

	cudaGetDeviceProperties(&prop, cfg->cuda_id);
	if (prop.major < 9) {
		result = doca_eth_rxq_gpu_enable_mcst_qp(io->rxq_cpu);
		if (result != DOCA_SUCCESS)
			return result;
	}

	io->rxq_ctx = doca_eth_rxq_as_doca_ctx(io->rxq_cpu);
	if (io->rxq_ctx == NULL)
		return DOCA_ERROR_BAD_STATE;
	result = doca_ctx_set_datapath_on_gpu(io->rxq_ctx, io->gpu_dev);
	if (result != DOCA_SUCCESS)
		return result;
	user_data.ptr = io;
	result = doca_ctx_set_user_data(io->rxq_ctx, user_data);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_ctx_start(io->rxq_ctx);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_rxq_get_gpu_handle(io->rxq_cpu, &io->rxq_gpu);
	if (result != DOCA_SUCCESS)
		return result;
	return create_rx_pipe(io);
}

static doca_error_t create_txq(struct dsb_doca_io *io, struct dsb_doca_cfg *cfg, size_t tx_bytes)
{
	doca_error_t result;

	result = doca_eth_txq_create(io->ddev, DSB_TX_SQ_DEPTH, &io->txq_cpu);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_txq_set_l3_chksum_offload(io->txq_cpu, 0);
	if (result == DOCA_SUCCESS)
		result = doca_eth_txq_set_l4_chksum_offload(io->txq_cpu, 0);
	if (result == DOCA_SUCCESS)
		result = doca_eth_txq_gpu_set_completion_on_gpu(io->txq_cpu);
	if (result != DOCA_SUCCESS)
		return result;
	io->txq_ctx = doca_eth_txq_as_doca_ctx(io->txq_cpu);
	if (io->txq_ctx == NULL)
		return DOCA_ERROR_BAD_STATE;
	result = doca_ctx_set_datapath_on_gpu(io->txq_ctx, io->gpu_dev);
	if (result != DOCA_SUCCESS)
		return result;
	if (cfg->tx_mode == 1) {
		enum doca_eth_wait_on_time_type mode;
		result = doca_eth_txq_cap_get_wait_on_time_offload_supported(doca_dev_as_devinfo(io->ddev), &mode);
		if (result != DOCA_SUCCESS || mode != DOCA_ETH_WAIT_ON_TIME_TYPE_NATIVE) {
			DOCA_LOG_ERR("--tx-wait-send needs native wait-on-time support on the NIC");
			return DOCA_ERROR_NOT_SUPPORTED;
		}
		result = doca_eth_txq_set_wait_on_time_offload(io->txq_cpu);
		if (result != DOCA_SUCCESS)
			return result;
	}
	result = doca_ctx_start(io->txq_ctx);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_txq_apply_queue_id(io->txq_cpu, 0);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_eth_txq_get_gpu_handle(io->txq_cpu, &io->txq_gpu);
	if (result != DOCA_SUCCESS)
		return result;

	result = register_gpu_buffer(io, tx_bytes, &io->tx_buf, &io->tx_dmabuf_fd, &io->tx_mmap);
	if (result != DOCA_SUCCESS)
		return result;
	result = doca_mmap_get_mkey(io->tx_mmap, io->ddev, &io->tx_mkey);
	if (result != DOCA_SUCCESS)
		return result;
	io->tx_mkey = htobe32(io->tx_mkey);
	return DOCA_SUCCESS;
}

static void destroy_io(struct dsb_doca_io *io)
{
	if (io->rx_pipe)
		doca_flow_pipe_destroy(io->rx_pipe);
	if (df_port)
		doca_flow_port_stop(df_port);
	doca_flow_destroy();
	if (io->rxq_ctx)
		doca_ctx_stop(io->rxq_ctx);
	if (io->rxq_cpu)
		doca_eth_rxq_destroy(io->rxq_cpu);
	if (io->rx_mmap)
		doca_mmap_destroy(io->rx_mmap);
	if (io->rx_buf)
		doca_gpu_mem_free(io->gpu_dev, io->rx_buf);
	if (io->txq_ctx)
		doca_ctx_stop(io->txq_ctx);
	if (io->txq_cpu)
		doca_eth_txq_destroy(io->txq_cpu);
	if (io->tx_mmap)
		doca_mmap_destroy(io->tx_mmap);
	if (io->tx_buf)
		doca_gpu_mem_free(io->gpu_dev, io->tx_buf);
	if (io->gpu_dev)
		doca_gpu_destroy(io->gpu_dev);
	if (io->ddev)
		doca_dev_close(io->ddev);
}

/* ------------------------------------------------------------ device memory */

static void *dev_zero(size_t bytes)
{
	void *p = NULL;
	if (cudaMalloc(&p, bytes) != cudaSuccess)
		return NULL;
	if (cudaMemset(p, 0, bytes) != cudaSuccess)
		return NULL;
	return p;
}

static double mono_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC_RAW, &t);
	return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

doca_error_t dsb_doca_run(struct dsb_doca_cfg *cfg)
{
	doca_error_t result;
	struct dsb_doca_io io;
	struct dsb_ctx ctx;
	cudaStream_t s_io = NULL, s_work = NULL;
	uint32_t *exit_flag = NULL, *exit_flag_dev = NULL;
	struct dsb_stats *stats = NULL;
	int optin = 0, max_blocks;
	uint32_t smem, workers;

	memset(&io, 0, sizeof(io));
	memset(&ctx, 0, sizeof(ctx));
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	/* worker shared memory */
	cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cfg->cuda_id);
	smem = (cfg->smem_kb ? cfg->smem_kb : 48u) * 1024u;
	/* keep 2 KB for the worker kernel's static shared memory (item view, scalars) */
	if (smem > (uint32_t)optin - 2048u)
		smem = ((uint32_t)optin - 2048u) & ~15u;
	ctx.smem_bytes = smem;
	ctx.rsp_on_host = 0;

	result = open_doca_device_with_pci(cfg->nic_pcie_addr, NULL, &io.ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to open NIC %s", cfg->nic_pcie_addr);
		return result;
	}
	result = doca_gpu_create(cfg->gpu_pcie_addr, &io.gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create DOCA GPU handler: %s", doca_error_get_descr(result));
		goto out;
	}
	result = init_doca_flow(io.ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("DOCA Flow init failed: %s", doca_error_get_descr(result));
		goto out;
	}
	io.port = df_port;
	result = create_rxq(&io, cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("RX queue: %s", doca_error_get_descr(result));
		goto out;
	}
	result = create_txq(&io, cfg, (size_t)DSB_RSP_CAP * DSB_RSP_SLOT);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("TX queue: %s", doca_error_get_descr(result));
		goto out;
	}

	/* queues and the model store: plain device memory, rsp ring = TX buffer.
	 * exit flag: host-mapped (zero-copy), as in dsb_bench; stats: managed, as in the TVG receiver */
	result = DOCA_ERROR_NO_MEMORY;
	if (cudaHostAlloc((void **)&exit_flag, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
	    cudaHostGetDevicePointer((void **)&exit_flag_dev, exit_flag, 0) != cudaSuccess ||
	    cudaMallocManaged((void **)&stats, sizeof(*stats), cudaMemAttachGlobal) != cudaSuccess)
		goto out;
	*exit_flag = 0;
	memset(stats, 0, sizeof(*stats));
	ctx.exit_flag = exit_flag_dev;
	ctx.stats = stats;
	/* the two ring counters are managed so the monitor can read them without a copy
	 * (a cudaMemcpy while the persistent kernels hold every SM slot does not complete) */
	if (cudaMallocManaged((void **)&ctx.work_claim, sizeof(unsigned long long), cudaMemAttachGlobal) != cudaSuccess ||
	    cudaMallocManaged((void **)&ctx.work_done, sizeof(unsigned long long), cudaMemAttachGlobal) != cudaSuccess)
		goto out;
	*ctx.work_claim = 0;
	*ctx.work_done = 0;
	ctx.items = dev_zero(sizeof(struct dsb_work_item) * DSB_WORK_CAP);
	ctx.frame_pool = dev_zero((size_t)DSB_WORK_CAP * DSB_MAX_FRAME);
	ctx.rsp_head = dev_zero(sizeof(unsigned long long));
	ctx.rsp_released = dev_zero(sizeof(unsigned long long));
	ctx.rsp_ready = dev_zero(sizeof(unsigned long long) * DSB_RSP_CAP);
	ctx.rsp_tend = dev_zero(sizeof(unsigned long long) * DSB_RSP_CAP);
	ctx.models = dev_zero(sizeof(struct dsb_model_desc) * DSB_MAX_MODELS);
	ctx.model_pool = dev_zero((size_t)DSB_MAX_MODELS * DSB_MODEL_SLOT_BYTES);
	ctx.rsp_buf = io.tx_buf;
	if (!ctx.work_claim || !ctx.work_done || !ctx.items || !ctx.frame_pool || !ctx.rsp_head || !ctx.rsp_released ||
	    !ctx.rsp_ready || !ctx.rsp_tend || !ctx.models || !ctx.model_pool)
		goto out;
	if (cudaMemset(io.tx_buf, 0, (size_t)DSB_RSP_CAP * DSB_RSP_SLOT) != cudaSuccess)
		goto out;
	cudaDeviceSynchronize();

	max_blocks = dsb_doca_max_blocks(smem);
	if (max_blocks < 5) {
		DOCA_LOG_ERR("Only %d resident worker CTAs with %u B shared memory; need >= 5", max_blocks, smem);
		result = DOCA_ERROR_NOT_SUPPORTED;
		goto out;
	}
	/* leave slots free: one each for the RX and TX kernels plus a margin, so the GPU is
	 * never completely full of persistent blocks (a fully booked GPU stalls anything the
	 * driver has to run) */
	workers = (uint32_t)max_blocks - 4u;
	if (cfg->workers > 0 && cfg->workers < workers)
		workers = cfg->workers;
	DOCA_LOG_INFO("Worker CTAs: %u (%u B shared memory each, %u threads), RX warp + TX thread (tx mode %u)",
		      workers, smem, DSB_THREADS, cfg->tx_mode);

	if (cudaStreamCreateWithFlags(&s_io, cudaStreamNonBlocking) != cudaSuccess ||
	    cudaStreamCreateWithFlags(&s_work, cudaStreamNonBlocking) != cudaSuccess) {
		result = DOCA_ERROR_BAD_STATE;
		goto out;
	}
	result = dsb_doca_launch(s_io, s_work, ctx, io.rxq_gpu, io.txq_gpu, io.tx_mkey, workers, cfg->tx_mode);
	if (result != DOCA_SUCCESS)
		goto out;

	DOCA_LOG_INFO("dSB server running on Ethertype 0x%04x (Ctrl+C to stop)", DSB_ETHERTYPE);
	setvbuf(stdout, NULL, _IOLBF, 0); /* line-buffered even through a pipe / tee */
	printf("%8s | %12s %12s %10s %12s %12s | %8s %6s %6s %6s | %6s %6s %6s %6s %6s\n", "t(s)", "rx_frames",
	       "solve_rx", "loads", "solved", "tx_frames", "kreq/s", "drop", "bad", "txerr", "ready", "acks",
	       "claim", "done", "posted");
	{
		double t0 = mono_s(), tp = t0;
		unsigned long long prev_solved = 0;
		int hinted = 0;
		while (!force_quit) {
			usleep(1000u * (cfg->report_ms ? cfg->report_ms : 1000u));
			const char *ended = NULL;
			cudaError_t qerr = cudaSuccess;
			if ((qerr = cudaStreamQuery(s_io)) != cudaErrorNotReady)
				ended = "RX/TX";
			else if ((qerr = cudaStreamQuery(s_work)) != cudaErrorNotReady)
				ended = "worker";
			if (ended) {
				fprintf(stderr, "[DSB_FATAL] %s kernel ended: %s (rx_bad=%llu tx_errors=%llu", ended,
					cudaGetErrorString(qerr), stats->rx_bad, stats->tx_errors);
				if (stats->tx_errors)
					fprintf(stderr, " last tx error: %s", doca_error_get_descr((doca_error_t)stats->tx_last_error));
				fprintf(stderr, ")\n");
				if (stats->tx_errors)
					fprintf(stderr, "[DSB_FATAL] the reply send failed in tx mode %u; retry with -W %u\n",
						cfg->tx_mode, (cfg->tx_mode + 1u) % 3u);
				break;
			}
			double t = mono_s();
			unsigned long long solved = stats->solved;
			/* work-ring progress (managed memory, read like stats): tickets claimed by
			 * worker CTAs (0 = the worker kernel is not running) and replies published */
			const unsigned long long claim = *ctx.work_claim, done = *ctx.work_done;
			printf("%8.1f | %12llu %12llu %10llu %12llu %12llu | %8.1f %6llu %6llu %6llu | %6llu %6llu %6llu %6llu %6llu\n",
			       t - t0, stats->rx_frames, stats->rx_solve, stats->rx_load, solved, stats->tx_frames,
			       (double)(solved - prev_solved) / (t - tp) * 1e-3, stats->rx_dropped, stats->rx_bad,
			       stats->tx_errors, stats->models_ready, stats->rx_acks, claim, done, stats->tx_posted);
			fflush(stdout);
			prev_solved = solved;
			tp = t;
			if (!hinted && stats->rx_frames == 0 && t - t0 > 10.0) {
				hinted = 1;
				fprintf(stderr,
					"[DSB_HINT] no frame with Ethertype 0x%04x has reached the GPU queue in 10 s.\n"
					"  The RX path is the TVG receiver's; if the client is sending, check:\n"
					"  - client --server MAC is this NIC's MAC (ip link show on the server), link up on both ends\n"
					"  - client sendto() did not fail (the client now aborts and prints the errno)\n"
					"  - no other DOCA/base server is running on this NIC\n"
					"  - the TVG receiver still works on this NIC (same steering, Ethertype 0xFEED)\n",
					DSB_ETHERTYPE);
			}
		}
	}
	__atomic_store_n(exit_flag, 1u, __ATOMIC_SEQ_CST);
	cudaStreamSynchronize(s_io);
	cudaStreamSynchronize(s_work);
	result = DOCA_SUCCESS;

out:
	if (exit_flag)
		__atomic_store_n(exit_flag, 1u, __ATOMIC_SEQ_CST);
	cudaDeviceSynchronize();
	if (s_io)
		cudaStreamDestroy(s_io);
	if (s_work)
		cudaStreamDestroy(s_work);
	cudaFree(ctx.work_claim);
	cudaFree(ctx.work_done);
	cudaFree(ctx.items);
	cudaFree(ctx.frame_pool);
	cudaFree(ctx.rsp_head);
	cudaFree(ctx.rsp_released);
	cudaFree(ctx.rsp_ready);
	cudaFree(ctx.rsp_tend);
	cudaFree(ctx.models);
	cudaFree(ctx.model_pool);
	if (exit_flag)
		cudaFreeHost(exit_flag);
	if (stats)
		cudaFree(stats);
	destroy_io(&io);
	return result;
}
