/*
 * doca_main.c -- CLI of the DOCA GPUNetIO dSB server.
 *
 *   sudo ./bin/dsb_doca_server -g <GPU PCIe> -n <NIC PCIe> [-s <smem KB>] [-w <workers>]
 *                              [-W <tx mode 0|1|2>] [-r <report ms>]
 *
 * Defaults are the PU2 addresses of the TVG suite (RTX PRO 6000 0000:ab:00.0,
 * ConnectX-7 0000:bd:00.0).
 */
#include <doca_argp.h>
#include <doca_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "doca_server.h"

DOCA_LOG_REGISTER(DSB::MAIN);

typedef doca_error_t (*dsb_param_cb)(void *param, void *config);

static doca_error_t copy_pci(char *dst, const char *src)
{
	if (strnlen(src, MAX_PCI_ADDRESS_LEN) >= MAX_PCI_ADDRESS_LEN) {
		DOCA_LOG_ERR("PCI address too long");
		return DOCA_ERROR_INVALID_VALUE;
	}
	snprintf(dst, MAX_PCI_ADDRESS_LEN, "%s", src);
	return DOCA_SUCCESS;
}

static doca_error_t gpu_cb(void *param, void *config)
{
	return copy_pci(((struct dsb_doca_cfg *)config)->gpu_pcie_addr, (const char *)param);
}
static doca_error_t nic_cb(void *param, void *config)
{
	return copy_pci(((struct dsb_doca_cfg *)config)->nic_pcie_addr, (const char *)param);
}
static doca_error_t smem_cb(void *param, void *config)
{
	int v = *(int *)param;
	if (v < 8 || v > 227)
		return DOCA_ERROR_INVALID_VALUE;
	((struct dsb_doca_cfg *)config)->smem_kb = (uint32_t)v;
	return DOCA_SUCCESS;
}
static doca_error_t workers_cb(void *param, void *config)
{
	int v = *(int *)param;
	if (v < 0)
		return DOCA_ERROR_INVALID_VALUE;
	((struct dsb_doca_cfg *)config)->workers = (uint32_t)v;
	return DOCA_SUCCESS;
}
static doca_error_t wait_cb(void *param, void *config)
{
	int v = *(int *)param;
	if (v < 0 || v > 2)
		return DOCA_ERROR_INVALID_VALUE;
	((struct dsb_doca_cfg *)config)->tx_mode = (uint32_t)v;
	return DOCA_SUCCESS;
}
static doca_error_t report_cb(void *param, void *config)
{
	int v = *(int *)param;
	((struct dsb_doca_cfg *)config)->report_ms = v > 0 ? (uint32_t)v : 1000u;
	return DOCA_SUCCESS;
}

static doca_error_t add_param(const char *s, const char *l, const char *args, const char *desc,
			      dsb_param_cb cb, enum doca_argp_type type)
{
	struct doca_argp_param *p;
	doca_error_t r = doca_argp_param_create(&p);
	if (r != DOCA_SUCCESS)
		return r;
	doca_argp_param_set_short_name(p, s);
	doca_argp_param_set_long_name(p, l);
	doca_argp_param_set_arguments(p, args);
	doca_argp_param_set_description(p, desc);
	doca_argp_param_set_callback(p, cb);
	doca_argp_param_set_type(p, type);
	return doca_argp_register_param(p);
}

static doca_error_t register_params(void)
{
	doca_error_t r;
	r = add_param("g", "gpu", "<GPU PCIe address>", "GPU that serves requests (PU2: 0000:ab:00.0)",
		      gpu_cb, DOCA_ARGP_TYPE_STRING);
	if (r == DOCA_SUCCESS)
		r = add_param("n", "nic", "<NIC PCIe address>", "ConnectX NIC (PU2: 0000:bd:00.0)", nic_cb,
			      DOCA_ARGP_TYPE_STRING);
	if (r == DOCA_SUCCESS)
		r = add_param("s", "smem", "<KB>", "shared memory per worker CTA in KB (default 48)", smem_cb,
			      DOCA_ARGP_TYPE_INT);
	if (r == DOCA_SUCCESS)
		r = add_param("w", "workers", "<count>", "worker CTAs (default: all that fit)", workers_cb,
			      DOCA_ARGP_TYPE_INT);
	if (r == DOCA_SUCCESS)
		r = add_param("W", "tx-mode", "<0|1|2>",
			      "reply send path: 0 txq_send (default), 1 wait_send ts=0 (TVG), 2 wqe_prepare+submit+poll_at",
			      wait_cb, DOCA_ARGP_TYPE_INT);
	if (r == DOCA_SUCCESS)
		r = add_param("r", "report-ms", "<ms>", "counter print interval (default 1000)", report_cb,
			      DOCA_ARGP_TYPE_INT);
	return r;
}

int main(int argc, char **argv)
{
	struct dsb_doca_cfg cfg;
	struct doca_log_backend *sdk_log;
	doca_error_t result;
	int exit_status = EXIT_FAILURE;

	/* more hardware work queues than streams, so the RX/TX kernel and the worker
	 * kernel never share one and block each other (must be set before CUDA starts) */
	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);

	memset(&cfg, 0, sizeof(cfg));
	snprintf(cfg.gpu_pcie_addr, sizeof(cfg.gpu_pcie_addr), "%s", "0000:ab:00.0");
	snprintf(cfg.nic_pcie_addr, sizeof(cfg.nic_pcie_addr), "%s", "0000:bd:00.0");
	cfg.report_ms = 1000;

	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);

	result = doca_argp_init(NULL, &cfg);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = register_params();
	if (result != DOCA_SUCCESS)
		goto argp_cleanup;
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS)
		goto argp_cleanup;

	/* CUDA context on the chosen GPU before any DOCA GPU object (TVG v1.1.1 order) */
	if (cudaDeviceGetByPCIBusId(&cfg.cuda_id, cfg.gpu_pcie_addr) != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU bus id %s", cfg.gpu_pcie_addr);
		goto argp_cleanup;
	}
	if (cudaSetDevice(cfg.cuda_id) != cudaSuccess || cudaFree(0) != cudaSuccess) {
		DOCA_LOG_ERR("CUDA initialisation failed on device %d", cfg.cuda_id);
		goto argp_cleanup;
	}
	DOCA_LOG_INFO("GPU %s (CUDA %d), NIC %s", cfg.gpu_pcie_addr, cfg.cuda_id, cfg.nic_pcie_addr);

	result = dsb_doca_run(&cfg);
	if (result == DOCA_SUCCESS)
		exit_status = EXIT_SUCCESS;
	else
		DOCA_LOG_ERR("server failed: %s", doca_error_get_descr(result));

argp_cleanup:
	doca_argp_destroy();
	return exit_status;
}
