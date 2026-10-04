# DOCA GPUNetIO dSB server (mirrors TVG-suite meson.build).
#   meson setup build -Ddsb_cuda_arch=sm_120    (PU2, RTX PRO 6000)
#   meson setup build -Ddsb_cuda_arch=sm_90     (GH200)
#   ninja -C build
project('dsb_doca', 'C', 'CPP', 'CUDA',
	version: '0.1.0',
	default_options: ['buildtype=release', 'c_std=gnu11', 'cpp_std=c++17'],
	meson_version: '>= 0.61.2'
)

add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp', 'cuda'])

inc_dirs = include_directories(
	'include',
	'server_doca',
	'/opt/mellanox/doca/include',
	'/opt/mellanox/doca/samples',
	'/usr/local/cuda/include'
)

doca_common_dep = dependency('doca-common')
doca_gpu_dep    = dependency('doca-gpunetio')
doca_argp_dep   = dependency('doca-argp')
doca_eth_dep    = dependency('doca-eth')
doca_flow_dep   = dependency('doca-flow')
cuda_dep        = dependency('cuda', version: '>= 12-1', modules: ['cudart', 'cuda'])

doca_samples_common = files('/opt/mellanox/doca/samples/common.c')

cuda_flags = ['-arch=' + get_option('dsb_cuda_arch'), '-O3', '-std=c++17']
if get_option('tx_wait_send_only')
	cuda_flags += ['-DDSB_TX_WAIT_SEND_ONLY']
endif

executable('dsb_doca_server',
	['server_doca/doca_main.c', 'server_doca/doca_host.c', 'server_doca/doca_kernel.cu',
	 doca_samples_common],
	include_directories: inc_dirs,
	dependencies: [doca_common_dep, doca_gpu_dep, doca_argp_dep, doca_eth_dep, doca_flow_dep, cuda_dep],
	cuda_args: cuda_flags,
	install: false
)
