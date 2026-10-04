# Streaming mode (DOCA GPUNetIO targets).  Separate from ../meson.build so the
# request-service build is untouched.
#   PU2: meson setup build_stream_x86_64  -Dst_cuda_arch=sm_120 && ninja -C build_stream_x86_64 st_recv
#   PU3: meson setup build_stream_aarch64 -Dst_cuda_arch=sm_90  && ninja -C build_stream_aarch64 st_gen
project('dsb_stream', 'C', 'CPP', 'CUDA',
	version: '0.1.0',
	default_options: ['buildtype=release', 'c_std=gnu11', 'cpp_std=c++17'],
	meson_version: '>= 0.61.2'
)
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp', 'cuda'])

inc_dirs = include_directories('.', '/opt/mellanox/doca/include', '/opt/mellanox/doca/samples', '/usr/local/cuda/include')

doca_common_dep = dependency('doca-common')
doca_gpu_dep    = dependency('doca-gpunetio')
doca_eth_dep    = dependency('doca-eth')
doca_flow_dep   = dependency('doca-flow')
cuda_dep        = dependency('cuda', version: '>= 12-1', modules: ['cudart', 'cuda'])
doca_samples_common = files('/opt/mellanox/doca/samples/common.c')
deps = [doca_common_dep, doca_gpu_dep, doca_eth_dep, doca_flow_dep, cuda_dep]
cuda_flags = ['-arch=' + get_option('st_cuda_arch'), '-O3', '-std=c++17', '-Xcompiler', '-ffp-contract=off']

# solver side (PU2)
executable('st_recv', ['recv_host.c', 'recv_kernel.cu', 'state.cu', doca_samples_common],
	include_directories: inc_dirs, dependencies: deps, cuda_args: cuda_flags, install: false)

# source side (PU3)
executable('st_gen', ['gen_host.c', 'gen_kernel.cu', doca_samples_common],
	include_directories: inc_dirs, dependencies: deps, cuda_args: cuda_flags, install: false)
