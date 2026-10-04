# Streaming mode.  Run from the repository root:  make -C stream -f stream.mk ARCH=sm_120
#   non-DOCA targets (nvcc):  st_ref (resident-matrix reference), st_cpu_recv (kernel-stack receiver)
#   DOCA targets (meson):     st_recv (PU2)  st_gen (PU3)   ->  make -C stream doca ARCH=sm_120|sm_90
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_120
HOSTARCH  := $(shell uname -m)
BIN       ?= ../bin/$(HOSTARCH)
# meson build dir OUTSIDE stream/: stream/ is replaced wholesale from the Windows side, which would wipe it
BUILD     ?= ../build_stream_$(HOSTARCH)
NVFLAGS   := -std=c++17 -O3 -arch=$(ARCH) -I. -Xcompiler -ffp-contract=off,-Wall,-pthread
HDRS      := stream_proto.h stream_kernels.cuh st_net.h
# DOCA pkg-config files (see README.md, Build); both the x86 (lib64) and the aarch64 layouts
export PKG_CONFIG_PATH := /opt/mellanox/doca/lib64/pkgconfig:/opt/mellanox/doca/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/doca/lib/aarch64-linux-gnu/pkgconfig:$(PKG_CONFIG_PATH)

all: $(BIN)/st_ref $(BIN)/st_cpu_recv

$(BIN):
	mkdir -p $(BIN)
$(BIN)/st_ref: ref.cu state.cu $(HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) ref.cu state.cu -o $@
$(BIN)/st_cpu_recv: cpu_recv.cu state.cu $(HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) cpu_recv.cu state.cu -o $@ -lpthread

doca: all
	@if [ ! -d $(BUILD) ]; then meson setup $(BUILD) -Dst_cuda_arch=$(ARCH); else meson configure $(BUILD) -Dst_cuda_arch=$(ARCH); fi
	ninja -C $(BUILD)
	mkdir -p $(BIN) && cp -f $(BUILD)/st_recv $(BIN)/st_recv && cp -f $(BUILD)/st_gen $(BIN)/st_gen   # real files, not symlinks into the build dir

clean:
	rm -rf $(BUILD) $(BIN)/st_ref $(BIN)/st_cpu_recv $(BIN)/st_recv $(BIN)/st_gen
.PHONY: all doca clean
