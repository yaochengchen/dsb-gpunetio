# Non-DOCA targets (plain CUDA + C++).  The DOCA server is built with meson:
#   make doca ARCH=sm_120      (PU2, RTX PRO 6000)   /  ARCH=sm_90 (GH200)
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_120
CXX       ?= g++

# -ffp-contract=off keeps the host reference bit-identical to the GPU update
NVFLAGS  := -std=c++17 -O3 -arch=$(ARCH) -Iinclude -Xcompiler -ffp-contract=off,-Wall
CXXFLAGS := -std=c++17 -O2 -ffp-contract=off -Wall -Iinclude
HDRS     := $(wildcard include/*.h include/*.hpp include/*.cuh)

# PU2 (x86_64) and PU3 (aarch64) share this directory: keep binaries per
# architecture, bin/x86_64 and bin/aarch64; the DOCA build dir likewise.
HOSTARCH := $(shell uname -m)
BIN      ?= bin/$(HOSTARCH)
BUILD    ?= build_$(HOSTARCH)
all: $(BIN)/selftest $(BIN)/dsb_bench $(BIN)/dsb_base_server $(BIN)/dsb_client $(BIN)/cpu_test $(BIN)/cpu_server

$(BIN):
	mkdir -p $(BIN)

$(BIN)/selftest: tools/selftest.cu $(HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@
$(BIN)/dsb_bench: tools/dsb_bench.cu $(HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@
$(BIN)/dsb_base_server: server_base/base_server.cu $(HDRS) | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@
$(BIN)/dsb_client: client/dsb_client.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ -lpthread
$(BIN)/cpu_test: tests/cpu_test.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BIN)/cpu_server: tests/cpu_server.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

# client only (e.g. on the load-generator host without CUDA)
client: $(BIN)/dsb_client $(BIN)/cpu_test

doca:
	@if [ ! -d $(BUILD) ]; then meson setup $(BUILD) -Ddsb_cuda_arch=$(ARCH); else meson configure $(BUILD) -Ddsb_cuda_arch=$(ARCH); fi
	ninja -C $(BUILD)
	mkdir -p $(BIN) && ln -sf $(CURDIR)/$(BUILD)/dsb_doca_server $(BIN)/dsb_doca_server

test: $(BIN)/cpu_test $(BIN)/selftest
	$(BIN)/cpu_test && $(BIN)/selftest

clean:
	rm -rf $(BIN) $(BUILD)

.PHONY: all client doca test clean
