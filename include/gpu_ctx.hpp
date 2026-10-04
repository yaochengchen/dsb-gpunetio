// gpu_ctx.hpp -- allocation of the queues/model store with the plain CUDA
// runtime (no DOCA).  Used by the baseline server, the self test and the
// benchmark.  The DOCA server allocates the same structure with doca_gpu_mem.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "dsb_queue.h"
#include "dsb_solve.cuh"

#define CK(call)                                                                      \
  do {                                                                                \
    cudaError_t e_ = (call);                                                          \
    if (e_ != cudaSuccess)                                                            \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_) +   \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));   \
  } while (0)

namespace dsbgpu {

/// Largest dynamic shared memory per worker CTA we will ask for.
inline uint32_t pick_smem(int device, uint32_t requested) {
  int optin = 0;
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  // static __shared__ of the kernels (Scalars, item, ...) is < 1 KB
  uint32_t lim = uint32_t(optin) - 1024u;
  uint32_t s = requested ? requested : 48u * 1024u;
  if (s > lim) s = lim;
  return s & ~15u;
}

/// Worker CTAs that can be resident at once with `smem` bytes each.
inline int resident_workers(const void* kernel, uint32_t smem, int device) {
  CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  int per_sm = 0, sms = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, DSB_THREADS, smem));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return per_sm * sms;
}

template <class T>
T* host_mapped(size_t count, T** dev) {
  void* h = nullptr;
  CK(cudaHostAlloc(&h, count * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
  std::memset(h, 0, count * sizeof(T));
  void* d = nullptr;
  CK(cudaHostGetDevicePointer(&d, h, 0));
  *dev = static_cast<T*>(d);
  return static_cast<T*>(h);
}

template <class T>
T* device_zero(size_t count) {
  void* d = nullptr;
  CK(cudaMalloc(&d, count * sizeof(T)));
  CK(cudaMemset(d, 0, count * sizeof(T)));
  return static_cast<T*>(d);
}

struct GpuCtx {
  dsb_ctx ctx{};         // what kernels see
  bool host_queues = false;

  // host views (valid when host_queues, except exit/stats which always are)
  uint32_t* exit_h = nullptr;
  dsb_work_item* items_h = nullptr;
  unsigned long long* ready_h = nullptr;
  unsigned long long* tend_h = nullptr;
  unsigned long long* released_h = nullptr;
  uint8_t* rsp_h = nullptr;
  dsb_stats* stats = nullptr;  // managed
  cudaStream_t copy_stream = nullptr;

  /// host_queues: work items and the reply ring live in pinned host memory
  /// (CPU producer / CPU consumer).  Otherwise everything is device memory.
  void init(bool hq, uint32_t smem_bytes) {
    host_queues = hq;
    ctx.smem_bytes = smem_bytes;
    ctx.rsp_on_host = hq ? 1u : 0u;
    uint32_t* exit_d = nullptr;
    exit_h = host_mapped<uint32_t>(1, &exit_d);
    ctx.exit_flag = exit_d;
    ctx.work_claim = device_zero<unsigned long long>(1);
    ctx.work_done = device_zero<unsigned long long>(1);
    ctx.rsp_head = device_zero<unsigned long long>(1);
    ctx.models = device_zero<dsb_model_desc>(DSB_MAX_MODELS);
    ctx.model_pool = device_zero<uint8_t>(size_t(DSB_MAX_MODELS) * DSB_MODEL_SLOT_BYTES);
    if (hq) {
      items_h = host_mapped<dsb_work_item>(DSB_WORK_CAP, &ctx.items);
      ready_h = host_mapped<unsigned long long>(DSB_RSP_CAP, &ctx.rsp_ready);
      tend_h = host_mapped<unsigned long long>(DSB_RSP_CAP, &ctx.rsp_tend);
      released_h = host_mapped<unsigned long long>(1, &ctx.rsp_released);
      rsp_h = host_mapped<uint8_t>(size_t(DSB_RSP_CAP) * DSB_RSP_SLOT, &ctx.rsp_buf);
    } else {
      ctx.items = device_zero<dsb_work_item>(DSB_WORK_CAP);
      ctx.rsp_ready = device_zero<unsigned long long>(DSB_RSP_CAP);
      ctx.rsp_tend = device_zero<unsigned long long>(DSB_RSP_CAP);
      ctx.rsp_released = device_zero<unsigned long long>(1);
      ctx.rsp_buf = device_zero<uint8_t>(size_t(DSB_RSP_CAP) * DSB_RSP_SLOT);
    }
    CK(cudaMallocManaged(&stats, sizeof(dsb_stats)));
    std::memset(stats, 0, sizeof(dsb_stats));
    ctx.stats = stats;
    CK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  }

  /// Upload a packed model; safe while persistent kernels run (separate
  /// non-blocking stream; the descriptor is written after the data).
  void upload_model(uint16_t id, uint16_t n, uint8_t format, float xi_default,
                    const std::vector<uint8_t>& packed, uint32_t generation) {
    if (id >= DSB_MAX_MODELS || packed.size() > DSB_MODEL_SLOT_BYTES)
      throw std::runtime_error("model does not fit its slot");
    dsb_model_desc d{};
    dsb_model_desc* dd = ctx.models + id;
    // invalidate, copy data, publish
    CK(cudaMemcpyAsync(dd, &d, sizeof(d), cudaMemcpyHostToDevice, copy_stream));
    CK(cudaMemcpyAsync(ctx.model_pool + size_t(id) * DSB_MODEL_SLOT_BYTES, packed.data(),
                       packed.size(), cudaMemcpyHostToDevice, copy_stream));
    d.generation = generation;
    d.total_bytes = uint32_t(packed.size());
    d.n = n;
    d.format = format;
    d.xi_default = xi_default;
    CK(cudaMemcpyAsync(dd, &d, sizeof(d), cudaMemcpyHostToDevice, copy_stream));  // ready_gen still 0
    // publish: ready_gen last, in its own copy, after every other field has landed
    static uint32_t ready_word;
    ready_word = generation ? generation : 1u;
    CK(cudaMemcpyAsync(&dd->ready_gen, &ready_word, sizeof(uint32_t), cudaMemcpyHostToDevice, copy_stream));
    CK(cudaStreamSynchronize(copy_stream));
  }

  void destroy() {
    auto freeh = [](void* p) { if (p) cudaFreeHost(p); };
    freeh(exit_h);
    cudaFree(ctx.work_claim);
    cudaFree(ctx.work_done);
    cudaFree(ctx.rsp_head);
    cudaFree(ctx.models);
    cudaFree(ctx.model_pool);
    if (host_queues) {
      freeh(items_h); freeh(ready_h); freeh(tend_h); freeh(released_h); freeh(rsp_h);
    } else {
      cudaFree(ctx.items); cudaFree(ctx.rsp_ready); cudaFree(ctx.rsp_tend);
      cudaFree(ctx.rsp_released); cudaFree(ctx.rsp_buf);
    }
    cudaFree(stats);
    if (copy_stream) cudaStreamDestroy(copy_stream);
  }
};

// ---- host-side producer / consumer for host_queues mode ---------------------

/// Publishes one work item (the frame must already be in GPU-readable memory).
/// Returns false when the ring is full (caller retries).  Single producer.
struct HostProducer {
  GpuCtx* g = nullptr;
  unsigned long long tail = 0;
  unsigned long long outstanding_limit = DSB_WORK_CAP;

  bool push(const uint8_t* frame_dev, uint32_t bytes, uint32_t kind, unsigned long long completed) {
    if (tail - completed >= outstanding_limit) return false;
    dsb_work_item* it = g->items_h + (tail & (DSB_WORK_CAP - 1u));
    if (!dsb_slot_free(__atomic_load_n(&it->ack, __ATOMIC_ACQUIRE), tail)) return false;
    it->t_rx = 0;
    it->frame = frame_dev;
    it->bytes = bytes;
    it->kind = kind;
    __atomic_store_n(&it->seq, tail + 1ull, __ATOMIC_RELEASE);
    ++tail;
    return true;
  }
};

/// Pops replies in order.  Single consumer.
struct HostConsumer {
  GpuCtx* g = nullptr;
  unsigned long long next = 0;

  /// Returns a pointer to the next reply (and its length), or nullptr.
  const uint8_t* peek(uint32_t& len) {
    const uint32_t slot = uint32_t(next & (DSB_RSP_CAP - 1u));
    const unsigned long long w = __atomic_load_n(g->ready_h + slot, __ATOMIC_ACQUIRE);
    if (dsb_ready_seq(w) != next + 1ull) return nullptr;
    len = dsb_ready_len(w);
    return g->rsp_h + size_t(slot) * DSB_RSP_SLOT;
  }
  void pop() {
    ++next;
    __atomic_store_n(g->released_h, next, __ATOMIC_RELEASE);
  }
};

}  // namespace dsbgpu
