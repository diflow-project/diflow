#pragma once

#include "data_engine/allocator/buddy_allocator.hpp"
// #include "data_engine/allocator/complicated_allocator.hpp"
#include "data_engine/engine/backend.hpp"
#include <memory>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace data_engine {
namespace engine {
namespace backend {
namespace nvshmem {
class NvshmemDataEngineBackend : public DataEngineBackend {
  public:
    NvshmemDataEngineBackend(int64_t arena_size,
                              // int64_t page_size, int64_t num_pages,
                              // int64_t soa_buffer_size, int64_t soa_threshold,
                              int64_t device_id, int64_t worker_id);
    ~NvshmemDataEngineBackend();

    int64_t nvshmem_pe() const;

    torch::Tensor create_tensor(const std::vector<int64_t> &size,
                                const pybind11::object &dtype) override;
    void free_tensor(torch::Tensor tensor) override;
    bool owns_tensor(torch::Tensor tensor);

    torch::Tensor fetch_tensor(int64_t remote_src,
                               const std::vector<int64_t> &size,
                               const pybind11::object &dtype,
                               int64_t remote_device_id);

  private:
    void _init_nvshmem();

    void *_m_arena_base; // Buddy Allocator
    // void *_m_paged_shm_base;      // Paged Allocator
    // void *_m_soa_buffer_shm_base; // Small Object Allocator

    std::unique_ptr<allocator::BuddyAllocator> _m_allocator;
    // std::unique_ptr<allocator::ComplicatedAllocator> _m_allocator;

    int64_t _m_arena_size;
    // int64_t _m_page_size;
    // int64_t _m_num_pages;
    // int64_t _m_soa_buffer_size;
    // int64_t _m_soa_threshold;

    int64_t _m_device_id;
    int64_t _m_worker_id;
    int64_t _m_nvshmem_pe;
};
} // namespace nvshmem
} // namespace backend
} // namespace engine
} // namespace data_engine
