#pragma once

#include "paged_allocator.hpp"
#include "small_object_allocator.hpp"

namespace data_engine {
namespace allocator {

class ComplicatedAllocator : public Allocator {
  public:
    ComplicatedAllocator(size_t threshold, size_t page_size, size_t num_pages,
                         size_t soa_buffer_size, void *pages_base,
                         void *soa_buffer_base);

    void *allocate(size_t size) override;
    void free(void *ptr) override;
    bool owns(void *ptr) override;

  private:
    PagedAllocator _m_paged_allocator;
    SmallObjectAllocator _m_soa;
    void *_m_pages_base;
    void *_m_soa_buffer_base;
    size_t _m_threshold;
    size_t _m_page_size;
    size_t _m_num_pages;
    size_t _m_soa_buffer_size;
};
} // namespace allocator
} // namespace data_engine
