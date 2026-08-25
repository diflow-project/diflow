#include "complicated_allocator.hpp"

namespace data_engine {
namespace allocator {

ComplicatedAllocator::ComplicatedAllocator(size_t threshold, size_t page_size,
                                           size_t num_pages,
                                           size_t soa_buffer_size,
                                           void *pages_base,
                                           void *soa_buffer_base)
    : _m_paged_allocator(page_size, num_pages, pages_base),
      _m_soa(soa_buffer_size, soa_buffer_base), _m_threshold(threshold) {}

void *ComplicatedAllocator::allocate(size_t size) {
    if (size < _m_threshold) {
        return _m_soa.allocate(size);
    }
    return _m_paged_allocator.allocate(size);
}

void ComplicatedAllocator::free(void *ptr) {
    if (_m_paged_allocator.owns(ptr)) {
        _m_paged_allocator.free(ptr);
        return;
    }
    if (_m_soa.owns(ptr)) {
        _m_soa.free(ptr);
        return;
    }
    throw std::runtime_error("Pointer not owned by allocator");
}

bool ComplicatedAllocator::owns(void *ptr) {
    return _m_paged_allocator.owns(ptr) || _m_soa.owns(ptr);
}

} // namespace allocator
} // namespace data_engine
