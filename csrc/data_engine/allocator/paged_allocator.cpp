#include "paged_allocator.hpp"

#include <stdexcept>
#include <mutex>
#include <condition_variable>

namespace data_engine {
namespace allocator {

PagedAllocator::PagedAllocator(size_t page_size, size_t num_pages,
                               void *pages_base)
    : _m_pages_base(pages_base), _m_page_size(page_size),
      _m_num_pages(num_pages) {
    std::lock_guard<std::mutex> lock(_m_mutex);
    for (size_t i = 0; i < num_pages; i++) {
        _m_free_pages.insert(static_cast<char *>(_m_pages_base) +
                             i * page_size);
    }
}

void *PagedAllocator::allocate(size_t size) {
    std::unique_lock<std::mutex> lock(_m_mutex);

    if (size > _m_page_size) {
        throw std::runtime_error("Size too large");
    }

    // Wait until a free page becomes available
    _m_cv.wait(lock, [this]() {
        return _m_free_pages.size() > 0;
    });

    auto page = *_m_free_pages.begin();
    _m_free_pages.erase(page);

    return page;
}

void PagedAllocator::free(void *ptr) {
    std::lock_guard<std::mutex> lock(_m_mutex);

    if (!_owns(ptr)) {
        throw std::runtime_error("Pointer not owned by allocator");
    }
    _m_free_pages.insert(ptr);

    // Notify waiting threads that a free page is available
    _m_cv.notify_one();
}

bool PagedAllocator::_owns(void *ptr) {
    bool in_range =
        ptr >= _m_pages_base &&
        ptr < static_cast<char *>(_m_pages_base) + _m_num_pages * _m_page_size;
    bool is_page_aligned = (reinterpret_cast<char *>(ptr) -
                            reinterpret_cast<char *>(_m_pages_base)) %
                               _m_page_size ==
                           0;
    return in_range && is_page_aligned;
}

bool PagedAllocator::owns(void *ptr) {
    std::lock_guard<std::mutex> lock(_m_mutex);

    return _owns(ptr);
}

} // namespace allocator
} // namespace data_engine
