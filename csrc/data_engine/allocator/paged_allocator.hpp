#pragma once

#include "allocator.hpp"

#include <condition_variable>
#include <mutex>
#include <unordered_set>

namespace data_engine {
namespace allocator {

class PagedAllocator : public Allocator {
  public:
    PagedAllocator(size_t page_size, size_t num_pages, void *pages_base);
    void *allocate(size_t size) override;
    void free(void *ptr) override;
    bool owns(void *ptr) override;

  private:
    void *_m_pages_base;
    size_t _m_page_size;
    size_t _m_num_pages;
    std::unordered_set<void *> _m_free_pages;

    mutable std::mutex _m_mutex;
    std::condition_variable _m_cv;

    bool _owns(void *ptr);
};

} // namespace allocator
} // namespace data_engine
