#pragma once

#include "data_engine/common/debug.hpp"

#include "allocator.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace data_engine {
namespace allocator {

class SmallObjectAllocator : public Allocator {
  public:
    SmallObjectAllocator(size_t soa_buffer_size, void *soa_buffer_base);
    void *allocate(size_t size) override;
    void free(void *ptr) override;
    bool owns(void *ptr) override;

  private:
    struct AllocatedObjectEntry {
        size_t offset;
        size_t size;
        bool is_freed;

        AllocatedObjectEntry() : offset(0), size(0), is_freed(false) {}

        AllocatedObjectEntry(size_t offset, size_t size, bool is_freed)
            : offset(offset), size(size), is_freed(is_freed) {}
    };

    std::shared_ptr<AllocatedObjectEntry> _get_entry(void *ptr);

    void _add_entry(void *ptr, size_t offset, size_t size);
    void _free_entry(std::shared_ptr<AllocatedObjectEntry> entry);
    void _check_overlap();

    void *_m_soa_buffer_base;
    size_t _m_soa_buffer_size;
    std::deque<std::shared_ptr<AllocatedObjectEntry>> _m_allocated_object_list;
    std::unordered_map<void *, std::shared_ptr<AllocatedObjectEntry>>
        _m_allocated_object_map;
    size_t _m_soa_buffer_offset_cur;

    mutable std::mutex _m_mutex;
    std::condition_variable _m_cv;
};

} // namespace allocator
} // namespace data_engine
