#include "small_object_allocator.hpp"

#include <chrono>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

namespace data_engine {
namespace allocator {

void SmallObjectAllocator::_check_overlap() {
#ifdef DATA_ENGINE_DEBUG_ENABLED
    // check each pair of entries
    for (auto it1 = _m_allocated_object_list.begin();
         it1 != _m_allocated_object_list.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != _m_allocated_object_list.end();
             ++it2) {
            if ((*it1)->offset < (*it2)->offset + (*it2)->size &&
                (*it1)->offset + (*it1)->size > (*it2)->offset) {
                std::cout << "[SOA] Overlap detected between " << (*it1)->offset
                          << " and " << (*it2)->offset << std::endl;
                std::cout << "[SOA] All entries: ";
                for (auto &e : _m_allocated_object_list) {
                    std::cout << "(" << e->offset << ", " << e->size << ", "
                              << e->is_freed << ") ";
                }
                std::cout << std::endl;
                throw std::runtime_error("Overlap detected");
            }
        }
    }
    std::cout << "[SOA] All entries: ";
    for (auto &e : _m_allocated_object_list) {
        std::cout << "(" << e->offset << ", " << e->size << ", " << e->is_freed
                  << ") ";
    }
    std::cout << std::endl;
#endif
}

SmallObjectAllocator::SmallObjectAllocator(size_t soa_buffer_size,
                                           void *soa_buffer_base)
    : _m_soa_buffer_base(soa_buffer_base), _m_soa_buffer_size(soa_buffer_size),
      _m_soa_buffer_offset_cur(0) {}

void *SmallObjectAllocator::allocate(size_t size) {
    std::unique_lock<std::mutex> lock(_m_mutex);

    if (size > _m_soa_buffer_size) {
        throw std::runtime_error("Size too large");
    }

    size_t new_starting_offset = _m_soa_buffer_offset_cur;
    if (new_starting_offset + size > _m_soa_buffer_size) {
        new_starting_offset = 0;
    }

    // Wait until we have enough space
    _m_cv.wait(lock, [this, size, new_starting_offset]() {
        // Clean up freed entries first
        while (_m_allocated_object_list.size() > 0) {
            auto &entry_below = _m_allocated_object_list.front();
            if (entry_below->is_freed) {
                _m_allocated_object_list.pop_front();
#ifdef DATA_ENGINE_DEBUG_ENABLED
                std::cout << "[SOA] Removing freed entry at offset "
                          << entry_below->offset << " with size "
                          << entry_below->size << std::endl;
#endif
            } else {
                break;
            }
        }

        // Check if we have enough space
        if (_m_allocated_object_list.size() == 0) {
            return true; // No entries, plenty of space
        }

        auto &entry_below = _m_allocated_object_list.front();
        bool has_entry_below = entry_below->offset >= new_starting_offset;
        if (!has_entry_below) {
            return true; // No entry below, we can allocate
        }

        if (entry_below->offset >= new_starting_offset + size) {
            return true; // Enough room
        }

        return false; // Not enough space, keep waiting
    });

    _m_soa_buffer_offset_cur = new_starting_offset + size;
    void *ptr = static_cast<char *>(_m_soa_buffer_base) + new_starting_offset;
    _add_entry(ptr, new_starting_offset, size);
#ifdef DATA_ENGINE_DEBUG_ENABLED
    _check_overlap();
#endif
    return ptr;
}

void SmallObjectAllocator::free(void *ptr) {
    std::lock_guard<std::mutex> lock(_m_mutex);

    auto entry = _get_entry(ptr);
    _free_entry(entry);
    _m_allocated_object_map.erase(ptr);

    // Notify waiting threads that space might be available
    _m_cv.notify_all();
}

bool SmallObjectAllocator::owns(void *ptr) {
    std::lock_guard<std::mutex> lock(_m_mutex);

    return _m_allocated_object_map.find(ptr) != _m_allocated_object_map.end();
}

std::shared_ptr<SmallObjectAllocator::AllocatedObjectEntry>
SmallObjectAllocator::_get_entry(void *ptr) {
    auto it = _m_allocated_object_map.find(ptr);
    if (it == _m_allocated_object_map.end()) {
        throw std::runtime_error("Pointer not found");
    }
    return it->second;
}

void SmallObjectAllocator::_add_entry(void *ptr, size_t offset, size_t size) {
    auto entry = std::make_shared<AllocatedObjectEntry>(offset, size, false);
    _m_allocated_object_list.push_back(entry);
    _m_allocated_object_map[ptr] = entry;
}

void SmallObjectAllocator::_free_entry(
    std::shared_ptr<AllocatedObjectEntry> entry) {
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[SOA] Freeing entry@" << (void *)entry.get() << " at offset "
              << entry->offset << " with size " << entry->size << std::endl;
#endif
    entry->is_freed = true;
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[SOA] Allocated object list: " << std::endl;
    for (auto &e : _m_allocated_object_list) {
        std::cout << "(" << e->offset << ", " << e->size << ", " << e->is_freed
                  << ")@" << (void *)e.get() << std::endl;
    }
    std::cout << std::endl;
#endif
}

} // namespace allocator
} // namespace data_engine
