#include "buddy_allocator.hpp"

#include <stdexcept>
#include <mutex>
#include <condition_variable>

#ifndef BUDDY_ALLOC_IMPLEMENTATION
#define BUDDY_ALLOC_IMPLEMENTATION
#endif
#include "buddy_alloc.h"
#undef BUDDY_ALLOC_IMPLEMENTATION

namespace data_engine {
namespace allocator {

BuddyAllocator::BuddyAllocator(size_t arena_size, void *arena_base)
    : _m_arena_base(arena_base), _m_arena_size(arena_size) {
    std::lock_guard<std::mutex> lock(_m_mutex);
	_m_buddy_buf = (unsigned char *) malloc(buddy_sizeof(arena_size));
    _m_buddy = buddy_init(
		reinterpret_cast<unsigned char *>(_m_buddy_buf),
		reinterpret_cast<unsigned char *>(_m_arena_base),
		_m_arena_size);
}

void *BuddyAllocator::allocate(size_t size) {
    std::unique_lock<std::mutex> lock(_m_mutex);

    if (size > _m_arena_size) {
        throw std::runtime_error("Size too large");
    }

    // Wait until a free page becomes available
	void *ptr;
    _m_cv.wait(lock, [this, &ptr, size]() {
        ptr = buddy_malloc(_m_buddy, size);
        return ptr != nullptr;
    });

	return ptr;
}

void BuddyAllocator::free(void *ptr) {
    std::lock_guard<std::mutex> lock(_m_mutex);

    if (!_owns(ptr)) {
        throw std::runtime_error("Pointer not owned by allocator");
    }
    buddy_free(_m_buddy, ptr);

    // Notify waiting threads that a free page is available
    _m_cv.notify_one();
}

bool BuddyAllocator::_owns(void *ptr) {
    bool in_range =
        ptr >= _m_arena_base &&
        ptr < static_cast<char *>(_m_arena_base) + _m_arena_size;

    return in_range;
}

bool BuddyAllocator::owns(void *ptr) {
    std::lock_guard<std::mutex> lock(_m_mutex);

    return _owns(ptr);
}

} // namespace allocator
} // namespace data_engine
