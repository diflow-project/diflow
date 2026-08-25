#pragma once

#include "allocator.hpp"

#include <condition_variable>
#include <mutex>

#include "buddy_alloc.h"

namespace data_engine {
namespace allocator {

class BuddyAllocator : public Allocator {
  public:
    BuddyAllocator(size_t arena_size, void *arena_base);
    void *allocate(size_t size) override;
    void free(void *ptr) override;
    bool owns(void *ptr) override;

  private:
    void *_m_arena_base;
	void *_m_buddy_buf;
    size_t _m_arena_size;
	struct buddy *_m_buddy;

    mutable std::mutex _m_mutex;
    std::condition_variable _m_cv;

    bool _owns(void *ptr);
};

} // namespace allocator
} // namespace data_engine
