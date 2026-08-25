#pragma once

#include <cstddef>

namespace data_engine {
namespace allocator {

class Allocator {
  public:
    virtual void *allocate(size_t size) = 0;
    virtual void free(void *ptr) = 0;
    virtual bool owns(void *ptr) = 0;
};

} // namespace allocator
} // namespace data_engine
