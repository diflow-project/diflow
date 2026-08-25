#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

namespace data_engine {
namespace engine {
namespace backend {
class DataEngineBackend {
  public:
    virtual torch::Tensor create_tensor(const std::vector<int64_t> &size,
                                        const pybind11::object &dtype) = 0;
    virtual void free_tensor(torch::Tensor tensor) = 0;
};
} // namespace backend
} // namespace engine
} // namespace data_engine
