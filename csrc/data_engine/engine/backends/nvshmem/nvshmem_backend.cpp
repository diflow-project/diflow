#include "nvshmem_backend.hpp"
#include "data_engine/common/debug.hpp"

#include <iostream>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

namespace {
class CudaDeviceGuard {
    int _m_prev_device_id;

  public:
    CudaDeviceGuard(int device_id) {
        cudaGetDevice(&_m_prev_device_id);
        cudaSetDevice(device_id);
    }
    ~CudaDeviceGuard() { cudaSetDevice(_m_prev_device_id); }
};
} // namespace

namespace py = pybind11;
namespace data_engine {
namespace engine {
namespace backend {
namespace nvshmem {

NvshmemDataEngineBackend::NvshmemDataEngineBackend(
    int64_t arena_size,
    // int64_t page_size, int64_t num_pages,
    // int64_t soa_buffer_size, int64_t soa_threshold,
    int64_t device_id, int64_t worker_id)
    : _m_arena_size(arena_size),
      // _m_page_size(page_size), _m_num_pages(num_pages),
      // _m_soa_buffer_size(soa_buffer_size), _m_soa_threshold(soa_threshold),
      _m_device_id(device_id), _m_worker_id(worker_id) {
    {
        CudaDeviceGuard guard(_m_device_id);
        _init_nvshmem();
        _m_arena_base = nvshmem_malloc(_m_arena_size);
        // _m_paged_shm_base = nvshmem_malloc(_m_page_size * _m_num_pages);
        // _m_soa_buffer_shm_base = nvshmem_malloc(_m_soa_buffer_size);
        _m_nvshmem_pe = nvshmem_my_pe();
    }

#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id
              << ")] Device ID: " << _m_device_id
              << ", NVSHMEM PE: " << _m_nvshmem_pe << std::endl;

    std::cout << "[NVSHMEM (worker " << _m_worker_id
              << ")] Arena SHM base: " << _m_arena_base << std::endl;
    // std::cout << "[NVSHMEM (worker " << _m_worker_id
    //           << ")] Paged SHM base: " << _m_paged_shm_base << std::endl;
    // std::cout << "[NVSHMEM (worker " << _m_worker_id
    //           << ")] SOA buffer SHM base: " << _m_soa_buffer_shm_base
    //           << std::endl;
#endif

    _m_allocator = std::make_unique<allocator::BuddyAllocator>(
        _m_arena_size, _m_arena_base);
    // _m_allocator = std::make_unique<allocator::ComplicatedAllocator>(
    //     _m_soa_threshold, _m_page_size, _m_num_pages, _m_soa_buffer_size,
    //     _m_paged_shm_base, _m_soa_buffer_shm_base);
}

NvshmemDataEngineBackend::~NvshmemDataEngineBackend() {
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id
              << ")] Destroying NVSHMEM backend" << std::endl;
#endif
    CudaDeviceGuard guard(_m_device_id);
    nvshmem_free(_m_arena_base);
    // nvshmem_free(_m_paged_shm_base);
    // nvshmem_free(_m_soa_buffer_shm_base);
    // Finalize NVSHMEM here, while MPI is still initialized. NVSHMEM >= 3.x
    // registers its own atexit cleanup which otherwise runs after mpi4py's
    // MPI_Finalize and segfaults in the MPI bootstrap; finalizing explicitly
    // makes that hook a no-op. (The engine releases this object in stop(),
    // so this runs at a well-defined point, not at interpreter-exit GC.)
    nvshmem_finalize();
}

int64_t NvshmemDataEngineBackend::nvshmem_pe() const { return _m_nvshmem_pe; }

torch::Tensor
NvshmemDataEngineBackend::create_tensor(const std::vector<int64_t> &size,
                                         const pybind11::object &dtype) {
    torch::ScalarType casted_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(torch::elementSize(casted_dtype));
    auto casted_size = torch::IntArrayRef(size);
    auto numel = std::accumulate(casted_size.begin(), casted_size.end(),
                                 int64_t(1), std::multiplies<int64_t>());
    auto size_in_bytes = element_bytes * numel;
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id << ")] Allocating "
              << size_in_bytes << " bytes" << std::endl;
#endif
    auto ptr = reinterpret_cast<uint8_t *>(_m_allocator->allocate(size_in_bytes));
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id << ")] Allocated "
              << size_in_bytes << " bytes at " << reinterpret_cast<void *>(ptr)
              << std::endl;
#endif
    return torch::from_blob(ptr, casted_size,
                            torch::TensorOptions()
                                .dtype(casted_dtype)
                                .device(torch::kCUDA, _m_device_id));
}

void NvshmemDataEngineBackend::free_tensor(torch::Tensor tensor) {
    _m_allocator->free(tensor.data_ptr());
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id << ")] Free'd tensor of "
              << "size " << tensor.numel() * tensor.element_size() << " at "
              << tensor.data_ptr() << std::endl;
#endif
}

bool NvshmemDataEngineBackend::owns_tensor(torch::Tensor tensor) {
    return _m_allocator->owns(tensor.data_ptr());
}

torch::Tensor NvshmemDataEngineBackend::fetch_tensor(
    int64_t remote_src, const std::vector<int64_t> &size,
    const pybind11::object &dtype, int64_t remote_device_id) {
    torch::ScalarType casted_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(torch::elementSize(casted_dtype));
    auto casted_size = torch::IntArrayRef(size);
    auto numel = std::accumulate(casted_size.begin(), casted_size.end(),
                                 int64_t(1), std::multiplies<int64_t>());
    if (remote_device_id == _m_device_id) {
        // TODO: no copy needed
    }
    auto size_in_bytes = element_bytes * numel;
    void *local_dst = _m_allocator->allocate(size_in_bytes);
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id << ")] Fetching tensor of "
              << "size " << size_in_bytes << " from device " << remote_device_id
              << "[" << reinterpret_cast<void *>(remote_src) << "] to device "
              << _m_device_id << "[" << reinterpret_cast<void *>(local_dst)
              << "]" << std::endl;
#endif
    {
        CudaDeviceGuard guard(_m_device_id);
        nvshmem_char_get((char *)local_dst, (char *)remote_src, size_in_bytes,
                         remote_device_id);
    }
    return torch::from_blob(local_dst, casted_size,
                            torch::TensorOptions()
                                .dtype(casted_dtype)
                                .device(torch::kCUDA, _m_device_id));
}

void NvshmemDataEngineBackend::_init_nvshmem() {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        throw std::runtime_error("MPI must be initialized before creating "
                                 "NvshmemDataEngineBackend");
    }
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id
              << ")] MPI initialized. Initializing NVSHMEM..." << std::endl;
#endif
    int nvshmem_initialized = nvshmemx_init_status();
    if (nvshmem_initialized == NVSHMEM_STATUS_NOT_INITIALIZED) {
#ifdef DATA_ENGINE_DEBUG_ENABLED
        std::cout << "[NVSHMEM (worker " << _m_worker_id
                  << ")] NVSHMEM not initialized. Initializing NVSHMEM..."
                  << std::endl;
#endif

        nvshmemx_init_attr_t attr;
        MPI_Comm comm = MPI_COMM_WORLD;
        attr.mpi_comm = &comm;

        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    }
#ifdef DATA_ENGINE_DEBUG_ENABLED
    std::cout << "[NVSHMEM (worker " << _m_worker_id
              << ")] NVSHMEM initialized. my_pe: " << nvshmem_my_pe()
              << ", n_pes: " << nvshmem_n_pes() << std::endl;
#endif
}

} // namespace nvshmem
} // namespace backend
} // namespace engine
} // namespace data_engine

using data_engine::engine::backend::nvshmem::NvshmemDataEngineBackend;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NvshmemDataEngineBackend>(m, "NvshmemDataEngineBackend")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("arena_size"), py::arg("device_id"), py::arg("worker_id"))
        // .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
        //      py::arg("page_size"), py::arg("num_pages"),
        //      py::arg("soa_buffer_size"), py::arg("soa_threshold"),
        //      py::arg("device_id"), py::arg("worker_id"))
        .def("nvshmem_pe", &NvshmemDataEngineBackend::nvshmem_pe)
        .def("create_tensor", &NvshmemDataEngineBackend::create_tensor)
        .def("free_tensor", &NvshmemDataEngineBackend::free_tensor)
        .def("fetch_tensor", &NvshmemDataEngineBackend::fetch_tensor)
        .def("owns_tensor", &NvshmemDataEngineBackend::owns_tensor);
};
