#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct TransferProfile {
  std::vector<long long> block_sizes;
  std::vector<double> fetch_overheads_us;
};

struct ModelProfile {
  double loading_latency = std::numeric_limits<double>::infinity();
  std::unordered_map<std::string, double> execution_latencies;
};

struct Cost {
  double queue = 0.0;
  double transfer = 0.0;
  double loading = 0.0;
  double execution = 0.0;
  double total = 0.0;

  double reserved_latency() const {
    return transfer + loading + execution;
  }
};

struct Reservation {
  std::size_t worker_index = 0;
  Cost cost;
};

long long next_power_of_two(long long value) {
  if (value < 1) {
    throw std::invalid_argument("batch size must be positive");
  }
  long long result = 1;
  while (result < value) {
    if (result > std::numeric_limits<long long>::max() / 2) {
      throw std::overflow_error("batch size is too large");
    }
    result *= 2;
  }
  return result;
}

std::string execution_prefix(const std::string& mode, long long batch_size) {
  return mode + '\x1f' + std::to_string(batch_size) + '\x1f';
}

std::string execution_key(
    const std::string& mode,
    long long batch_size,
    long long height,
    long long width) {
  return execution_prefix(mode, batch_size) + std::to_string(height) + '\x1f' +
      std::to_string(width);
}

double lookup_execution_latency(
    const ModelProfile& profile,
    const std::string& mode,
    long long batch_size,
    long long height,
    long long width) {
  if (height > 0 && width > 0) {
    const auto exact = profile.execution_latencies.find(
        execution_key(mode, batch_size, height, width));
    if (exact != profile.execution_latencies.end()) {
      return exact->second;
    }
  }
  const auto legacy = profile.execution_latencies.find(
      execution_key(mode, batch_size, 0, 0));
  if (legacy != profile.execution_latencies.end()) {
    return legacy->second;
  }

  const std::string prefix = execution_prefix(mode, batch_size);
  double best_distance = std::numeric_limits<double>::infinity();
  double best_latency = std::numeric_limits<double>::infinity();
  for (const auto& entry : profile.execution_latencies) {
    if (entry.first.rfind(prefix, 0) != 0) {
      continue;
    }
    const std::string dimensions = entry.first.substr(prefix.size());
    const std::size_t separator = dimensions.find('\x1f');
    if (separator == std::string::npos) {
      continue;
    }
    const long long candidate_height = std::stoll(dimensions.substr(0, separator));
    const long long candidate_width = std::stoll(dimensions.substr(separator + 1));
    if (candidate_height <= 0 || candidate_width <= 0) {
      continue;
    }
    const double distance = std::abs(static_cast<double>(candidate_height - height)) +
        std::abs(static_cast<double>(candidate_width - width));
    if (distance < best_distance ||
        (distance == best_distance && entry.second < best_latency)) {
      best_distance = distance;
      best_latency = entry.second;
    }
  }
  return best_latency;
}

void validate_profile(const TransferProfile& profile, const char* name) {
  if (profile.block_sizes.size() != profile.fetch_overheads_us.size()) {
    throw std::invalid_argument(
        std::string(name) + " block sizes and overheads must have equal length");
  }
  if (!std::is_sorted(profile.block_sizes.begin(), profile.block_sizes.end())) {
    throw std::invalid_argument(
        std::string(name) + " block sizes must be sorted");
  }
}

double lookup_transfer_latency(
    long long size_bytes, const TransferProfile& profile) {
  if (profile.block_sizes.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  const auto iter = std::lower_bound(
      profile.block_sizes.begin(), profile.block_sizes.end(), size_bytes);
  const std::size_t index =
      iter == profile.block_sizes.end()
          ? profile.block_sizes.size() - 1
          : static_cast<std::size_t>(iter - profile.block_sizes.begin());
  return profile.fetch_overheads_us[index] / 1e6;
}

py::dict cost_to_dict(const Cost& cost) {
  py::dict result;
  result["queue"] = cost.queue;
  result["transfer"] = cost.transfer;
  result["loading"] = cost.loading;
  result["execution"] = cost.execution;
  result["total"] = cost.total;
  return result;
}

class SchedulingCore {
 public:
  SchedulingCore(
      std::vector<int> worker_ranks,
      std::vector<int> worker_host_ids,
      const std::vector<std::vector<std::string>>& active_models,
      const std::vector<std::string>& model_names,
      const std::vector<double>& loading_latencies,
      const std::vector<std::string>& execution_model_names,
      const std::vector<std::string>& execution_modes,
      const std::vector<long long>& execution_batch_sizes,
      const std::vector<long long>& execution_heights,
      const std::vector<long long>& execution_widths,
      const std::vector<double>& execution_latencies,
      std::vector<long long> intra_block_sizes,
      std::vector<double> intra_fetch_overheads_us,
      std::vector<long long> inter_block_sizes,
      std::vector<double> inter_fetch_overheads_us,
      double worker_latency_threshold)
      : worker_ranks_(std::move(worker_ranks)),
        worker_host_ids_(std::move(worker_host_ids)),
        intra_profile_{
            std::move(intra_block_sizes),
            std::move(intra_fetch_overheads_us)},
        inter_profile_{
            std::move(inter_block_sizes),
            std::move(inter_fetch_overheads_us)},
        worker_latency_threshold_(worker_latency_threshold) {
    if (worker_ranks_.empty()) {
      throw std::invalid_argument("at least one worker is required");
    }
    if (worker_ranks_.size() != worker_host_ids_.size() ||
        worker_ranks_.size() != active_models.size()) {
      throw std::invalid_argument(
          "worker ranks, host ids, and active model lists must have equal length");
    }
    if (model_names.size() != loading_latencies.size()) {
      throw std::invalid_argument(
          "model names and loading latencies must have equal length");
    }
    if (execution_model_names.size() != execution_modes.size() ||
        execution_model_names.size() != execution_batch_sizes.size() ||
        execution_model_names.size() != execution_heights.size() ||
        execution_model_names.size() != execution_widths.size() ||
        execution_model_names.size() != execution_latencies.size()) {
      throw std::invalid_argument("execution profile arrays must have equal length");
    }
    validate_profile(intra_profile_, "intra-node transfer profile");
    validate_profile(inter_profile_, "inter-node transfer profile");

    active_models_.reserve(worker_ranks_.size());
    queue_latencies_.assign(worker_ranks_.size(), 0.0);
    for (std::size_t index = 0; index < worker_ranks_.size(); ++index) {
      const int worker_rank = worker_ranks_[index];
      if (!worker_indexes_.emplace(worker_rank, index).second) {
        throw std::invalid_argument("worker ranks must be unique");
      }
      active_models_.emplace_back(
          active_models[index].begin(), active_models[index].end());
    }

    for (std::size_t index = 0; index < model_names.size(); ++index) {
      model_profiles_[model_names[index]].loading_latency =
          loading_latencies[index];
    }
    for (std::size_t index = 0; index < execution_model_names.size(); ++index) {
      model_profiles_[execution_model_names[index]]
          .execution_latencies[execution_key(
              execution_modes[index], execution_batch_sizes[index],
              execution_heights[index], execution_widths[index])] =
          execution_latencies[index];
    }
  }

  py::object select_and_reserve(
      const std::string& task_id,
      const std::string& model_name,
      const std::string& mode,
      long long batch_size,
      long long height,
      long long width,
      bool uses_model_profile,
      const std::vector<long long>& tensor_offsets,
      const std::vector<int>& source_worker_ranks,
      const std::vector<int>& source_host_ids,
      const std::vector<long long>& source_sizes_bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    validate_task_arrays(
        tensor_offsets,
        source_worker_ranks,
        source_host_ids,
        source_sizes_bytes);

    const auto existing = reservations_.find(task_id);
    if (existing != reservations_.end()) {
      return reservation_to_dict(existing->second);
    }

    std::size_t best_index = worker_ranks_.size();
    Cost best_cost;
    best_cost.total = std::numeric_limits<double>::infinity();
    for (std::size_t worker_index = 0; worker_index < worker_ranks_.size();
         ++worker_index) {
      if (queue_latencies_[worker_index] > worker_latency_threshold_) {
        continue;
      }
      const Cost cost = calculate_cost(
          worker_index,
          model_name,
          mode,
          batch_size,
          height,
          width,
          uses_model_profile,
          tensor_offsets,
          source_worker_ranks,
          source_host_ids,
          source_sizes_bytes);
      if (cost.total < best_cost.total) {
        best_index = worker_index;
        best_cost = cost;
      }
    }

    if (best_index == worker_ranks_.size() || std::isinf(best_cost.total)) {
      return py::none();
    }
    return record_reservation(task_id, best_index, best_cost);
  }

  py::dict reserve_on_worker(
      const std::string& task_id,
      const std::string& model_name,
      const std::string& mode,
      long long batch_size,
      long long height,
      long long width,
      bool uses_model_profile,
      const std::vector<long long>& tensor_offsets,
      const std::vector<int>& source_worker_ranks,
      const std::vector<int>& source_host_ids,
      const std::vector<long long>& source_sizes_bytes,
      int worker_rank) {
    std::lock_guard<std::mutex> guard(mutex_);
    validate_task_arrays(
        tensor_offsets,
        source_worker_ranks,
        source_host_ids,
        source_sizes_bytes);
    const auto worker_iter = worker_indexes_.find(worker_rank);
    if (worker_iter == worker_indexes_.end()) {
      throw std::invalid_argument("unknown worker rank: " + std::to_string(worker_rank));
    }

    const auto existing = reservations_.find(task_id);
    if (existing != reservations_.end()) {
      if (existing->second.worker_index != worker_iter->second) {
        throw std::invalid_argument(
            "task " + task_id + " is already reserved on another worker");
      }
      return reservation_to_dict(existing->second);
    }

    const Cost cost = calculate_cost(
        worker_iter->second,
        model_name,
        mode,
        batch_size,
        height,
        width,
        uses_model_profile,
        tensor_offsets,
        source_worker_ranks,
        source_host_ids,
        source_sizes_bytes);
    return record_reservation(task_id, worker_iter->second, cost);
  }

  bool complete(const std::string& task_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto reservation_iter = reservations_.find(task_id);
    if (reservation_iter == reservations_.end()) {
      return false;
    }
    const Reservation reservation = reservation_iter->second;
    reservations_.erase(reservation_iter);

    const double reserved_latency = reservation.cost.reserved_latency();
    if (std::isinf(reserved_latency)) {
      double queue_latency = 0.0;
      for (const auto& entry : reservations_) {
        if (entry.second.worker_index == reservation.worker_index) {
          queue_latency += entry.second.cost.reserved_latency();
        }
      }
      queue_latencies_[reservation.worker_index] = queue_latency;
    } else {
      queue_latencies_[reservation.worker_index] = std::max(
          0.0,
          queue_latencies_[reservation.worker_index] - reserved_latency);
    }
    return true;
  }

  void update_active_models(
      int worker_rank, const std::vector<std::string>& active_models) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto worker_iter = worker_indexes_.find(worker_rank);
    if (worker_iter == worker_indexes_.end()) {
      throw std::invalid_argument("unknown worker rank: " + std::to_string(worker_rank));
    }
    active_models_[worker_iter->second] =
        std::unordered_set<std::string>(active_models.begin(), active_models.end());
  }

  std::size_t available_worker_count() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<std::size_t>(std::count_if(
        queue_latencies_.begin(),
        queue_latencies_.end(),
        [&](double latency) { return latency <= worker_latency_threshold_; }));
  }

  py::dict snapshot() const {
    std::lock_guard<std::mutex> guard(mutex_);
    py::dict result;
    for (std::size_t worker_index = 0; worker_index < worker_ranks_.size();
         ++worker_index) {
      py::dict reservations;
      for (const auto& entry : reservations_) {
        if (entry.second.worker_index == worker_index) {
          reservations[py::str(entry.first)] =
              entry.second.cost.reserved_latency();
        }
      }
      py::dict worker_snapshot;
      worker_snapshot["queue_latency"] = queue_latencies_[worker_index];
      worker_snapshot["reservations"] = std::move(reservations);
      result[py::int_(worker_ranks_[worker_index])] = std::move(worker_snapshot);
    }
    return result;
  }

 private:
  static void validate_task_arrays(
      const std::vector<long long>& tensor_offsets,
      const std::vector<int>& source_worker_ranks,
      const std::vector<int>& source_host_ids,
      const std::vector<long long>& source_sizes_bytes) {
    if (source_worker_ranks.size() != source_host_ids.size() ||
        source_worker_ranks.size() != source_sizes_bytes.size()) {
      throw std::invalid_argument("tensor source arrays must have equal length");
    }
    if (tensor_offsets.empty() || tensor_offsets.front() != 0 ||
        tensor_offsets.back() !=
            static_cast<long long>(source_worker_ranks.size())) {
      throw std::invalid_argument(
          "tensor offsets must start at zero and end at the source count");
    }
    if (!std::is_sorted(tensor_offsets.begin(), tensor_offsets.end())) {
      throw std::invalid_argument("tensor offsets must be non-decreasing");
    }
  }

  double transfer_latency(
      std::size_t worker_index,
      const std::vector<long long>& tensor_offsets,
      const std::vector<int>& source_worker_ranks,
      const std::vector<int>& source_host_ids,
      const std::vector<long long>& source_sizes_bytes) const {
    const int dst_worker_rank = worker_ranks_[worker_index];
    const int dst_host_id = worker_host_ids_[worker_index];
    double latency = 0.0;
    for (std::size_t tensor_index = 0; tensor_index + 1 < tensor_offsets.size();
         ++tensor_index) {
      const std::size_t begin =
          static_cast<std::size_t>(tensor_offsets[tensor_index]);
      const std::size_t end =
          static_cast<std::size_t>(tensor_offsets[tensor_index + 1]);
      if (begin == end) {
        continue;
      }

      std::size_t selected_source = begin;
      for (std::size_t source_index = begin; source_index < end; ++source_index) {
        if (source_host_ids[source_index] == dst_host_id) {
          selected_source = source_index;
          break;
        }
      }
      if (source_worker_ranks[selected_source] == dst_worker_rank) {
        continue;
      }

      const bool intra_node = source_host_ids[selected_source] == dst_host_id;
      latency += lookup_transfer_latency(
          source_sizes_bytes[selected_source],
          intra_node ? intra_profile_ : inter_profile_);
    }
    return latency;
  }

  Cost calculate_cost(
      std::size_t worker_index,
      const std::string& model_name,
      const std::string& mode,
      long long batch_size,
      long long height,
      long long width,
      bool uses_model_profile,
      const std::vector<long long>& tensor_offsets,
      const std::vector<int>& source_worker_ranks,
      const std::vector<int>& source_host_ids,
      const std::vector<long long>& source_sizes_bytes) const {
    Cost cost;
    cost.queue = queue_latencies_[worker_index];
    cost.transfer = transfer_latency(
        worker_index,
        tensor_offsets,
        source_worker_ranks,
        source_host_ids,
        source_sizes_bytes);

    if (uses_model_profile) {
      const auto profile_iter = model_profiles_.find(model_name);
      if (profile_iter == model_profiles_.end()) {
        cost.loading = std::numeric_limits<double>::infinity();
        cost.execution = std::numeric_limits<double>::infinity();
      } else {
        if (active_models_[worker_index].count(model_name) == 0) {
          cost.loading = profile_iter->second.loading_latency;
        }
        cost.execution = lookup_execution_latency(
            profile_iter->second,
            mode,
            next_power_of_two(batch_size),
            height,
            width);
      }
    }
    cost.total = cost.queue + cost.reserved_latency();
    return cost;
  }

  py::dict reservation_to_dict(const Reservation& reservation) const {
    py::dict result;
    result["worker_rank"] = worker_ranks_[reservation.worker_index];
    result["cost"] = cost_to_dict(reservation.cost);
    return result;
  }

  py::dict record_reservation(
      const std::string& task_id, std::size_t worker_index, const Cost& cost) {
    Reservation reservation{worker_index, cost};
    reservations_[task_id] = reservation;
    queue_latencies_[worker_index] += cost.reserved_latency();
    return reservation_to_dict(reservation);
  }

  std::vector<int> worker_ranks_;
  std::vector<int> worker_host_ids_;
  std::unordered_map<int, std::size_t> worker_indexes_;
  std::vector<std::unordered_set<std::string>> active_models_;
  std::unordered_map<std::string, ModelProfile> model_profiles_;
  TransferProfile intra_profile_;
  TransferProfile inter_profile_;
  double worker_latency_threshold_;
  std::vector<double> queue_latencies_;
  std::unordered_map<std::string, Reservation> reservations_;
  mutable std::mutex mutex_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Stateful C++ scheduling core for DiFlow";
  py::class_<SchedulingCore>(module, "SchedulingCore")
      .def(py::init<
           std::vector<int>,
           std::vector<int>,
           const std::vector<std::vector<std::string>>&,
           const std::vector<std::string>&,
           const std::vector<double>&,
           const std::vector<std::string>&,
           const std::vector<std::string>&,
           const std::vector<long long>&,
           const std::vector<long long>&,
           const std::vector<long long>&,
           const std::vector<double>&,
           std::vector<long long>,
           std::vector<double>,
           std::vector<long long>,
           std::vector<double>,
           double>())
      .def("select_and_reserve", &SchedulingCore::select_and_reserve)
      .def("reserve_on_worker", &SchedulingCore::reserve_on_worker)
      .def("complete", &SchedulingCore::complete)
      .def("update_active_models", &SchedulingCore::update_active_models)
      .def("available_worker_count", &SchedulingCore::available_worker_count)
      .def("snapshot", &SchedulingCore::snapshot);
}
