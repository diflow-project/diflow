from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import inspect
import io
import logging
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
import zmq
from mpi4py import MPI
from PIL import Image

from diflow.backend.data_engine.engine.nvshmem_data_engine import (
    FetchingTask,
    FreeingTask,
    NvshmemDataEngine,
)
from diflow.interface.node_io import string_to_type, type_to_string
from diflow.interface.workflow import WorkflowNode
from diflow.operators.execution_modes import PATCH_OFF, PATCH_ON
from diflow.operators.models.patches.base_patch import BasePatch
from diflow.operators.schedulers.base_scheduler import BaseScheduler
from diflow.operators.utils import get_op

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class ProcessTaskThread(threading.Thread):
    def __init__(self, worker: DistributedWorker):
        super().__init__()
        self.worker = worker
        self.logger = worker.logger

    def run(self):
        # Load models from YAML config file
        prefetch_models = self.worker._load_prefetch_models_config(
            self.worker.prefetch_models_config
        )
        if prefetch_models:
            self.worker.prefetch_models(models=prefetch_models, device="cpu")

        # Signal that prefetching is complete
        self.worker.prefetching_complete_event.set()
        self.logger.info(
            f"Worker {self.worker.global_rank} initialized with {len(self.worker.cpu_models)} models prefetched to CPU memory"
        )

        while self.worker.running or self.worker.task_queue.qsize() > 0:
            try:
                task = self.worker.task_queue.get(block=True, timeout=None)
                self.worker.process_task(task)
            except Empty:
                pass


class DistributedWorker:
    def __init__(
        self,
        local_rank: int,
        global_rank: int,
        hostname: str,
        prefetch_models_config: str,
        base_port: int = 14000,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.hostname = hostname
        self.base_port = base_port
        self.prefetch_models_config = prefetch_models_config

        # ZMQ setup
        self.context = zmq.Context()
        self.task_socket = self.context.socket(zmq.PULL)
        self.result_socket = self.context.socket(zmq.PUSH)

        # Setup logging
        self._setup_logging()

        # Ports for this worker
        self.task_port = self._get_task_port()
        self.result_port = self._get_result_port()

        assert torch.cuda.is_available()
        self.device = f"cuda:{self.local_rank}"
        self.max_gpu_memory_fraction = 0.85
        self.nvshmem_memory = 8 * 1024 * 1024 * 1024
        self.total_gpu_memory = torch.cuda.get_device_properties(
            self.local_rank
        ).total_memory
        self.available_gpu_memory = (
            self.total_gpu_memory * self.max_gpu_memory_fraction - self.nvshmem_memory
        )
        self.logger.info(
            f"Available GPU memory: {self.available_gpu_memory / (1024**3):.2f} GiB, NVSHMEM memory: {self.nvshmem_memory / (1024**3):.2f} GiB, Total GPU memory: {self.total_gpu_memory / (1024**3):.2f} GiB, Max GPU memory fraction: {self.max_gpu_memory_fraction}"
        )

        # Setup NVSHMEM
        self.data_engine = NvshmemDataEngine(
            arena_size=self.nvshmem_memory,
            device_id=self.local_rank,
            worker_id=self.global_rank,
        )

        # Model management for LRU caching
        self.cpu_models: Dict[str, Any] = {}  # Models stored in CPU memory
        self.gpu_models: OrderedDict[str, Any] = (
            OrderedDict()
        )  # Models in GPU memory (LRU ordered)
        self.model_memory_usage: Dict[str, int] = (
            {}
        )  # Model ID -> GPU memory usage in bytes
        self.model_access_count: Dict[str, int] = {}  # Model ID -> access count for LRU
        self.request_schedulers: Dict[str, BaseScheduler] = {}

        # Patch management
        self.patches: Dict[str, Any] = {}  # Patch ID -> patch instance
        # Track which models have had patches applied
        # Format: model_id -> set of patch IDs that have been applied
        self.model_patch_status: Dict[str, set] = {}

        # Maps request_id -> NodeIO name -> tensor
        self.tensor_map: Dict[str, Dict[str, torch.Tensor]] = {}

        self.task_queue = Queue()
        self.process_loop_task = ProcessTaskThread(self)

        # Prefetching completion tracking
        self.prefetching_complete_event = asyncio.Event()

        self.running = True

    def _setup_logging(self):
        """Setup logging for this worker"""
        log_dir = os.environ.get("DIFLOW_LOG_DIR", "logs")

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/worker_{self.global_rank}_{timestamp}.log"

        log_level_str = os.environ.get("LOGLEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Setup the logger
        self.logger = logging.getLogger(f"worker_{self.global_rank}")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler that uses sys.stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Configure the root logger to use the same handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.logger.info(
            f"Initialized logging for worker {self.global_rank}, log file: {log_file}"
        )

    def _load_prefetch_models_config(
        self, prefetch_models_config: str
    ) -> Optional[List[Tuple[str, str]]]:
        """Load prefetch models configuration from YAML file"""
        try:
            with open(prefetch_models_config, "r") as file:
                config = yaml.safe_load(file)
                prefetch_models = config.get("prefetch_models", [])

                if prefetch_models is None or len(prefetch_models) == 0:
                    self.logger.warning("No prefetch models found in config")
                    return None

                # Convert the config format to the expected format for prefetch_models
                models = [
                    (model["model_id"], model["model_path"])
                    for model in prefetch_models
                ]

                self.logger.info(f"Loaded {len(models)} prefetch models from config")
                return models
        except FileNotFoundError:
            self.logger.warning(
                f"Prefetch models config file not found: {prefetch_models_config}"
            )
            return None
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing prefetch models config file: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading prefetch models config: {e}")
            return None

    def _get_task_port(self) -> int:
        """Calculate unique task port for this worker"""
        return self.base_port + self.global_rank * 2

    def _get_result_port(self) -> int:
        """Calculate unique result port for this worker"""
        return self._get_task_port() + 1

    def setup(self):
        # Setup ZMQ connection
        self._setup_zmq_connection()

        # Set GPU device for this worker
        torch.cuda.set_device(self.local_rank)

    def _setup_zmq_connection(self):
        self.logger.info(
            f"Worker {self.global_rank} => Task Port: {self.task_port}, Result Port: {self.result_port}"
        )

        self.task_socket.bind(f"tcp://*:{self.task_port}")
        self.result_socket.bind(f"tcp://*:{self.result_port}")

    def _deserialize_inputs(
        self, inputs: Dict[str, Any], workflow_node: WorkflowNode
    ) -> Dict[str, Any]:
        """Deserialize inputs just before model execution"""
        # print(
        #     f"Begin to deserialize inputs: {inputs}, workflow_node: {workflow_node.name}"
        # )
        deserialized = {}
        for name, value in inputs.items():
            # Get input type from the node's input info
            input_info = workflow_node.get_inputs().get(name)
            if (
                input_info
                and input_info.data_type == Image.Image
                and isinstance(value, str)
            ):
                # Convert base64 string to PIL Image
                try:
                    img_data = base64.b64decode(value)
                    deserialized[name] = Image.open(io.BytesIO(img_data)).convert("RGB")
                except Exception as e:
                    raise ValueError(
                        f"Failed to decode image for input {name}: {str(e)}"
                    )
            else:
                deserialized[name] = value
        return deserialized

    def _remove_unnecessary_patches(
        self,
        model_id: str,
        needed_patch_ids: set[str],
        model_components: Dict[str, Any],
    ):
        """Remove patches that are applied to the model but are not needed."""
        # Initialize patch status tracking for this model if not exists
        if model_id not in self.model_patch_status:
            self.model_patch_status[model_id] = set()

        # Get currently applied patches
        currently_applied_patches = self.model_patch_status[model_id].copy()

        # Find patches that are applied but not needed
        patches_to_remove = currently_applied_patches - needed_patch_ids

        if not patches_to_remove:
            return

        self.logger.debug(
            f"Removing unnecessary patches {patches_to_remove} from model {model_id}"
        )

        for patch_id in patches_to_remove:
            if patch_id not in self.patches:
                continue

            patch_instance = self.patches[patch_id]
            patch_op = patch_instance["patch"]
            patch_components = patch_instance["patch_components"]

            self.logger.debug(
                f"Removing patch {patch_id} from model {model_id} on {self.device}"
            )

            # Execute patch in "patch_off" mode
            patch_op.execute(
                model_components=patch_components,
                device=self.device,
                mode=PATCH_OFF,
                target_model_id=model_id,
                target_model_components=model_components,
            )

            # Remove patch from applied status
            self.model_patch_status[model_id].discard(patch_id)

    def _apply_necessary_patches(
        self,
        model_id: str,
        patches: List[BasePatch],
        model_components: Dict[str, Any],
    ):
        """Apply patches that are needed but not yet applied to the model."""
        # Initialize patch status tracking for this model if not exists
        if model_id not in self.model_patch_status:
            self.model_patch_status[model_id] = set()

        # Apply each patch that hasn't been applied yet
        for patch in patches:
            patch_id = patch.id

            # Skip if patch already applied
            if patch_id in self.model_patch_status[model_id]:
                continue

            # Load patch if not already loaded
            if patch_id not in self.patches:
                patch_path = (
                    patch.config.model_path if patch.config is not None else None
                )
                patch_op = get_op(patch_id)
                if patch_path is not None:
                    self.logger.debug(
                        f"Initializing patch {patch_id} from {patch_path} on {self.device}"
                    )
                    patch_components = patch_op.initialize(patch_path, self.device)
                    self.patches[patch_id] = {
                        "patch": patch_op,
                        "patch_components": patch_components,
                    }
                else:
                    self.patches[patch_id] = {
                        "patch": patch_op,
                        "patch_components": None,
                    }

            # Apply patch to the model
            patch_instance = self.patches[patch_id]
            patch_op = patch_instance["patch"]
            patch_components = patch_instance["patch_components"]

            self.logger.debug(
                f"Applying patch {patch_id} to model {model_id} on {self.device}"
            )

            # Execute patch in "patch_on" mode
            patch_op.execute(
                model_components=patch_components,
                device=self.device,
                mode=PATCH_ON,
                target_model_id=model_id,
                target_model_components=model_components,
            )

            # Mark patch as applied
            self.model_patch_status[model_id].add(patch_id)

    def process_task(self, task: Dict[str, Any]):
        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            start_time = time.time()

        # Extract task information
        data_list = task["data"]
        batch_size = len(data_list)
        task_group = [(data["node_name"], data["request_id"]) for data in data_list]
        self.logger.debug(
            f"Started to process task group {', '.join([f'({node_name}, {request_id})' for node_name, request_id in task_group])}"
        )

        ###################################################
        # Phase 0: Free tensors that are no longer needed #
        ###################################################
        free_tensor_dict = task.get("free_tensor_dict", None)
        if free_tensor_dict is not None:
            for request_id, tensor_list in free_tensor_dict.items():
                for tensor_id in tensor_list:
                    if tensor_id in self.tensor_map[request_id]:
                        self.free_tensor(self.tensor_map[request_id][tensor_id])
                        # self.logger.debug(
                        #     f"Freed tensor {tensor_id} for request {request_id}"
                        # )
                        del self.tensor_map[request_id][tensor_id]
        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            free_tensor_end_time = time.time()
            self.logger.debug(
                f"Time taken to free tensors for task group {', '.join([f'({node_name}, {request_id})' for node_name, request_id in task_group])}: {free_tensor_end_time - start_time:.4f} seconds"
            )

        ########################################################
        # Phase 1: Collect all inputs and prepare for batching #
        ########################################################
        batched_infos = []
        for data in data_list:
            request_id = data["request_id"]
            workflow_node = WorkflowNode.from_dict(data["workflow_node"])
            self.logger.debug(
                f"Processing task {workflow_node.name} for request {request_id}"
            )

            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                parse_inputs_start_time = time.time()

            # map input_name -> input_info.name
            input_map = data.get("input_map", {})
            # map output_name -> output_info.name
            output_map = data.get("output_map", {})
            # map input_name -> input_value
            required_inputs = data["node_inputs"]
            # map input_name -> (worker_id, tensor_info: {ptr, size, dtype})
            input_locations = data.get("node_input_locations", {})
            # list of output_name
            required_outputs = data.get("required_outputs", [])
            # map input_name -> input_info.name for lazy inputs
            lazy_inputs = data.get("lazy_inputs", {})

            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                parse_inputs_end_time = time.time()
                self.logger.debug(
                    f"Time taken to parse inputs from the coordinator ({workflow_node.name}, {request_id}): {parse_inputs_end_time - parse_inputs_start_time:.4f} seconds"
                )

            # self.logger.debug(f"input_map: {input_map}")
            # self.logger.debug(f"output_map: {output_map}")
            # self.logger.debug(f"required_inputs: {required_inputs}")
            # self.logger.debug(f"input_locations: {input_locations}")
            # self.logger.debug(f"required_outputs: {required_outputs}")
            # self.logger.debug(f"lazy_inputs: {lazy_inputs}")
            # for cur_request_id, tensor_map in self.tensor_map.items():
            #     self.logger.debug(f"keys in tensor_map for request {cur_request_id}: {tensor_map.keys()}")

            # Construct node inputs
            node_inputs = {}
            tensor_infos = {}
            for input_name, input_info in workflow_node.get_inputs().items():
                if self.logger.level == logging.DEBUG:
                    torch.cuda.synchronize(self.local_rank)
                    construct_node_input_start_time = time.time()

                if input_name in input_locations:
                    location_dict = input_locations[
                        input_name
                    ]  # map worker_rank -> tensor_info
                    if (
                        str(self.global_rank) not in location_dict
                    ):  # the rank in location_dict is a string
                        remote_worker_rank = int(
                            list(location_dict.keys())[0]
                        )  # TODO @ Lingyun: what if there are multiple workers?
                        remote_tensor_info = location_dict[str(remote_worker_rank)]
                        if input_info is None:
                            raise ValueError(
                                f"Input {input_name} is None, should not happen"
                            )
                        # self.logger.debug(
                        #     f"Fetching tensor {input_name} ({input_info.name}) from worker {remote_worker_rank}, ptr: {remote_tensor_info['ptr']}, size: {remote_tensor_info['size']}, dtype: {remote_tensor_info['dtype']}"
                        # )
                        # Retrieve tensor from remote worker
                        tensor = self.fetch_tensor(
                            remote_ptr=remote_tensor_info["ptr"],
                            size=remote_tensor_info["size"],
                            dtype=string_to_type(remote_tensor_info["dtype"]),
                            remote_nvshmem_pe=remote_worker_rank,
                            tensor_id=input_info.name,
                        )
                        node_inputs[input_name] = tensor
                        if request_id not in self.tensor_map:
                            self.tensor_map[request_id] = {}
                        self.tensor_map[request_id][input_info.name] = tensor
                        tensor_infos[input_info.name] = {
                            "ptr": tensor.data_ptr(),
                            "size": tensor.size(),
                            "dtype": type_to_string(tensor.dtype),
                        }
                    else:
                        # Input is in local storage
                        if input_info.name not in self.tensor_map[request_id]:
                            self.logger.debug(
                                f"input_name: {input_name}, input_info.name: {input_info.name}"
                            )
                            raise ValueError(
                                f"Input {input_info.name} is not found in tensor_map for request {request_id}"
                            )
                        node_inputs[input_name] = self.tensor_map[request_id][
                            input_info.name
                        ]
                elif input_name in lazy_inputs:
                    # Lazy input - create a wrapper function for lazy acquisition
                    lazy_tensor_name = lazy_inputs[input_name]
                    self.logger.debug(
                        f"Creating lazy acquisition function for input {input_name} ({lazy_tensor_name})"
                    )

                    # Create a wrapper function that will acquire the tensor when called
                    def create_lazy_acquisition_func(tensor_id, input_name, request_id):
                        def lazy_acquisition_func():
                            self.logger.debug(
                                f"Acquiring lazy input {input_name} ({tensor_id})"
                            )
                            if tensor_id not in self.tensor_map[request_id]:
                                self.logger.debug(
                                    f"Tensor {tensor_id} is not found in tensor_map for request {request_id}, fetching from data engine"
                                )
                                tensor = self.data_engine.get(tensor_id=tensor_id)

                                # Store the tensor in tensor_map for future use
                                if request_id not in self.tensor_map:
                                    self.tensor_map[request_id] = {}
                                self.tensor_map[request_id][tensor_id] = tensor

                            return self.tensor_map[request_id][tensor_id]

                        return lazy_acquisition_func

                    node_inputs[input_name] = create_lazy_acquisition_func(
                        lazy_tensor_name, input_name, request_id
                    )
                else:
                    # Input is from workflow input
                    if input_name not in required_inputs:
                        raise ValueError(
                            f"Input {input_name} is not found in required_inputs"
                        )
                    node_inputs[input_name] = required_inputs[input_name]

                if self.logger.level == logging.DEBUG:
                    torch.cuda.synchronize(self.local_rank)
                    construct_node_input_end_time = time.time()
                    self.logger.debug(
                        f"Time taken to construct node input {input_name} ({workflow_node.name}, {request_id}): {construct_node_input_end_time - construct_node_input_start_time:.4f} seconds"
                    )

            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                construct_node_inputs_end_time = time.time()
                self.logger.debug(
                    f"Time taken to construct node inputs ({workflow_node.name}, {request_id}): {construct_node_inputs_end_time - parse_inputs_end_time:.4f} seconds"
                )

            node_inputs = self._deserialize_inputs(node_inputs, workflow_node)

            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                deserialize_inputs_end_time = time.time()
                self.logger.debug(
                    f"Time taken to deserialize inputs ({workflow_node.name}, {request_id}): {deserialize_inputs_end_time - construct_node_inputs_end_time:.4f} seconds"
                )

            batched_infos.append(
                {
                    "request_id": request_id,
                    "workflow_node": workflow_node,
                    "input_map": input_map,
                    "output_map": output_map,
                    "node_inputs": node_inputs,
                    "required_outputs": required_outputs,
                    "lazy_inputs": lazy_inputs,
                    "tensor_infos": tensor_infos,
                }
            )

        model_id = batched_infos[0]["workflow_node"].op.id
        workflow_node = batched_infos[0]["workflow_node"]

        is_batch_processing = True
        if (
            workflow_node.op.id == "IndexedTensor"
            or workflow_node.op.id == "GuidanceTensor"
        ):
            is_batch_processing = False
        if batch_size == 1:
            is_batch_processing = False

        self.logger.debug(f"is_batch_processing: {is_batch_processing}")

        # Prepare node inputs for model execution
        if is_batch_processing:
            batched_node_inputs = {}
            for input_name, node_input in batched_infos[0]["node_inputs"].items():
                if isinstance(node_input, torch.Tensor):
                    tensors = [
                        info["node_inputs"][input_name] for info in batched_infos
                    ]
                    batched_node_inputs[input_name] = torch.cat(tensors, dim=0)
                elif isinstance(node_input, int) or isinstance(node_input, float):
                    # All the node_inputs should have the same value for this input
                    if not all(
                        info["node_inputs"][input_name]
                        == batched_infos[0]["node_inputs"][input_name]
                        for info in batched_infos
                    ):
                        raise ValueError(
                            f"All the node_inputs should have the same value for input {input_name}"
                        )
                    batched_node_inputs[input_name] = batched_infos[0]["node_inputs"][
                        input_name
                    ]
                elif isinstance(node_input, str):
                    batched_node_inputs[input_name] = [
                        batched_infos[i]["node_inputs"][input_name]
                        for i in range(batch_size)
                    ]
                elif isinstance(node_input, Image.Image):
                    batched_node_inputs[input_name] = [
                        batched_infos[i]["node_inputs"][input_name]
                        for i in range(batch_size)
                    ]
                else:
                    raise ValueError(f"Unsupported type: {type(node_input)}")

        #############################################
        # Phase 2: Load model and execute the model #
        #############################################
        if isinstance(workflow_node.op, BaseScheduler):
            assert batch_size == 1, "Scheduler should only be used for batch size 1"
            request_id = batched_infos[0]["request_id"]
            model_instance = self._load_scheduler(
                model_id=model_id,
                request_id=request_id,
                model_path=workflow_node.op.config.model_path,
            )
        else:
            # Get required GPU memory from task message
            gpu_memory_info = task.get("gpu_memory_info", None)
            gpu_memory_required = gpu_memory_info.get(
                "required", None
            )  # Base memory required
            gpu_memory_used = gpu_memory_info.get("used", None)  # Max memory used

            model = get_op(model_id)
            model_path = (
                workflow_node.op.config.model_path
                if workflow_node.op.config is not None
                else None
            )

            # Ensure model is in GPU memory with LRU management
            self._ensure_model_in_gpu(
                model_id,
                model_path=model_path,
                model_base_memory=gpu_memory_required,
                model_max_used_memory=gpu_memory_used,
            )

            # Get model instance from GPU memory
            model_instance = (
                self.gpu_models[model_id]
                if model_path is not None
                else {"model": model, "model_components": None}
            )

        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            load_model_end_time = time.time()
            self.logger.debug(
                f"Time taken to load model ({model_id}): {load_model_end_time - deserialize_inputs_end_time:.4f} seconds"
            )

        model = model_instance["model"]
        model_components = model_instance["model_components"]

        # Get needed patches from workflow node
        patches = workflow_node.op.get_patches()
        needed_patch_ids = {patch.id for patch in patches}

        # Remove patches that are applied but not needed
        self._remove_unnecessary_patches(
            model_id=model_id,
            needed_patch_ids=needed_patch_ids,
            model_components=model_components,
        )

        # Apply patches to the model if needed
        if patches:
            self._apply_necessary_patches(
                model_id=model_id,
                patches=patches,
                model_components=model_components,
            )

        is_execute_generator = inspect.isgeneratorfunction(model.execute)
        self.logger.debug(f"is_execute_generator: {is_execute_generator}")

        batched_node_outputs = []
        notified_tensor_ids = set()
        if is_batch_processing:
            # Execute the model
            if not is_execute_generator:
                execution_results = model.execute(
                    model_components=model_components,
                    device=self.device,
                    mode=workflow_node.mode,
                    **batched_node_inputs,
                )
            else:
                execution_results = {}
                for generator_results in model.execute(
                    model_components=model_components,
                    device=self.device,
                    mode=workflow_node.mode,
                    **batched_node_inputs,
                ):
                    # Prepare all generator results for batch notification
                    notification_data = []
                    for i in range(batch_size):
                        request_id = batched_infos[i]["request_id"]
                        output_map = batched_infos[i]["output_map"]

                        notified_tensor_infos = {}
                        for name, value in generator_results.items():
                            if isinstance(value, torch.Tensor):
                                stored_tensor = self.store_tensor_in_nvshmem(
                                    request_id, output_map[name], value[i : i + 1]
                                )
                                notified_tensor_ids.add(output_map[name])
                                notified_tensor_infos[output_map[name]] = {
                                    "ptr": stored_tensor.data_ptr(),
                                    "size": stored_tensor.size(),
                                    "dtype": type_to_string(stored_tensor.dtype),
                                }
                            else:
                                raise ValueError(f"Unsupported type: {type(value)}")

                        notification_data.append(
                            {
                                "request_id": request_id,
                                "tensor_infos": notified_tensor_infos,
                            }
                        )
                    self._send_tensor_batch_notification(
                        data=notification_data,
                    )
                    execution_results.update(generator_results)

            # Split the results into `batch_size` parts
            for i in range(batch_size):
                node_outputs = {}
                for name, value in execution_results.items():
                    if isinstance(value, torch.Tensor):
                        node_outputs[name] = value[i : i + 1]
                    else:
                        raise ValueError(f"Unsupported type: {type(value)}")
                batched_node_outputs.append(node_outputs)
        else:
            for i in range(batch_size):
                request_id = batched_infos[i]["request_id"]
                workflow_node = batched_infos[i]["workflow_node"]
                node_inputs = batched_infos[i]["node_inputs"]
                output_map = batched_infos[i]["output_map"]

                if not is_execute_generator:
                    node_outputs = model.execute(
                        model_components=model_components,
                        device=self.device,
                        mode=workflow_node.mode,
                        **node_inputs,
                    )
                else:
                    node_outputs = {}
                    for generator_results in model.execute(
                        model_components=model_components,
                        device=self.device,
                        mode=workflow_node.mode,
                        **node_inputs,
                    ):
                        notified_tensor_infos = {}
                        for name, value in generator_results.items():
                            if isinstance(value, torch.Tensor):
                                stored_tensor = self.store_tensor_in_nvshmem(
                                    request_id, output_map[name], value
                                )
                                notified_tensor_ids.add(output_map[name])
                                notified_tensor_infos[output_map[name]] = {
                                    "ptr": stored_tensor.data_ptr(),
                                    "size": stored_tensor.size(),
                                    "dtype": type_to_string(stored_tensor.dtype),
                                }
                            else:
                                raise ValueError(f"Unsupported type: {type(value)}")
                        notification_data = [
                            {
                                "request_id": request_id,
                                "tensor_infos": notified_tensor_infos,
                            }
                        ]
                        self._send_tensor_batch_notification(
                            data=notification_data,
                        )

                        node_outputs.update(generator_results)

                batched_node_outputs.append(node_outputs)

        # Update lazy tensors in the tensor_infos
        for i in range(batch_size):
            request_id = batched_infos[i]["request_id"]
            lazy_inputs = batched_infos[i]["lazy_inputs"]
            tensor_infos = batched_infos[i]["tensor_infos"]
            if lazy_inputs:
                for _, input_info_name in lazy_inputs.items():
                    assert (
                        input_info_name in self.tensor_map[request_id]
                    ), f"Lazy input {input_info_name} is not found in tensor_map for request {request_id}"
                    tensor = self.tensor_map[request_id][input_info_name]
                    tensor_infos.update(
                        {
                            input_info_name: {
                                "ptr": tensor.data_ptr(),
                                "size": tensor.size(),
                                "dtype": type_to_string(tensor.dtype),
                            }
                        }
                    )

        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            execute_model_end_time = time.time()
            self.logger.debug(
                f"Time taken to execute model ({model_id}): {execute_model_end_time - load_model_end_time:.4f} seconds"
            )

        #############################################################################
        # Phase 3: Store tensors in NVSHMEM and prepare results for the coordinator #
        #############################################################################
        task_results = []
        for i in range(batch_size):
            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                store_nvshmem_results_start_time = time.time()

            node_outputs = batched_node_outputs[i]

            request_id = batched_infos[i]["request_id"]
            workflow_node = batched_infos[i]["workflow_node"]
            output_map = batched_infos[i]["output_map"]
            required_outputs = batched_infos[i]["required_outputs"]
            tensor_infos = batched_infos[i]["tensor_infos"]

            # Store all results locally (skip tensors already processed in generator mode)
            for name, value in node_outputs.items():
                if not isinstance(value, torch.Tensor):
                    self.logger.warning(
                        f"Tensor {output_map[name]} is not a tensor, skipping"
                    )
                    continue

                # Skip if tensor was already processed in generator mode
                if output_map[name] in notified_tensor_ids:
                    self.logger.debug(
                        f"Tensor {output_map[name]} already processed in generator mode, skipping"
                    )
                    continue

                stored_tensor = self.store_tensor_in_nvshmem(
                    request_id, output_map[name], value
                )

                tensor_infos[output_map[name]] = {
                    "ptr": stored_tensor.data_ptr(),
                    "size": stored_tensor.size(),
                    "dtype": type_to_string(stored_tensor.dtype),
                }

            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                store_nvshmem_results_end_time = time.time()
                self.logger.debug(
                    f"Time taken to store results in nvshmem ({workflow_node.name}, {request_id}): {store_nvshmem_results_end_time - store_nvshmem_results_start_time:.4f} seconds"
                )

            # Only send required outputs back to coordinator
            output_results = {}

            def _encode_image(img: Image.Image) -> str:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()

            for output_name in required_outputs:
                if output_name in node_outputs:
                    value = node_outputs[output_name]
                    if isinstance(value, torch.Tensor):
                        output_results[output_name] = (
                            value.detach().cpu().numpy().tolist()
                        )
                    elif isinstance(value, list) and all(
                        isinstance(img, Image.Image) for img in value
                    ):
                        # Convert list of images to base64 strings
                        output_results[output_name] = []
                        for img in value:
                            img_str = _encode_image(img)
                            output_results[output_name].append(img_str)
                    elif isinstance(value, Image.Image):
                        # Convert single image to base64 string
                        img_str = _encode_image(value)
                        output_results[output_name] = img_str
                    else:
                        output_results[output_name] = value

            if self.logger.level == logging.DEBUG:
                torch.cuda.synchronize(self.local_rank)
                store_required_results_end_time = time.time()
                self.logger.debug(
                    f"Time taken to store required results by coordinator ({workflow_node.name}, {request_id}): {store_required_results_end_time - store_nvshmem_results_end_time:.4f} seconds"
                )

            task_results.append(
                {
                    "request_id": request_id,
                    "node_name": workflow_node.name,
                    "tensor_infos": tensor_infos,
                    "results": output_results,
                }
            )

        ############################################################
        # Phase 4: Send task completion message to the coordinator #
        ############################################################

        # To make sure that the allocated tensors are correctly synchronized before sending the task completion message to the coordinator
        torch.cuda.synchronize(self.local_rank)

        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            send_task_complete_start_time = time.time()

        # Send task completion message
        self.result_socket.send_json(
            {
                "type": "task_complete",
                "data": task_results,
                "active_models": list(
                    self.gpu_models.keys()
                ),  # Only models currently in GPU memory
                "gpu_memory_info": self._get_gpu_memory_info(self.device),
            },
            flags=zmq.NOBLOCK,
        )

        if self.logger.level == logging.DEBUG:
            torch.cuda.synchronize(self.local_rank)
            send_task_complete_end_time = time.time()
            self.logger.debug(
                f"Time taken to send task completion message for task group {', '.join([f'({node_name}, {request_id})' for node_name, request_id in task_group])}: {send_task_complete_end_time - send_task_complete_start_time:.4f} seconds"
            )
            self.logger.debug(
                f"Time taken to process task group {', '.join([f'({node_name}, {request_id})' for node_name, request_id in task_group])}: {time.time() - start_time:.4f} seconds"
            )

    async def run(self):
        """Main worker loop"""
        self.setup()
        self.logger.info(
            f"[DistributedWorker] Initialized => Hostname: {self.hostname}, "
            f"Local Rank: {self.local_rank}, Global Rank: {self.global_rank}"
        )

        # Start processing tasks in a separate thread
        self.process_loop_task.start()

        # Log initial GPU memory status
        memory_info = self._get_gpu_memory_info(self.device)
        self.logger.info(
            f"Worker {self.global_rank} GPU memory: {memory_info['used'] / (1024**3):.2f} GiB used, "
            f"{memory_info['free'] / (1024**3):.2f} GiB free, {memory_info['total'] / (1024**3):.2f} GiB total"
        )

        poller = zmq.Poller()
        poller.register(self.task_socket, zmq.POLLIN)

        with self.data_engine:
            while self.running:
                try:
                    socks = dict(poller.poll(timeout=0))
                    if self.task_socket not in socks:
                        await asyncio.sleep(0.00001)  # 10 us
                        continue

                    message = self.task_socket.recv_json()

                    if message.get("type") == "stop":  # Poison pill
                        self.logger.info("Received stop signal, initiating shutdown...")
                        break
                    elif message.get("type") == "ping":  # Health check
                        self.result_socket.send_json({"type": "pong"})
                    elif message.get("type") == "task":
                        self.task_queue.put(message)
                    elif message.get("type") == "lazy_arrival":
                        asyncio.create_task(self._handle_lazy_arrival(message))
                    elif message.get("type") == "model_loading":
                        asyncio.create_task(self._handle_model_loading(message))
                    else:
                        self.logger.warning(
                            f"Unknown message type: {message.get('type')}"
                        )

                except Exception as e:
                    traceback.print_exc()
                    self.logger.error(f"Error in worker loop: {e}")
                    continue

        self.cleanup()

    async def _handle_lazy_arrival(self, message: Dict[str, Any]):
        """Handle lazy input notification from coordinator"""
        try:
            request_id = message["request_id"]
            worker_rank = message["worker_rank"]
            tensor_id = message["tensor_id"]
            tensor_info = message["tensor_info"]

            self.logger.debug(
                f"Received lazy input notification for {tensor_id} (request {request_id}), worker {worker_rank}, tensor info: {tensor_info}"
            )

            # Submit fetch task to the data engine instead of immediately fetching
            self.data_engine.submit_fetch_task(
                FetchingTask(
                    id=tensor_id,
                    remote_address=tensor_info["ptr"],
                    size=list(tensor_info["size"]),
                    dtype=string_to_type(tensor_info["dtype"]),
                    remote_nvshmem_pe=worker_rank,
                )
            )

            self.logger.debug(
                f"Submitted fetch task for lazy input {tensor_id} from worker {worker_rank}"
            )

        except Exception as e:
            self.logger.error(f"Error handling lazy input notification: {e}")
            traceback.print_exc()

    async def _handle_model_loading(self, message: Dict[str, Any]):
        """Handle model loading request from coordinator"""
        try:
            model_id = message["model_id"]
            gpu_memory_required = message["gpu_memory_required"]
            gpu_memory_used = message["gpu_memory_used"]

            self.logger.info(
                f"Received model loading request for {model_id}, gpu_memory_required: {gpu_memory_required}, gpu_memory_used: {gpu_memory_used}"
            )

            # Wait for prefetching to complete before processing model loading
            await self.prefetching_complete_event.wait()

            try:
                self._move_model_to_gpu(model_id, gpu_memory_required, gpu_memory_used)
            except Exception as e:
                self.logger.error(f"Failed to load model {model_id}: {e}")

            # Send response back to coordinator with active models and GPU memory info
            response = {
                "type": "model_loading_complete",
                "active_models": list(self.gpu_models.keys()),
                "gpu_memory_info": self._get_gpu_memory_info(self.device),
            }

            self.result_socket.send_json(response, flags=zmq.NOBLOCK)

        except Exception as e:
            self.logger.error(f"Error handling model loading request: {e}")
            traceback.print_exc()

    def _get_gpu_memory_info(self, device: str) -> Dict[str, Any]:
        """Get GPU memory information in bytes"""
        if not torch.cuda.is_available():
            return {"free": 0, "total": 0, "used": 0}

        device_idx = int(device.split(":")[-1])
        torch.cuda.synchronize(device_idx)

        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_idx)
        allocated_memory = torch.cuda.memory_allocated(device_idx)
        free_memory = total_memory - reserved_memory

        return {"free": free_memory, "total": total_memory, "used": allocated_memory}

    def _get_available_gpu_memory(self) -> int:
        """Get available GPU memory in bytes"""
        memory_info = self._get_gpu_memory_info(self.device)
        # self.logger.info(
        #     f"Available GPU memory: {self.available_gpu_memory / (1024**3):.2f} GiB, used: {memory_info['used'] / (1024**3):.2f} GiB"
        # )
        return self.available_gpu_memory - memory_info["used"]

    def _move_model_to_cpu(self, model_id: str):
        """Move a model from GPU to CPU memory"""
        if model_id not in self.gpu_models:
            return

        self.logger.info(f"Moving model {model_id} from GPU to CPU memory")

        # Move model components to CPU
        model_data = self.gpu_models[model_id]
        assert "model_components" in model_data
        # Move all model components to CPU
        for model in model_data["model_components"].values():
            if hasattr(model, "to"):
                model.to("cpu")

        # Store in CPU models
        self.cpu_models[model_id] = model_data

        # Remove from GPU models and update memory tracking
        del self.gpu_models[model_id]
        if model_id in self.model_memory_usage:
            del self.model_memory_usage[model_id]

        # Clear GPU cache
        torch.cuda.empty_cache()

    def _move_model_to_gpu(
        self, model_id: str, model_base_memory: int, model_max_used_memory: int
    ):
        """Move a model from CPU to GPU memory with LRU eviction if needed"""
        if model_id not in self.cpu_models:
            raise RuntimeError(f"Model {model_id} not found in CPU models")

        # Check if we have enough GPU memory
        available_memory = self._get_available_gpu_memory()
        if model_max_used_memory > available_memory:
            # Need to evict models using LRU policy
            self.logger.info(
                f"Available GPU memory: {available_memory / (1024**3):.2f} GiB, model_max_used_memory: {model_max_used_memory / (1024**3):.2f} GiB"
            )
            self._evict_models_for_memory(model_max_used_memory - available_memory)

        self.logger.info(f"Moving model {model_id} from CPU to GPU memory")

        # Move model components to GPU
        model_instance = self.cpu_models[model_id]
        for model in model_instance["model_components"].values():
            if hasattr(model, "to"):
                model.to(self.device)

        # Store in GPU models (most recently used)
        self.gpu_models[model_id] = model_instance
        self.gpu_models.move_to_end(model_id)

        self.logger.info(
            f"Moved model {model_id} from CPU to GPU memory, allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GiB"
        )

        # Remove from CPU models
        del self.cpu_models[model_id]

        # Update memory tracking
        self.model_memory_usage[model_id] = model_base_memory

    def _evict_models_for_memory(self, required_memory: int):
        """Evict least recently used models to free up GPU memory"""
        evicted_memory = 0

        # Evict models in LRU order until we have enough memory
        while evicted_memory < required_memory and self.gpu_models:
            # Get least recently used model
            lru_model_id = next(iter(self.gpu_models))
            model_memory = self.model_memory_usage.get(lru_model_id, 0)

            self.logger.info(
                f"Evicting LRU model {lru_model_id} to free {model_memory:.2f} GiB"
            )
            self._move_model_to_cpu(lru_model_id)
            evicted_memory += model_memory

    def _validate_memory_availability(self, required_memory: int) -> bool:
        """Validate if there's enough GPU memory available for the required memory"""
        if required_memory > self.available_gpu_memory:
            raise RuntimeError(
                f"Insufficient total GPU memory: required {required_memory / (1024**3):.2f} GiB, "
                f"total {self.available_gpu_memory / (1024**3):.2f} GiB"
            )

    def _ensure_model_in_gpu(
        self,
        model_id: str,
        model_path: Optional[str],
        model_base_memory: int,
        model_max_used_memory: int,
    ):
        """Ensure a model is loaded in GPU memory, loading from CPU if needed"""
        if model_path is None:
            # The model doesn't need to be loaded to GPU memory
            return

        # Validate memory availability first
        self._validate_memory_availability(model_max_used_memory)

        if model_id in self.gpu_models:
            # Model is already in GPU, move to end (most recently used)
            self.gpu_models.move_to_end(model_id)
            self.model_access_count[model_id] = (
                self.model_access_count.get(model_id, 0) + 1
            )
        elif model_id in self.cpu_models:
            # Model is in CPU, move to GPU
            self._move_model_to_gpu(model_id, model_base_memory, model_max_used_memory)
        else:
            # Fallback for models that were not listed in the prefetch config.
            available_memory = self._get_available_gpu_memory()
            if model_max_used_memory > available_memory:
                self.logger.info(
                    f"Available GPU memory: {available_memory / (1024**3):.2f} GiB, model_max_used_memory: {model_max_used_memory / (1024**3):.2f} GiB"
                )
                self._evict_models_for_memory(model_max_used_memory - available_memory)

            self.logger.info(
                f"Model {model_id} not found in CPU or GPU memory; loading from {model_path} to GPU"
            )
            self._load_model(model_id, model_path=model_path, device=self.device)
            self.gpu_models.move_to_end(model_id)
            self.model_access_count[model_id] = (
                self.model_access_count.get(model_id, 0) + 1
            )
            self.model_memory_usage[model_id] = model_base_memory

    def prefetch_models(self, models: List[Tuple[str, str]], device: str = "cpu"):
        """Prefetch models to CPU memory for later GPU loading"""
        for model_id, model_path in models:
            self.logger.info(f"Prefetching model {model_id} to {device} memory")
            self._load_model(model_id, model_path=model_path, device=device)

    def _load_scheduler(
        self, model_id: str, request_id: str, model_path: str
    ) -> Dict[str, Any]:
        scheduler = get_op(model_id)
        if model_id not in self.cpu_models:
            self.cpu_models[model_id] = {
                "model": scheduler,
                "model_components": scheduler.initialize(model_path, "cpu"),
            }

        if request_id not in self.request_schedulers:
            self.request_schedulers[request_id] = copy.deepcopy(
                self.cpu_models[model_id]["model_components"]
            )
        return {
            "model": scheduler,
            "model_components": self.request_schedulers[request_id],
        }

    def _load_model(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a model and cache it"""
        model = get_op(model_id)

        if device == "cpu" and model_id in self.cpu_models:
            return self.cpu_models[model_id]
        elif device == self.device and model_id in self.gpu_models:
            return self.gpu_models[model_id]
        else:
            model_components = model.initialize(model_path, device)

            self.logger.debug(
                f"Loading model: {model_id}, model_path: {model_path}, device: {device}"
            )

            model_instance = {
                "model": model,
                "model_components": model_components,
            }

            # Store in appropriate location based on device
            if device == "cpu":
                self.cpu_models[model_id] = model_instance
            elif device == self.device:
                self.gpu_models[model_id] = model_instance
            else:
                raise ValueError(f"Invalid device: {device}")

            return model_instance

    def create_tensor(self, size: Sequence[int], dtype: torch.dtype):
        return self.data_engine.create_tensor(size, dtype=dtype)

    def fetch_tensor(
        self,
        tensor_id: str,
        remote_ptr: int,
        size: Sequence[int],
        dtype: torch.dtype,
        remote_nvshmem_pe: int,
    ):
        self.data_engine.submit_fetch_task(
            FetchingTask(
                id=tensor_id,
                remote_address=remote_ptr,
                size=list(size),
                dtype=dtype,
                remote_nvshmem_pe=remote_nvshmem_pe,
            )
        )
        return self.data_engine.get(tensor_id=tensor_id)

    def free_tensor(self, tensor: torch.Tensor):
        self.data_engine.submit_free_task(FreeingTask(tensor))

    def store_tensor_in_nvshmem(
        self, request_id: str, tensor_id: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        if request_id not in self.tensor_map:
            self.tensor_map[request_id] = {}

        allocated_tensor = self.create_tensor(
            size=tensor.size(),
            dtype=tensor.dtype,
        )
        allocated_tensor.copy_(tensor, True)

        self.tensor_map[request_id][tensor_id] = allocated_tensor

        self.logger.debug(
            f"Tensor {tensor_id} is allocated in nvshmem, ptr: {allocated_tensor.data_ptr()}, size: {allocated_tensor.size()}, dtype: {allocated_tensor.dtype}"
        )

        return allocated_tensor

    def _send_tensor_batch_notification(
        self,
        data: List[Dict[str, Any]],
    ):
        """Send batch tensor notification to coordinator"""
        try:
            notification_message = {
                "type": "tensor_batch_ready",
                "data": data,
            }

            # To make sure that the allocated tensors are correctly synchronized before sending the batch tensor notification to the coordinator.
            torch.cuda.synchronize(self.local_rank)

            self.result_socket.send_json(
                notification_message,
                flags=zmq.NOBLOCK,
            )

            self.logger.debug(
                f"Sent batch tensor notification for {len(data)} tensors to coordinator"
            )

        except Exception as e:
            self.logger.error(f"Failed to send tensor batch notification: {e}")
            traceback.print_exc()

    def cleanup(self):
        try:
            if self.running:
                self.running = False
                self.process_loop_task.join()

                self.logger.info(f"Worker {self.global_rank} is stopping")

                # Cleanup models
                self.request_schedulers.clear()
                self.cpu_models.clear()
                self.gpu_models.clear()
                self.model_memory_usage.clear()
                self.model_access_count.clear()
                self.tensor_map.clear()

                # Cleanup patches
                self.patches.clear()

                # Cleanup ZMQ resources
                self.task_socket.setsockopt(zmq.LINGER, 1000)  # 1 second timeout
                self.result_socket.setsockopt(zmq.LINGER, 1000)

                self.task_socket.close()
                self.result_socket.close()
                self.context.term()

                self.logger.info("Cleanup completed successfully")
                # Close all handlers
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
        except Exception as e:
            self.logger.error(f"Error during worker cleanup: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-port", type=int, default=14000)
    parser.add_argument(
        "--prefetch-models-config",
        type=str,
        default=str(DEFAULT_CONFIG_DIR / "prefetch_models.yaml"),
        help="Path to prefetch models YAML config file",
    )
    args = parser.parse_args()

    base_port = args.base_port

    global_rank = MPI.COMM_WORLD.Get_rank()
    global_size = MPI.COMM_WORLD.Get_size()
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()
    hostname = MPI.Get_processor_name()
    print(
        f"Worker {global_rank}/{global_size} (local rank {local_rank}/{local_size}) starting on {hostname}..."
    )

    worker = DistributedWorker(
        local_rank=local_rank,
        global_rank=global_rank,
        base_port=base_port,
        hostname=hostname,
        prefetch_models_config=args.prefetch_models_config,
    )
    print(f"Worker {global_rank} initialized")
    asyncio.run(worker.run())
