import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
import zmq

from diflow.backend import dependency
from diflow.backend.scheduler import (
    SchedulingPolicy,
    Task,
    create_scheduler,
)
from diflow.backend.scheduler.task import get_task_id
from diflow.interface.node_io import SourceType
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_expand import expand_workflow
from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.operator_ids import (
    GUIDANCE_TENSOR_ID,
    INDEXED_TENSOR_ID,
)
from diflow.operators.schedulers.base_scheduler import BaseScheduler
from diflow.profiling.runtime_profile import RuntimeProfile, RuntimeProfileError


class ExecutionTimeoutError(Exception):
    """Exception raised when a request is rejected by the early-abort policy."""

    pass


class Coordinator:
    def __init__(
        self,
        worker_hostnames: List[str],
        scheduling_policy: SchedulingPolicy,
        preload_models_config: str,
        model_batch_config: str,
        runtime_profile: RuntimeProfile,
        base_port: int = 14000,
        enable_early_abort: bool = False,
    ):
        self.worker_hostnames = worker_hostnames
        self.scheduling_policy = scheduling_policy
        self.base_port = base_port
        self.preload_models_config = preload_models_config
        self.model_batch_config = model_batch_config
        self.enable_early_abort = enable_early_abort
        self.runtime_profile = runtime_profile

        # Setup logging
        self._setup_logging()

        # ZMQ setup
        self.context = zmq.Context()
        self.task_sockets: Dict[int, zmq.Socket] = {}  # For sending tasks
        self.result_sockets: Dict[int, zmq.Socket] = {}  # For receiving results

        # Coordinator setup
        # map worker_rank to (hostname, task_port, result_port)
        self.all_workers_info = self._gather_all_workers_info()
        self._setup_coordinator_sockets()

        # map request_id -> NodeIO name -> worker_rank -> tensor_info (ptr, size, dtype)
        self.tensor_map: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

        # Task tracking
        # Tasks ready to be scheduled
        self.ready_tasks: List[Task] = []
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        # map request_id -> Dict[WorkflowNode name -> Task]
        self.request_tasks: Dict[str, Dict[str, Task]] = {}
        # map request_id -> results
        self.request_results: Dict[str, Dict[str, Any]] = {}
        # map request_id -> required outputs
        self.request_required_outputs: Dict[str, Dict[str, str]] = {}
        # map request_id -> completion event
        self.request_completed: Dict[str, asyncio.Event] = {}
        # map request_id -> arrival time
        self.request_arrival_times: Dict[str, float] = {}
        # map request_id -> worker_rank
        # TODO @ Lingyun: currently, we assume only one denoise scheduler per request
        self.denoise_scheduler_worker: Dict[str, int] = {}

        # Load preload models from YAML config file
        self.preload_models: List[str] = self._load_preload_models_config()

        # Dependency tracking
        # map request_id -> node_name -> set of prerequisite node names
        self.node_prerequisites: Dict[str, Dict[str, set]] = {}
        # map request_id -> node_name -> set of lazy prerequisite node names
        self.node_lazy_prerequisites: Dict[str, Dict[str, set]] = {}
        # map request_id -> node_name -> set of non-lazy prerequisite node names
        self.node_non_lazy_prerequisites: Dict[str, Dict[str, set]] = {}
        # map request_id -> node_name -> set of successor node names (reverse dependencies)
        self.node_successors: Dict[str, Dict[str, set]] = {}
        # map request_id -> set of scheduled node names
        self.scheduled_nodes: Dict[str, set] = {}
        # map request_id -> set of completed node names
        self.completed_nodes: Dict[str, set] = {}
        # map request_id -> node_name -> depth (1: root node, 2: first level dependent node, etc.)
        self.node_depth: Dict[str, Dict[str, int]] = {}
        # map request_id -> NodeIO name -> reference count
        # In order to free tensors that are no longer needed.
        self.tensor_reference_count: Dict[str, Dict[str, int]] = {}
        # Lock for freeing tensors
        self.free_tensor_lock = asyncio.Lock()
        # map worker_rank -> request_id -> tensor list
        self.free_tensor_dict: Dict[int, Dict[str, List[str]]] = {}

        # Lazy input tracking
        # map request_id -> NodeIO name -> set of (node_name, worker_rank) that are waiting for this lazy input
        self.lazy_input_dependents: Dict[str, Dict[str, set]] = {}

        self.scheduler = create_scheduler(
            scheduling_policy=self.scheduling_policy,
            all_workers_info=self.all_workers_info,
            runtime_profile=self.runtime_profile,
        )

        # Read model batch configs
        self.model_batch_configs = json.load(open(self.model_batch_config))
        self.logger.info(f"Loaded model batch configs: {self.model_batch_configs}")

        # Asynchronous tasks
        self._scheduler_task = None
        self._gather_result_task = None

        # Status
        self.is_running = True

    def _setup_logging(self):
        """Setup logging for the coordinator"""
        log_dir = os.environ.get("DIFLOW_LOG_DIR", "logs")

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/coordinator_{timestamp}.log"

        log_level_str = os.environ.get("LOGLEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Setup the logger
        self.logger = logging.getLogger("coordinator")
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

        self.logger.info(f"Initialized logging for coordinator, log file: {log_file}")

    def _load_preload_models_config(self) -> List[str]:
        """Load preload models configuration from YAML file"""
        try:
            with open(self.preload_models_config, "r") as file:
                config = yaml.safe_load(file)
                preload_models = config.get("preload_models") or []
                self.logger.info(
                    f"Loaded {len(preload_models)} preload models from config: {preload_models}"
                )
                return preload_models
        except FileNotFoundError:
            self.logger.warning(
                f"Preload models config file not found: {self.preload_models_config}"
            )
            return []
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing preload models config file: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading preload models config: {e}")
            return []

    async def wait_for_workers_ready(
        self,
        timeout_seconds: float = 60,
        health_check=None,
    ):
        """Wait for all workers to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if not self.is_running:
                raise RuntimeError("Coordinator was stopped before workers were ready")
            if health_check is not None and not health_check():
                raise RuntimeError("Worker process exited before becoming ready")
            if self._check_workers_ready():
                await self.load_models_on_workers()
                return
            await asyncio.sleep(1)
        raise TimeoutError(
            f"Workers failed to initialize within {timeout_seconds:g} seconds"
        )

    def _check_workers_ready(self) -> bool:
        """Check if all workers are ready"""
        try:
            # Send ping to all workers
            for worker_rank, socket in self.task_sockets.items():
                socket.send_json({"type": "ping"})

            # Wait for responses
            for socket in self.result_sockets.values():
                if socket.poll(timeout=200, flags=zmq.POLLIN) == 0:
                    return False
                response = socket.recv_json()
                if response.get("type") != "pong":
                    return False
            return True
        except zmq.ZMQError:
            return False

    def _setup_coordinator_sockets(self):
        """Setup ZMQ sockets for the coordinator"""
        for worker_rank, worker_info in self.all_workers_info.items():

            task_socket = self.context.socket(zmq.PUSH)
            result_socket = self.context.socket(zmq.PULL)

            task_socket.connect(
                f"tcp://{worker_info['hostname']}:{worker_info['task_port']}"
            )
            result_socket.connect(
                f"tcp://{worker_info['hostname']}:{worker_info['result_port']}"
            )
            self.logger.info(
                f"Coordinator connected to worker_{worker_rank}, {worker_info['hostname']}:{worker_info['task_port']} and {worker_info['hostname']}:{worker_info['result_port']}"
            )

            self.task_sockets[worker_rank] = task_socket
            self.result_sockets[worker_rank] = result_socket

    def _gather_all_workers_info(self) -> Dict[int, Dict[str, Any]]:
        """Gather information about all workers across all nodes"""
        workers_info = {}
        for rank, hostname in enumerate(self.worker_hostnames):
            worker_rank = rank
            task_port = self.base_port + worker_rank * 2
            result_port = task_port + 1

            self.logger.info(
                f"Worker: {worker_rank}, "
                f"task_port: {task_port}, result_port: {result_port}"
            )
            workers_info[worker_rank] = {
                "hostname": hostname,
                "task_port": task_port,
                "result_port": result_port,
            }
        return workers_info

    def _get_node_dependencies(self, workflow: Workflow) -> Dict[str, set]:
        """Build dependency map for each node"""
        return dependency.get_node_dependencies(workflow)

    def _get_lazy_input_dependencies(self, workflow: Workflow) -> Dict[str, set]:
        """Build dependency map for lazy inputs only"""
        return dependency.get_node_dependencies(workflow, lazy_only=True)

    def _topological_sort(self, workflow: Workflow) -> List[WorkflowNode]:
        """Sort workflow nodes in topological order"""
        return dependency.topological_sort(workflow)

    def _build_dependency_graph(self, request_id: str, workflow: Workflow):
        """Build dependency graph for a request"""
        prerequisites = self._get_node_dependencies(workflow)
        lazy_prerequisites = self._get_lazy_input_dependencies(workflow)

        # Initialize dependency tracking for this request
        self.node_prerequisites[request_id] = prerequisites
        self.node_lazy_prerequisites[request_id] = lazy_prerequisites
        # for node, deps in self.node_lazy_prerequisites[request_id].items():
        #     self.logger.debug(f"Lazy prerequisites for {node}: {deps}")
        # Compute non-lazy prerequisites per node: all prerequisites minus lazy prerequisites for that node
        self.node_non_lazy_prerequisites[request_id] = {
            node: prerequisites[node] - lazy_prerequisites.get(node, set())
            for node in prerequisites
        }
        # for node, deps in self.node_non_lazy_prerequisites[request_id].items():
        #     self.logger.debug(f"Non-lazy prerequisites for {node}: {deps}")
        self.completed_nodes[request_id] = set()

        # Build reverse dependency map (successors)
        self.node_successors[request_id] = dependency.build_successors(
            workflow, prerequisites
        )

        # Record node depth
        self.node_depth[request_id] = dependency.compute_node_depth(
            workflow, prerequisites, self.node_successors[request_id]
        )

        # Initialize tensor reference count for this request
        self.tensor_reference_count[request_id] = (
            dependency.build_tensor_reference_count(workflow)
        )

    def _is_task_ready(self, task: Task) -> bool:
        """Check if a task's dependencies are satisfied"""
        request_id = task.request_id
        node_name = task.workflow_node.name

        if request_id not in self.node_prerequisites:
            raise ValueError(f"Request {request_id} has no dependency graph")

        # Check if all non-lazy prerequisites are completed
        non_lazy_prerequisites = self.node_non_lazy_prerequisites[request_id].get(
            node_name, set()
        )
        completed = self.completed_nodes[request_id]

        return non_lazy_prerequisites.issubset(completed)

    def _prepare_task_inputs(self, task: Task) -> Task:
        """Prepare inputs for a task"""
        node = task.workflow_node
        inputs = task.inputs

        # Prepare inputs for this node
        input_map = {}
        output_map = {}
        node_inputs = {}
        node_input_locations = {}
        lazy_inputs = {}

        # All prerequisites should be satisfied
        for input_name, input_info in node.get_inputs().items():
            input_map[input_name] = input_info.name

            # For lazy inputs, we don't require them to be available at scheduling time.
            if input_info.lazy:
                # Record lazy inputs that need to be provided at execution time
                lazy_inputs[input_name] = input_info.name
                continue

            if input_info.name in inputs:
                # input is from workflow input
                node_inputs[input_name] = inputs[input_info.name]
            # Suyi: this branch supports non-tensor intermediate data needed by subsequent nodes.
            # Suyi: this branch actually also support other intermediate data that is not a tensor, but registered as the workflow output in workflow registration.
            # Suyi: Not sure if this is a good design, but it works for now.
            elif input_info.name in self.request_required_outputs[task.request_id]:
                input_name = self.request_required_outputs[task.request_id][
                    input_info.name
                ]
                node_inputs[input_name] = self.request_results[task.request_id][
                    input_name
                ]
            elif (
                task.request_id in self.tensor_map
                and input_info.name in self.tensor_map[task.request_id]
            ):
                # input is stored in local storage of a worker
                node_inputs[input_name] = None
                node_input_locations[input_name] = self.tensor_map[task.request_id][
                    input_info.name
                ]
            else:
                # Error: This shouldn't happen for tasks in queue
                self.logger.error(
                    f"Task {node.name} in queue but missing input {input_info.name}"
                )
                continue

        # Determine which outputs are needed
        required_outputs = []  # list of output_info.name
        for output_name, output_info in node.get_outputs().items():
            output_map[output_name] = output_info.name
            if output_info.name in self.request_required_outputs[task.request_id]:
                required_outputs.append(output_name)

        task.input_map = input_map
        task.output_map = output_map
        task.node_inputs = node_inputs
        task.node_input_locations = node_input_locations
        task.required_outputs = required_outputs
        task.lazy_inputs = lazy_inputs

        return task

    async def _notify_lazy_input_available(
        self,
        request_id: str,
        tensor_id: str,
        src_worker_rank: int,
    ):
        """Send a lazy input notification to all workers that are waiting for this lazy input"""
        try:
            if (
                request_id in self.lazy_input_dependents
                and tensor_id in self.lazy_input_dependents[request_id]
            ):
                for (
                    _,
                    dst_worker_rank,
                ) in self.lazy_input_dependents[
                    request_id
                ][tensor_id]:
                    tensor_info = self.tensor_map[request_id][tensor_id][
                        src_worker_rank
                    ]
                    notification_message = {
                        "type": "lazy_arrival",
                        "request_id": request_id,
                        "tensor_id": tensor_id,
                        "worker_rank": src_worker_rank,
                        "tensor_info": tensor_info,
                    }

                    self.task_sockets[dst_worker_rank].send_json(
                        notification_message,
                        flags=zmq.NOBLOCK,
                    )

                    self.logger.debug(
                        f"Sent lazy input notification for {tensor_id} to worker {dst_worker_rank}"
                    )

                del self.lazy_input_dependents[request_id][tensor_id]

        except Exception as e:
            self.logger.error(f"Failed to send lazy input notification: {e}")

    def _mark_node_completed(self, request_id: str, node_name: str):
        """Mark a node as completed and add ready dependent tasks to queue"""
        if request_id not in self.completed_nodes:
            raise ValueError(f"Request {request_id} has no completed nodes")

        if request_id not in self.node_successors:
            raise ValueError(f"Request {request_id} has no successor graph")

        self.completed_nodes[request_id].add(node_name)
        self.logger.debug(
            f"Marked node {node_name} as completed for request {request_id}"
        )

        # Check if any successor nodes are now ready
        successors = self.node_successors[request_id].get(node_name, set())
        for successor_name in successors:
            task = self.request_tasks[request_id][successor_name]
            if (
                self._is_task_ready(task)
                and successor_name not in self.scheduled_nodes[request_id]
            ):
                task = self._prepare_task_inputs(task)
                self.scheduled_nodes[request_id].add(successor_name)
                asyncio.create_task(self.task_queue.put(task))
                self.logger.debug(
                    f"Added task {successor_name} to task queue for request {request_id}"
                )

    async def _update_free_tensor_dict(
        self, request_id: str, workflow_node: WorkflowNode
    ):
        """Update the free tensor dict for a request"""
        start_time = time.time()

        # Check if any tensors from this node are no longer needed
        node = workflow_node
        unused_tensors = []
        for input_name, input_info in node.get_inputs().items():
            if input_info.source_type == SourceType.INPUT:
                continue
            if input_info.name not in self.tensor_reference_count[request_id]:
                self.logger.error(
                    f"Tensor {input_info.name} not found in tensor_reference_count for request {request_id}"
                )
                continue
            self.tensor_reference_count[request_id][input_info.name] -= 1
            if self.tensor_reference_count[request_id][input_info.name] == 0:
                unused_tensors.append(input_info.name)
        if len(unused_tensors) > 0:
            async with self.free_tensor_lock:
                for tensor_id in unused_tensors:
                    for worker_rank in self.tensor_map[request_id][tensor_id]:
                        if worker_rank not in self.free_tensor_dict:
                            self.free_tensor_dict[worker_rank] = {}
                        if request_id not in self.free_tensor_dict[worker_rank]:
                            self.free_tensor_dict[worker_rank][request_id] = []
                        self.free_tensor_dict[worker_rank][request_id].append(tensor_id)

        end_time = time.time()
        self.logger.debug(
            f"Time taken to update free tensor dict: {end_time - start_time} seconds, request_id: {request_id}, node_name: {workflow_node.name}"
        )

    async def _prepare_ready_tasks(self):
        """Prepare ready tasks for scheduling"""
        sorted_ready_tasks = False
        while not self.task_queue.empty():
            task = await self.task_queue.get()
            self.ready_tasks.append(task)
            sorted_ready_tasks = True

        # Sort the list if there is any new ready tasks.
        if sorted_ready_tasks:
            # Lingyun: is it necessary for different scheduling policies to implement their own sorting logic?
            self.ready_tasks.sort(
                key=lambda task: (
                    self.request_arrival_times[task.request_id],
                    -self.node_depth[task.request_id][task.workflow_node.name],
                )
            )

    def _calculate_required_gpu_memory(self, task_group: List[Task]) -> Tuple[int, int]:
        """Return base model memory and total peak memory for a task group."""
        workflow_node = task_group[0].workflow_node
        batch_size = len(task_group)
        model_name = workflow_node.op.id

        if (
            model_name == INDEXED_TENSOR_ID
            or model_name == GUIDANCE_TENSOR_ID
            or isinstance(workflow_node.op, BaseScheduler)
        ):
            return 0, 0

        inputs = task_group[0].inputs or {}
        height = int(inputs.get("height", 0))
        width = int(inputs.get("width", 0))
        try:
            base_memory = self.runtime_profile.model_memory(model_name)
            total_peak_memory = self.runtime_profile.peak_total_memory(
                model_name,
                workflow_node.mode,
                batch_size,
                height,
                width,
            )
        except RuntimeProfileError as error:
            self.logger.warning(
                "No usable memory profile for %s (mode=%s, batch=%d, "
                "shape=%dx%d); continuing without proactive memory reservation: %s",
                model_name,
                workflow_node.mode,
                batch_size,
                height,
                width,
                error,
            )
            return 0, 0

        self.logger.debug(
            f"Calculated GPU memory requirement for {model_name} "
            f"(batch_size={batch_size}, shape={height}x{width}): "
            f"base={base_memory / (1024**3):.2f}GB, "
            f"total_peak={total_peak_memory / (1024**3):.2f}GB"
        )
        return base_memory, total_peak_memory

    async def _select_worker_for_task_group(self, task_group: List[Task]) -> int:
        """Select a worker for a task group"""
        is_scheduler_op = isinstance(task_group[0].workflow_node.op, BaseScheduler)
        if is_scheduler_op:
            assert len(task_group) == 1, "Scheduler operation should only have one task"

            request_id = task_group[0].request_id
            if request_id in self.denoise_scheduler_worker:
                selected_worker_rank = self.denoise_scheduler_worker[request_id]
                await self.scheduler.reserve_worker_for_task_group(
                    worker_rank=selected_worker_rank,
                    task_group=task_group,
                )
            else:
                selected_worker_rank = (
                    await self.scheduler.select_worker_for_task_group(task_group)
                )
        else:
            selected_worker_rank = await self.scheduler.select_worker_for_task_group(
                task_group
            )

        await self.scheduler.update_worker_status_after_scheduling(
            worker_rank=selected_worker_rank,
            task_group=task_group,
        )

        if is_scheduler_op:
            # Mark the worker as the scheduler worker for this request
            request_id = task_group[0].request_id
            if request_id not in self.denoise_scheduler_worker:
                self.denoise_scheduler_worker[request_id] = selected_worker_rank
            else:
                assert self.denoise_scheduler_worker[request_id] == selected_worker_rank

        return selected_worker_rank

    async def run_scheduler(self):
        """Run the scheduler that processes the request queue"""
        self._scheduler_task = asyncio.create_task(self._process_task_queue_loop())
        self._gather_result_task = asyncio.create_task(self._gather_result_loop())

    async def _process_task_queue_loop(self):
        """Continuously process the task queue"""
        while self.is_running:
            await self._prepare_ready_tasks()

            if len(self.ready_tasks) == 0:
                await asyncio.sleep(0.00001)  # 10 us
                continue

            # Check if any worker is available
            if not await self.scheduler.check_worker_availability(self.ready_tasks[0]):
                await asyncio.sleep(0.00001)  # 10 us
                continue

            task_group = await self.scheduler.select_tasks_for_grouping(
                self.ready_tasks,
                self.model_batch_configs,
            )

            # Select a worker for the task group
            selected_worker_rank = await self._select_worker_for_task_group(task_group)

            self.logger.debug(
                f"Selected worker {selected_worker_rank} for task group with batch size {len(task_group)}: {', '.join([f'({task.workflow_node.name}, {task.request_id})' for task in task_group])}"
            )

            if selected_worker_rank is None:
                raise ValueError(
                    "No worker available for task group after checking worker availability, should not happen"
                )

            # Schedule task
            asyncio.create_task(
                self._schedule_task_group(task_group, selected_worker_rank)
            )

    async def _gather_result_loop(self):
        """Continuously gather results for completed tasks"""
        poller = zmq.Poller()
        for result_socket in self.result_sockets.values():
            poller.register(result_socket, zmq.POLLIN)

        while self.is_running:
            socks = dict(poller.poll(timeout=0))

            if len(socks) == 0:
                await asyncio.sleep(0.00001)  # 10 us
                continue

            for worker_rank, result_socket in self.result_sockets.items():
                if result_socket in socks:
                    response = result_socket.recv_json()

                    if response.get("type") == "pong":
                        continue

                    elif response.get("type") == "tensor_batch_ready":
                        # Handle batch tensor notification from generator mode
                        response_data_list = response["data"]
                        self.logger.debug(
                            f"Received batch tensor notification for {len(response_data_list)} tensors from worker_{worker_rank}"
                        )
                        for response_data in response_data_list:
                            request_id = response_data["request_id"]
                            tensor_infos = response_data["tensor_infos"]

                            self.logger.debug(
                                f"Received batch tensor notification for {len(tensor_infos)} tensors (request {request_id}) from worker_{worker_rank}"
                            )

                            # Update tensor_map for all tensors in this batch
                            if request_id not in self.tensor_map:
                                self.tensor_map[request_id] = {}

                            for tensor_id, tensor_info in tensor_infos.items():
                                if tensor_id not in self.tensor_map[request_id]:
                                    self.tensor_map[request_id][tensor_id] = {}
                                self.tensor_map[request_id][tensor_id][
                                    worker_rank
                                ] = tensor_info

                                # Notify tasks waiting for this lazy input
                                asyncio.create_task(
                                    self._notify_lazy_input_available(
                                        request_id,
                                        tensor_id,
                                        worker_rank,
                                    )
                                )

                    elif response.get("type") == "task_complete":
                        response_data_list = response["data"]
                        active_models = response.get("active_models", [])
                        gpu_memory_info = response.get("gpu_memory_info", {})

                        task_info = ", ".join(
                            [
                                f"({task['node_name']}, {task['request_id']})"
                                for task in response_data_list
                            ]
                        )
                        self.logger.debug(
                            f"Received results of task group {task_info} from worker_{worker_rank}"
                        )
                        self.logger.debug(
                            f"Worker {worker_rank} active models: {active_models}, GPU memory: {gpu_memory_info}"
                        )

                        await self.scheduler.update_worker_status_after_completion(
                            worker_rank=worker_rank,
                            active_models=active_models,
                            gpu_memory_info=gpu_memory_info,
                            task_id=get_task_id(response_data_list),
                        )

                        for response_data in response_data_list:
                            node_name = response_data["node_name"]
                            request_id = response_data["request_id"]
                            # map output_info.name -> tensor_info (ptr, size, dtype)
                            tensor_infos = response_data["tensor_infos"]
                            results = response_data["results"]

                            # Update tensor_map
                            for tensor_id, tensor_info in tensor_infos.items():
                                if request_id not in self.tensor_map:
                                    self.tensor_map[request_id] = {}
                                if tensor_id not in self.tensor_map[request_id]:
                                    self.tensor_map[request_id][tensor_id] = {}
                                self.tensor_map[request_id][tensor_id][
                                    worker_rank
                                ] = tensor_info

                                # self.logger.debug(
                                #     f"Updating tensor_map for request {request_id} with tensor_id {tensor_id} and worker_rank {worker_rank}"
                                # )

                                # Notify tasks waiting for this lazy input
                                asyncio.create_task(
                                    self._notify_lazy_input_available(
                                        request_id,
                                        tensor_id,
                                        worker_rank,
                                    )
                                )

                            node_task = self.request_tasks[request_id][node_name]

                            # Store results
                            if len(results) > 0:
                                self.logger.info(
                                    f"Storing results for {node_name} in request {request_id}"
                                )
                                if request_id not in self.request_results:
                                    self.request_results[request_id] = {}
                                for (
                                    output_name,
                                    output_info,
                                ) in node_task.workflow_node.get_outputs().items():
                                    if output_name in results:
                                        final_output_name = (
                                            self.request_required_outputs[request_id][
                                                output_info.name
                                            ]
                                        )
                                        self.request_results[request_id][
                                            final_output_name
                                        ] = results[output_name]

                            # Mark node as completed and add ready dependent tasks to queue
                            self._mark_node_completed(request_id, node_name)

                            # Update free tensor dict
                            asyncio.create_task(
                                self._update_free_tensor_dict(
                                    request_id, node_task.workflow_node
                                )
                            )

                            # Check if request is complete
                            if self._is_request_complete(request_id):
                                await self.scheduler.cleanup_request(request_id)
                                self.request_completed[request_id].set()

                    elif response.get("type") == "model_loading_complete":
                        self.logger.debug(
                            f"Received model loading complete response from worker_{worker_rank}"
                        )

                        active_models = response.get("active_models", [])
                        gpu_memory_info = response.get("gpu_memory_info", {})

                        self.logger.debug(
                            f"Worker {worker_rank} active models: {active_models}, GPU memory: {gpu_memory_info}"
                        )

                        await self.scheduler.update_worker_status_after_completion(
                            worker_rank=worker_rank,
                            active_models=active_models,
                            gpu_memory_info=gpu_memory_info,
                        )

                    else:
                        self.logger.error(
                            f"Received unknown response from worker_{worker_rank}: {response}"
                        )

    async def _schedule_task_group(self, task_group: List[Task], worker_rank: int):
        """Schedule a task to a specific worker"""
        try:
            task_message = {"type": "task", "data": []}

            # Compose free tensor dict for this task group together
            async with self.free_tensor_lock:
                if worker_rank in self.free_tensor_dict:
                    task_message["free_tensor_dict"] = self.free_tensor_dict[
                        worker_rank
                    ]
                    self.free_tensor_dict[worker_rank] = {}

            # Calculate required GPU memory for the task group
            gpu_memory_required, gpu_memory_used = self._calculate_required_gpu_memory(
                task_group
            )
            task_message["gpu_memory_info"] = {
                "required": gpu_memory_required,
                "used": gpu_memory_used,
            }

            self.logger.debug(
                f"Scheduling task group {', '.join([f'({task.workflow_node.name}, {task.request_id})' for task in task_group])} on worker_{worker_rank}"
            )

            for task in task_group:
                if task.lazy_inputs:
                    removed_lazy_inputs = []
                    for (
                        lazy_input_name,
                        lazy_input_info_name,
                    ) in task.lazy_inputs.items():
                        # If the lazy input is already available, update the node_input_locations.
                        if lazy_input_info_name in self.tensor_map[task.request_id]:
                            task.node_input_locations[lazy_input_name] = (
                                self.tensor_map[task.request_id][lazy_input_info_name]
                            )
                            removed_lazy_inputs.append(lazy_input_name)
                            # self.logger.debug(
                            #     f"Lazy input {lazy_input_info_name} is already available for task {task.workflow_node.name}"
                            # )
                        else:
                            if task.request_id not in self.lazy_input_dependents:
                                self.lazy_input_dependents[task.request_id] = {}
                            if (
                                lazy_input_info_name
                                not in self.lazy_input_dependents[task.request_id]
                            ):
                                self.lazy_input_dependents[task.request_id][
                                    lazy_input_info_name
                                ] = set()
                            self.lazy_input_dependents[task.request_id][
                                lazy_input_info_name
                            ].add((task.workflow_node.name, worker_rank))

                            # self.logger.debug(
                            #     f"Added lazy input {lazy_input_info_name} dependency for task {task.workflow_node.name} on worker_{worker_rank}"
                            # )

                    for lazy_input_name in removed_lazy_inputs:
                        del task.lazy_inputs[lazy_input_name]

                task_message["data"].append(
                    {
                        "request_id": task.request_id,
                        "node_name": task.workflow_node.name,
                        "workflow_node": task.workflow_node.to_dict(),
                        "input_map": task.input_map,
                        "output_map": task.output_map,
                        "node_inputs": task.node_inputs,
                        "node_input_locations": task.node_input_locations,
                        "required_outputs": task.required_outputs,
                        "lazy_inputs": task.lazy_inputs,
                    }
                )

            self.task_sockets[worker_rank].send_json(
                task_message,
                flags=zmq.NOBLOCK,
            )
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error scheduling task: {e}")

    def _is_request_complete(self, request_id: str) -> bool:
        """Check if all tasks for a request are complete"""
        if request_id not in self.completed_nodes:
            return False

        # Check if all nodes are completed
        if request_id not in self.request_tasks:
            return False

        total_nodes = len(self.request_tasks[request_id].keys())
        completed_count = len(self.completed_nodes[request_id])
        self.logger.debug(
            f"Completed {completed_count} out of {total_nodes} nodes for request {request_id}"
        )

        return completed_count == total_nodes

    def _is_system_overloaded(self, slo_slack: Optional[float] = None) -> bool:
        """Estimate whether accepting a new request would miss its remaining SLO."""
        if not self.enable_early_abort or slo_slack is None:
            return False

        waiting_task_latencies: Dict[str, List[float]] = {}
        waiting_tasks = []
        num_executors = len(self.all_workers_info)

        inflight_request_ids = list(self.request_arrival_times.keys())
        for request_id in inflight_request_ids:
            if (
                request_id not in self.request_tasks
                or request_id not in self.completed_nodes
            ):
                # Another request can be between arrival registration and task
                # graph setup; it is not schedulable work yet.
                continue

            waiting_task_latencies[request_id] = []
            for task_name in self.request_tasks[request_id]:
                if task_name in self.completed_nodes[request_id]:
                    continue

                task = self.request_tasks[request_id][task_name]
                waiting_tasks.append(task_name)
                task_inputs = task.inputs or {}
                waiting_task_latencies[request_id].append(
                    self.runtime_profile.execution_latency(
                        task.workflow_node.op.id,
                        task.workflow_node.mode,
                        1,
                        int(task_inputs.get("height", 0)),
                        int(task_inputs.get("width", 0)),
                    )
                )

        if num_executors > len(waiting_task_latencies):
            return False

        def calculate_request_completion_time(
            request_latencies: Dict[str, List[float]], num_executors: int
        ) -> float:
            """Lower-bound completion time for request-serial work on workers."""
            if num_executors <= 0:
                return float("inf")

            total_work = 0.0
            max_request_work = 0.0
            for latencies in request_latencies.values():
                # Tasks inside one request must preserve dependency order, while
                # separate requests can occupy separate executors.
                request_work = sum(latencies)
                total_work += request_work
                max_request_work = max(max_request_work, request_work)

            if total_work == 0.0:
                return 0.0

            return max(total_work / num_executors, max_request_work)

        estimated_completion_time = calculate_request_completion_time(
            waiting_task_latencies, num_executors
        )
        self.logger.debug(
            f"Estimated time to complete all inflight requests: {estimated_completion_time:.4f}s, "
            f"slo_slack: {slo_slack:.4f}s"
        )
        self.logger.debug(f"Waiting task latencies: {waiting_task_latencies}")
        self.logger.debug(f"Waiting tasks: {waiting_tasks}")

        return estimated_completion_time > slo_slack

    async def execute_workflow(
        self,
        request_id: str,
        workflow: Workflow,
        inputs: Dict[str, Any],
        slo_slack: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow asynchronously"""

        if self._is_system_overloaded(slo_slack):
            failure_msg = f"Request {request_id} is rejected due to system overload"
            raise ExecutionTimeoutError(failure_msg)

        # Validate the expanded request before registering any inflight state.
        workflow = expand_workflow(workflow, inputs)
        height = int(inputs.get("height", 0))
        width = int(inputs.get("width", 0))
        for node in workflow.workflow_nodes:
            try:
                self.runtime_profile.validate_execution(
                    node.op.id, node.mode, 1, height, width
                )
            except RuntimeProfileError as error:
                if self.enable_early_abort and slo_slack is not None:
                    raise RuntimeProfileError(
                        "Admission control profile unavailable for this request: "
                        f"{error}"
                    ) from error
                self.logger.warning(
                    "Proceeding without a complete runtime profile because this "
                    "request has no active SLA admission decision: %s",
                    error,
                )
                break

        workflow_start_time = time.time()
        self.logger.info(
            f"Started to execute workflow {workflow.name} for request {request_id}"
        )
        self.request_arrival_times[request_id] = workflow_start_time

        # Sort nodes in topological order
        # sorted_nodes = self._topological_sort(workflow)

        # Build dependency graph for this request
        self._build_dependency_graph(request_id, workflow)

        # Store required outputs for this request
        self.request_required_outputs[request_id] = workflow.outputs
        self.logger.debug(
            f"request_required_outputs: {self.request_required_outputs[request_id]}"
        )

        # Create tasks for each WorkflowNode
        tasks = {}
        self.scheduled_nodes[request_id] = set()
        for node in workflow.workflow_nodes:
            tasks[node.name] = Task(
                request_id=request_id,
                workflow_node=node,
                inputs=inputs,
            )

        # Store tasks for this request
        self.request_tasks[request_id] = tasks
        self.request_completed[request_id] = asyncio.Event()

        for node_name, task in tasks.items():
            # If task has no dependencies, add it to task queue immediately
            if (
                self._is_task_ready(task)
                and node_name not in self.scheduled_nodes[request_id]
            ):
                task = self._prepare_task_inputs(task)
                self.scheduled_nodes[request_id].add(node_name)
                await self.task_queue.put(task)
                self.logger.debug(
                    f"Added task {node_name} to task queue (no dependencies)"
                )

        # Wait for request to complete
        await self.request_completed[request_id].wait()
        self.logger.info(f"Request {request_id} is complete")

        # Get results
        results = self.request_results[request_id]

        # Clean up
        del self.request_tasks[request_id]
        del self.request_results[request_id]
        del self.request_completed[request_id]
        del self.request_arrival_times[request_id]
        del self.node_prerequisites[request_id]
        del self.node_lazy_prerequisites[request_id]
        del self.node_non_lazy_prerequisites[request_id]
        del self.node_successors[request_id]
        del self.scheduled_nodes[request_id]
        del self.completed_nodes[request_id]
        del self.tensor_reference_count[request_id]
        if request_id in self.lazy_input_dependents:
            del self.lazy_input_dependents[request_id]
        del self.tensor_map[request_id]

        workflow_end_time = time.time()
        self.logger.info(
            f"Time taken to execute request {request_id} for workflow {workflow.name}: {workflow_end_time - workflow_start_time} seconds"
        )

        return results

    async def load_models_on_workers(self):
        """Load pre-configured models on all workers"""
        self.logger.info(f"Loading {len(self.preload_models)} models on all workers...")
        self.logger.info(f"Preload models: {self.preload_models}")

        # Load each model on each worker
        for model_id in self.preload_models:
            self.logger.info(f"Loading model {model_id} on all workers...")

            # Send model loading requests to all workers for this model
            for worker_rank in self.all_workers_info.keys():
                asyncio.create_task(self._load_model_on_worker(worker_rank, model_id))

    async def _load_model_on_worker(self, worker_rank: int, model_id: str):
        """Load a single model on a specific worker"""
        try:
            base_memory = self.runtime_profile.model_memory(model_id)
            total_peak_memory = self.runtime_profile.max_peak_total_memory(model_id)

            # Prepare model loading message
            message = {
                "type": "model_loading",
                "model_id": model_id,
                "gpu_memory_required": base_memory,
                "gpu_memory_used": total_peak_memory,
            }

            # Send model loading request
            self.task_sockets[worker_rank].send_json(message, flags=zmq.NOBLOCK)
            self.logger.debug(
                f"Sent model loading request for {model_id} to worker {worker_rank}"
            )
        except Exception as e:
            self.logger.error(
                f"Error sending model loading request for {model_id} to worker {worker_rank}: {e}"
            )
            raise e

    def cleanup(self):
        """Cleanup coordinator resources"""
        self.is_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()

        if self._gather_result_task:
            self._gather_result_task.cancel()

        # Stop all workers first with timeout
        for worker_rank, socket in self.task_sockets.items():
            try:
                # Use non-blocking send with retry
                for _ in range(3):
                    try:
                        socket.send_json({"type": "stop"}, zmq.NOBLOCK)
                        break
                    except zmq.Again:
                        time.sleep(0.1)
            except Exception as e:
                self.logger.error(
                    f"Error sending stop signal to worker_{worker_rank}: {e}"
                )

        # Give workers time to process stop signal
        time.sleep(3)

        # Send stop signal to all workers
        for worker_rank, socket in list(self.task_sockets.items()):
            try:
                socket.close(linger=1000)  # 1 second linger
                del self.task_sockets[worker_rank]
            except Exception as e:
                self.logger.error(
                    f"Error closing task socket for worker_{worker_rank}: {e}"
                )

        for worker_rank, socket in list(self.result_sockets.items()):
            try:
                socket.close(linger=1000)
                del self.result_sockets[worker_rank]
            except Exception as e:
                self.logger.error(
                    f"Error closing result socket for worker_{worker_rank}: {e}"
                )
