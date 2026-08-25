import argparse
import inspect
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from diflow.backend.coordinator import Coordinator, ExecutionTimeoutError
from diflow.backend.scheduler import SchedulingPolicy
from diflow.interface.request import InferenceRequest
from diflow.interface.workflow import Workflow
from diflow.profiling.runtime_profile import RuntimeProfile

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class WorkflowService:
    def __init__(
        self,
        worker_hostnames: List[str],
        scheduling_policy: SchedulingPolicy,
        base_port: int,
        preload_models_config: str,
        model_batch_config: str,
        runtime_profile: RuntimeProfile,
        enable_early_abort: bool = False,
    ):
        # self.dist_config = dist_config
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.coordinator = Coordinator(
            worker_hostnames=worker_hostnames,
            scheduling_policy=scheduling_policy,
            base_port=base_port,
            preload_models_config=preload_models_config,
            model_batch_config=model_batch_config,
            enable_early_abort=enable_early_abort,
            runtime_profile=runtime_profile,
        )

    async def startup(
        self,
        timeout_seconds: float = 60,
        worker_health_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialize the distributed system on service startup"""
        print("Starting workflow service")

        # Wait for all workers to be ready before accepting requests
        await self.coordinator.wait_for_workers_ready(
            timeout_seconds=timeout_seconds,
            health_check=worker_health_check,
        )

        # Run the scheduler
        await self.coordinator.run_scheduler()

        print("Workflow service ready")

    def register_workflow(
        self, workflow_dict: Dict[str, Any], service_config: Dict
    ) -> str:
        service_id = f"{workflow_dict['name']}"
        print(f"Registering workflow: {service_id}")
        if service_id in self.workflows:
            print(f"Workflow {service_id} already registered")
            return service_id

        # Convert the raw dict to a Workflow
        workflow = Workflow.from_dict(workflow_dict)
        self.coordinator.runtime_profile.validate_workflow(workflow)
        # print(f"{workflow}")
        self.workflows[service_id] = {"workflow": workflow, "config": service_config}

        return service_id

    async def run_inference(
        self, service_id: str, inputs: Dict[str, Any], slo_slack: Optional[float] = None
    ) -> Dict[str, Any]:
        if service_id not in self.workflows:
            raise ValueError(f"Workflow {service_id} not found")

        workflow = self.workflows[service_id]["workflow"]
        # print(f"Running inference for workflow: {service_id}")

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Execute workflow asynchronously
        return await self.coordinator.execute_workflow(
            request_id, workflow, inputs, slo_slack=slo_slack
        )

    async def shutdown(self):
        """Cleanup on service shutdown"""
        print("Shutting down workflow service...")
        self.coordinator.cleanup()
        print("Workflow service shutdown complete")


class WorkflowRegistration(BaseModel):
    workflow: str
    service_config: Dict[str, Any]


def create_app(
    workflow_service: WorkflowService,
    *,
    initial_workflow: Optional[Workflow] = None,
    service_config: Optional[Dict[str, Any]] = None,
    startup_timeout: float = 60,
    worker_health_check: Optional[Callable[[], bool]] = None,
    on_ready: Optional[Callable[[Optional[str]], None]] = None,
    on_shutdown: Optional[Callable[[], Any]] = None,
) -> FastAPI:
    """Create a FastAPI application backed by ``workflow_service``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await workflow_service.startup(
                timeout_seconds=startup_timeout,
                worker_health_check=worker_health_check,
            )
            service_id = None
            if initial_workflow is not None:
                service_id = workflow_service.register_workflow(
                    initial_workflow.to_dict(), service_config or {}
                )
            app.state.workflow_service = workflow_service
            app.state.service_id = service_id
            if on_ready is not None:
                on_ready(service_id)
            yield
        finally:
            try:
                await workflow_service.shutdown()
            finally:
                if on_shutdown is not None:
                    result = on_shutdown()
                    if inspect.isawaitable(result):
                        await result

    app = FastAPI(lifespan=lifespan)

    @app.post("/api/workflow/register")
    async def register_workflow(registration: WorkflowRegistration):
        try:
            workflow_dict = json.loads(registration.workflow)
            service_id = workflow_service.register_workflow(
                workflow_dict, registration.service_config
            )
            return {
                "status": "success",
                "service_id": service_id,
                "message": f"Workflow '{service_id}' registered successfully",
            }
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/workflow/{service_id}/inference")
    async def run_inference(service_id: str, request: InferenceRequest):
        try:
            if request.timeout is None or request.profiled_latency is None:
                slo_slack = None
            else:
                slo_slack = request.timeout - request.profiled_latency

            results = await workflow_service.run_inference(
                service_id, request.inputs, slo_slack=slo_slack
            )
            return {"status": "success", "results": results}
        except ExecutionTimeoutError as e:
            # Early abort is an admission-control decision, not an internal error.
            return {"status": "rejected", "error": str(e)}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))

    return app


def _read_worker_hostnames(hostfile: str) -> List[str]:
    with open(hostfile, "r") as file:
        return [line.strip() for line in file if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-port", type=int, default=14000)
    parser.add_argument("--hostfile", type=str, default="hostfile")
    parser.add_argument(
        "--scheduling-policy",
        type=str,
        default="dynamic",
        choices=["exclusive", "random", "dynamic"],
    )
    parser.add_argument(
        "--preload-models-config",
        type=str,
        default=str(DEFAULT_CONFIG_DIR / "preload_models.yaml"),
        help="Path to preload models YAML config file",
    )
    parser.add_argument(
        "--model-batch-config",
        type=str,
        default=str(DEFAULT_CONFIG_DIR / "model_batch.json"),
        help="Path to model batch JSON config file",
    )
    parser.add_argument(
        "--enable-early-abort",
        action="store_true",
        help="Reject requests early when estimated inflight work exceeds SLO slack",
    )
    parser.add_argument(
        "--runtime-profile",
        required=True,
        help="Schema-v2 runtime profile produced by automatic benchmarking",
    )
    args = parser.parse_args(argv)

    worker_hostnames = _read_worker_hostnames(args.hostfile)
    print(f"Worker hostnames: {worker_hostnames}")

    # Initialize the service
    workflow_service = WorkflowService(
        worker_hostnames=worker_hostnames,
        scheduling_policy=SchedulingPolicy(args.scheduling_policy),
        base_port=args.base_port,
        preload_models_config=args.preload_models_config,
        model_batch_config=args.model_batch_config,
        enable_early_abort=args.enable_early_abort,
        runtime_profile=RuntimeProfile.from_file(args.runtime_profile),
    )

    app = create_app(workflow_service)
    try:
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            loop="asyncio",
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30,
        )
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error during server execution: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
