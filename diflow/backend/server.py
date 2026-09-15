import argparse
import asyncio
import inspect
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    request_validation_exception_handler as default_validation_error_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diflow.api.catalog import operator_catalog, template_catalog
from diflow.api.errors import WorkflowAPIError
from diflow.api.prompt_skills import PromptSkillRegistry
from diflow.api.schemas import (
    PromptSkillApplyRequest,
    PromptSkillDefinition,
    PublicRunRequest,
    WorkflowSpec,
)
from diflow.api.workflow_compiler import CompilationResult, WorkflowCompiler
from diflow.backend.coordinator import Coordinator, ExecutionTimeoutError
from diflow.backend.scheduler import SchedulingPolicy
from diflow.interface.request import InferenceRequest
from diflow.interface.workflow import Workflow
from diflow.profiling.runtime_profile import RuntimeProfile, RuntimeProfileError

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
        model_registry: Optional[Mapping[str, str]] = None,
        prompt_skill_registry: Optional[PromptSkillRegistry] = None,
        max_registered_workflows: int = 64,
        max_run_records: int = 1024,
    ):
        # self.dist_config = dist_config
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.model_registry = dict(model_registry or {})
        self.workflow_compiler = WorkflowCompiler(self.model_registry)
        self.prompt_skills = prompt_skill_registry or PromptSkillRegistry()
        self.public_workflows: Dict[str, Dict[str, Any]] = {}
        self._public_workflow_digests: Dict[str, str] = {}
        self.public_runs: Dict[str, Dict[str, Any]] = {}
        self._run_tasks: Dict[str, asyncio.Task] = {}
        self._run_cancellations: set[str] = set()
        self.max_registered_workflows = max_registered_workflows
        self.max_run_records = max_run_records
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
        self,
        service_id: str,
        inputs: Dict[str, Any],
        slo_slack: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if service_id not in self.workflows:
            raise ValueError(f"Workflow {service_id} not found")

        workflow = self.workflows[service_id]["workflow"]
        # print(f"Running inference for workflow: {service_id}")

        # Generate unique request ID
        request_id = request_id or str(uuid.uuid4())

        # Execute workflow asynchronously
        return await self.coordinator.execute_workflow(
            request_id, workflow, inputs, slo_slack=slo_slack
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _compile_public_workflow(self, spec: WorkflowSpec) -> CompilationResult:
        result = self.workflow_compiler.compile(spec)
        try:
            self.coordinator.runtime_profile.validate_workflow(result.workflow)
        except RuntimeProfileError as exc:
            raise WorkflowAPIError(
                "PROFILE_NOT_AVAILABLE",
                str(exc),
                status_code=409,
            ) from exc
        return result

    def _resolved_preprocessors(self, spec: WorkflowSpec) -> List[Dict[str, Any]]:
        resolved = []
        for index, use in enumerate(spec.preprocessors):
            try:
                skill = self.prompt_skills.resolve(use.skill_id, use.version)
                # Validate parameters and choices without changing the registered graph.
                self.prompt_skills.apply(
                    skill.id, "", use.parameters, version=skill.version
                )
            except WorkflowAPIError as exc:
                if exc.path is None:
                    exc.path = f"preprocessors.{index}"
                raise
            resolved.append(
                {
                    "input": use.input,
                    "skill_id": skill.id,
                    "version": skill.version,
                    "parameters": dict(use.parameters),
                    "digest": self.prompt_skills.digest(skill),
                }
            )
        return resolved

    def validate_public_workflow(self, spec: WorkflowSpec) -> Dict[str, Any]:
        result = self._compile_public_workflow(spec)
        preprocessors = self._resolved_preprocessors(spec)
        return {
            "valid": True,
            "digest": result.digest,
            "source": result.source,
            "inputs": result.input_contract,
            "outputs": result.output_contract,
            "node_count": len(result.workflow.workflow_nodes),
            "region_count": len(result.workflow.regions),
            "preprocessors": preprocessors,
            "profile_status": "ready",
        }

    def register_public_workflow(self, spec: WorkflowSpec) -> Dict[str, Any]:
        digest = self.workflow_compiler.digest(spec)
        existing_id = self._public_workflow_digests.get(digest)
        if existing_id is not None:
            response = self.describe_public_workflow(existing_id)
            response["created"] = False
            return response
        if len(self.public_workflows) >= self.max_registered_workflows:
            raise WorkflowAPIError(
                "WORKFLOW_LIMIT",
                "The server workflow registration limit has been reached",
                status_code=429,
                retryable=True,
            )
        result = self._compile_public_workflow(spec)
        preprocessors = self._resolved_preprocessors(spec)
        workflow_id = f"{spec.name}-{result.digest[:12]}"
        self.workflows[workflow_id] = {
            "workflow": result.workflow,
            "config": dict(spec.metadata),
        }
        record = {
            "workflow_id": workflow_id,
            "name": spec.name,
            "digest": result.digest,
            "status": "ready",
            "source": result.source,
            "created_at": self._timestamp(),
            "spec": spec.model_dump(mode="json"),
            "inputs": result.input_contract,
            "outputs": result.output_contract,
            "preprocessors": preprocessors,
        }
        self.public_workflows[workflow_id] = record
        self._public_workflow_digests[result.digest] = workflow_id
        response = self.describe_public_workflow(workflow_id)
        response["created"] = True
        return response

    def describe_public_workflow(self, workflow_id: str) -> Dict[str, Any]:
        try:
            record = self.public_workflows[workflow_id]
        except KeyError as exc:
            raise WorkflowAPIError(
                "WORKFLOW_NOT_FOUND",
                f"Workflow {workflow_id!r} was not found",
                status_code=404,
            ) from exc
        return dict(record)

    def list_public_workflows(self) -> List[Dict[str, Any]]:
        return [
            self.describe_public_workflow(workflow_id)
            for workflow_id in sorted(self.public_workflows)
        ]

    @staticmethod
    def _input_value_matches(type_name: str, value: Any) -> bool:
        if type_name in ("str", "image"):
            return isinstance(value, str)
        if type_name == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if type_name == "bool":
            return isinstance(value, bool)
        # Tensor and other internal types are produced by nodes rather than JSON
        # request inputs. Unknown custom types are left to the worker serializer.
        return True

    def _prepare_public_inputs(
        self, workflow_id: str, supplied: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        record = self.public_workflows[workflow_id]
        contract = record["inputs"]
        unknown = sorted(set(supplied) - set(contract))
        if unknown:
            raise WorkflowAPIError(
                "UNKNOWN_WORKFLOW_INPUT",
                f"Unknown workflow inputs: {', '.join(unknown)}",
                path="inputs",
            )
        prepared = dict(supplied)
        for name, input_spec in contract.items():
            if name not in prepared:
                if input_spec["required"]:
                    raise WorkflowAPIError(
                        "MISSING_WORKFLOW_INPUT",
                        f"Workflow input {name!r} is required",
                        path=f"inputs.{name}",
                    )
                prepared[name] = input_spec["default"]
            if not self._input_value_matches(input_spec["type"], prepared[name]):
                raise WorkflowAPIError(
                    "INVALID_WORKFLOW_INPUT",
                    f"Workflow input {name!r} must be {input_spec['type']}",
                    path=f"inputs.{name}",
                )
        trace = []
        for preprocessor in record["preprocessors"]:
            input_name = preprocessor["input"]
            value = prepared[input_name]
            if not isinstance(value, str):
                raise WorkflowAPIError(
                    "INVALID_WORKFLOW_INPUT",
                    f"Prompt preprocessor input {input_name!r} must be a string",
                    path=f"inputs.{input_name}",
                )
            applied = self.prompt_skills.apply(
                preprocessor["skill_id"],
                value,
                preprocessor["parameters"],
                version=preprocessor["version"],
            )
            prepared[input_name] = applied["prompt"]
            trace.append(applied)
        return prepared, trace

    def start_public_run(
        self, workflow_id: str, request: PublicRunRequest
    ) -> Dict[str, Any]:
        if workflow_id not in self.public_workflows:
            raise WorkflowAPIError(
                "WORKFLOW_NOT_FOUND",
                f"Workflow {workflow_id!r} was not found",
                status_code=404,
            )
        if len(self.public_runs) >= self.max_run_records:
            terminal = {
                run_id: record
                for run_id, record in self.public_runs.items()
                if record["status"] in {"succeeded", "failed", "rejected", "cancelled"}
            }
            if terminal:
                oldest = min(
                    terminal, key=lambda run_id: terminal[run_id]["updated_at"]
                )
                del self.public_runs[oldest]
            else:
                raise WorkflowAPIError(
                    "RUN_LIMIT",
                    "The server run-record limit has been reached",
                    status_code=429,
                    retryable=True,
                )
        inputs, trace = self._prepare_public_inputs(workflow_id, request.inputs)
        run_id = str(uuid.uuid4())
        now = self._timestamp()
        record = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "prompt_trace": trace,
            "result": None,
            "error": None,
        }
        self.public_runs[run_id] = record
        task = asyncio.create_task(
            self._execute_public_run(run_id, inputs, request),
            name=f"diflow-run-{run_id}",
        )
        self._run_tasks[run_id] = task
        task.add_done_callback(lambda _: self._run_tasks.pop(run_id, None))
        return dict(record)

    async def _execute_public_run(
        self, run_id: str, inputs: Dict[str, Any], request: PublicRunRequest
    ) -> None:
        record = self.public_runs[run_id]
        record["status"] = "running"
        record["updated_at"] = self._timestamp()
        if request.timeout is None or request.profiled_latency is None:
            slo_slack = None
        else:
            slo_slack = request.timeout - request.profiled_latency
        try:
            record["result"] = await self.run_inference(
                record["workflow_id"],
                inputs,
                slo_slack=slo_slack,
                request_id=run_id,
            )
            record["status"] = "succeeded"
        except ExecutionTimeoutError as exc:
            record["status"] = "rejected"
            record["error"] = {
                "code": "SLO_REJECTED",
                "message": str(exc),
                "retryable": True,
            }
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["error"] = {
                "code": "RUN_CANCELLED",
                "message": "The run was cancelled",
                "retryable": False,
            }
        except Exception as exc:
            traceback.print_exc()
            record["status"] = "failed"
            record["error"] = {
                "code": "EXECUTION_FAILED",
                "message": str(exc),
                "retryable": False,
            }
        finally:
            if run_id in self._run_cancellations:
                record["result"] = None
                record["status"] = "cancelled"
                record["error"] = {
                    "code": "RUN_CANCELLED",
                    "message": "The run was cancelled",
                    "retryable": False,
                }
                self._run_cancellations.discard(run_id)
            record["updated_at"] = self._timestamp()

    def get_public_run(self, run_id: str) -> Dict[str, Any]:
        try:
            return dict(self.public_runs[run_id])
        except KeyError as exc:
            raise WorkflowAPIError(
                "RUN_NOT_FOUND",
                f"Run {run_id!r} was not found",
                status_code=404,
            ) from exc

    def cancel_public_run(self, run_id: str) -> Dict[str, Any]:
        record = self.get_public_run(run_id)
        if record["status"] == "queued":
            task = self._run_tasks.get(run_id)
            if task is not None:
                task.cancel()
            self.public_runs[run_id]["status"] = "cancelled"
            self.public_runs[run_id]["updated_at"] = self._timestamp()
            self.public_runs[run_id]["error"] = {
                "code": "RUN_CANCELLED",
                "message": "The run was cancelled",
                "retryable": False,
            }
        elif record["status"] == "running":
            # Coordinator requests cannot currently preempt already dispatched
            # GPU work safely. Keep awaiting it so coordinator state is cleaned,
            # then discard the result in _execute_public_run.
            self._run_cancellations.add(run_id)
            self.public_runs[run_id]["status"] = "cancelling"
            self.public_runs[run_id]["updated_at"] = self._timestamp()
        return self.get_public_run(run_id)

    async def shutdown(self):
        """Cleanup on service shutdown"""
        print("Shutting down workflow service...")
        tasks = list(self._run_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
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

    @app.exception_handler(WorkflowAPIError)
    async def workflow_api_error_handler(
        request: Request, exc: WorkflowAPIError
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        if not request.url.path.startswith("/api/v1/"):
            return await default_validation_error_handler(request, exc)
        first = errors[0] if errors else {}
        path = ".".join(str(item) for item in first.get("loc", ())[1:]) or None
        error = WorkflowAPIError(
            "SCHEMA_VALIDATION_FAILED",
            first.get("msg", "Request validation failed"),
            path=path,
            status_code=422,
            details={
                "errors": [
                    {
                        "type": item.get("type"),
                        "loc": item.get("loc"),
                        "msg": item.get("msg"),
                    }
                    for item in errors
                ]
            },
        )
        return JSONResponse(status_code=422, content=error.to_dict())

    @app.get("/api/v1/operators")
    async def list_operators():
        return {"operators": list(operator_catalog())}

    @app.get("/api/v1/models")
    async def list_models():
        # Deliberately expose aliases only. Filesystem paths are an administrator
        # concern and never enter an agent-authored WorkflowSpec.
        return {
            "models": [
                {"model_ref": model_ref}
                for model_ref in sorted(workflow_service.model_registry)
            ]
        }

    @app.get("/api/v1/workflow-templates")
    async def list_workflow_templates():
        return {"templates": template_catalog(set(workflow_service.model_registry))}

    @app.post("/api/v1/workflows/validate")
    async def validate_public_workflow(spec: WorkflowSpec):
        return workflow_service.validate_public_workflow(spec)

    @app.post("/api/v1/workflows", status_code=201)
    async def register_public_workflow(spec: WorkflowSpec):
        return workflow_service.register_public_workflow(spec)

    @app.get("/api/v1/workflows")
    async def list_public_workflows():
        return {"workflows": workflow_service.list_public_workflows()}

    @app.get("/api/v1/workflows/{workflow_id}")
    async def get_public_workflow(workflow_id: str):
        return workflow_service.describe_public_workflow(workflow_id)

    @app.post("/api/v1/workflows/{workflow_id}/runs", status_code=202)
    async def start_public_run(workflow_id: str, request: PublicRunRequest):
        return workflow_service.start_public_run(workflow_id, request)

    @app.get("/api/v1/runs/{run_id}")
    async def get_public_run(run_id: str):
        return workflow_service.get_public_run(run_id)

    @app.delete("/api/v1/runs/{run_id}")
    async def cancel_public_run(run_id: str):
        return workflow_service.cancel_public_run(run_id)

    @app.get("/api/v1/prompt-skills")
    async def list_prompt_skills():
        return {"prompt_skills": workflow_service.prompt_skills.list()}

    @app.post("/api/v1/prompt-skills", status_code=201)
    async def register_prompt_skill(skill: PromptSkillDefinition):
        return workflow_service.prompt_skills.register(skill)

    @app.post("/api/v1/prompt-skills/{skill_id}/apply")
    async def apply_prompt_skill(skill_id: str, request: PromptSkillApplyRequest):
        return workflow_service.prompt_skills.apply(
            skill_id,
            request.prompt,
            request.parameters,
            version=request.version,
        )

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
