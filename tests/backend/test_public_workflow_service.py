import asyncio
import time

from fastapi.testclient import TestClient

from diflow.api.prompt_skills import PromptSkillRegistry
from diflow.api.schemas import PublicRunRequest, WorkflowSpec
from diflow.api.workflow_compiler import WorkflowCompiler
from diflow.backend.server import WorkflowService, create_app


class FakeRuntimeProfile:
    def validate_workflow(self, workflow):
        assert workflow.workflow_nodes


class FakeCoordinator:
    def __init__(self):
        self.runtime_profile = FakeRuntimeProfile()

    async def wait_for_workers_ready(self, timeout_seconds, health_check):
        return None

    async def run_scheduler(self):
        return None

    def cleanup(self):
        return None

    async def execute_workflow(self, request_id, workflow, inputs, slo_slack=None):
        return {
            "request_id": request_id,
            "prompt": inputs.get("prompt"),
            "slo_slack": slo_slack,
        }


def make_service():
    service = object.__new__(WorkflowService)
    service.workflows = {}
    service.model_registry = {}
    service.workflow_compiler = WorkflowCompiler({})
    service.prompt_skills = PromptSkillRegistry()
    service.public_workflows = {}
    service._public_workflow_digests = {}
    service.public_runs = {}
    service._run_tasks = {}
    service._run_cancellations = set()
    service.max_registered_workflows = 64
    service.max_run_records = 1024
    service.coordinator = FakeCoordinator()
    return service


def prompt_guidance_spec():
    return WorkflowSpec.model_validate(
        {
            "name": "agent-prompt",
            "inputs": {
                "prompt": {"type": "string"},
                "guidance_scale": {"type": "number"},
            },
            "preprocessors": [
                {
                    "input": "prompt",
                    "skill_id": "portrait-composition",
                    "version": "1.0.0",
                    "parameters": {"pose": "walking naturally"},
                }
            ],
            "nodes": [
                {
                    "id": "guidance",
                    "operator": "GuidanceTensor",
                    "inputs": {"guidance_scale": {"input": "guidance_scale"}},
                }
            ],
            "outputs": {"guidance": {"node": "guidance", "output": "guidance_tensor"}},
        }
    )


def test_registration_is_content_idempotent():
    service = make_service()
    first = service.register_public_workflow(prompt_guidance_spec())
    second = service.register_public_workflow(prompt_guidance_spec())

    assert first["workflow_id"] == second["workflow_id"]
    assert first["created"] is True
    assert second["created"] is False
    assert len(service.public_workflows) == 1


def test_async_run_applies_pinned_prompt_skill_and_preserves_trace():
    async def scenario():
        service = make_service()
        workflow = service.register_public_workflow(prompt_guidance_spec())
        started = service.start_public_run(
            workflow["workflow_id"],
            PublicRunRequest(
                inputs={"prompt": "a traveler", "guidance_scale": 3.5},
                timeout=10,
                profiled_latency=2,
            ),
        )
        await service._run_tasks[started["run_id"]]
        completed = service.get_public_run(started["run_id"])

        assert completed["status"] == "succeeded"
        assert "walking naturally" in completed["result"]["prompt"]
        assert completed["result"]["slo_slack"] == 8
        assert completed["prompt_trace"][0]["original_prompt"] == "a traveler"
        assert completed["prompt_trace"][0]["skill_version"] == "1.0.0"

    asyncio.run(scenario())


def test_running_cancel_waits_for_coordinator_cleanup_and_discards_result():
    async def scenario():
        service = make_service()
        started_execution = asyncio.Event()
        finish_execution = asyncio.Event()

        async def blocking_execute(request_id, workflow, inputs, slo_slack=None):
            started_execution.set()
            await finish_execution.wait()
            return {"guidance": [inputs["guidance_scale"]]}

        service.coordinator.execute_workflow = blocking_execute
        workflow = service.register_public_workflow(prompt_guidance_spec())
        started = service.start_public_run(
            workflow["workflow_id"],
            PublicRunRequest(inputs={"prompt": "a traveler", "guidance_scale": 3.5}),
        )
        await started_execution.wait()

        cancelling = service.cancel_public_run(started["run_id"])
        assert cancelling["status"] == "cancelling"
        assert not service._run_tasks[started["run_id"]].cancelled()

        finish_execution.set()
        await service._run_tasks[started["run_id"]]
        completed = service.get_public_run(started["run_id"])
        assert completed["status"] == "cancelled"
        assert completed["result"] is None
        assert completed["error"]["code"] == "RUN_CANCELLED"

    asyncio.run(scenario())


def test_http_workflow_lifecycle_is_agent_friendly():
    service = make_service()
    app = create_app(service)
    payload = prompt_guidance_spec().model_dump(mode="json")

    with TestClient(app) as client:
        validation = client.post("/api/v1/workflows/validate", json=payload)
        assert validation.status_code == 200
        assert validation.json()["valid"] is True

        registration = client.post("/api/v1/workflows", json=payload)
        assert registration.status_code == 201
        workflow_id = registration.json()["workflow_id"]

        duplicate = client.post("/api/v1/workflows", json=payload)
        assert duplicate.json()["workflow_id"] == workflow_id
        assert duplicate.json()["created"] is False

        started = client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {"prompt": "a traveler", "guidance_scale": 3.5}},
        )
        assert started.status_code == 202
        run_id = started.json()["run_id"]

        for _ in range(50):
            completed = client.get(f"/api/v1/runs/{run_id}").json()
            if completed["status"] == "succeeded":
                break
            time.sleep(0.01)
