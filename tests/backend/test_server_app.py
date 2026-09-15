import asyncio

from fastapi.testclient import TestClient

from diflow.backend.server import create_app
from diflow.interface.workflow import Workflow


class FakeWorkflowService:
    def __init__(self):
        self.events = []

    async def startup(self, timeout_seconds, worker_health_check):
        self.events.append(("startup", timeout_seconds, worker_health_check()))

    def register_workflow(self, workflow_dict, service_config):
        self.events.append(("register", workflow_dict["name"], service_config))
        return workflow_dict["name"]

    async def shutdown(self):
        self.events.append(("shutdown",))


def test_create_app_starts_registers_and_stops_service():
    service = FakeWorkflowService()
    ready = []
    app = create_app(
        service,
        initial_workflow=Workflow("test-workflow"),
        service_config={"mode": "test"},
        startup_timeout=12,
        worker_health_check=lambda: True,
        on_ready=ready.append,
        on_shutdown=lambda: service.events.append(("worker-stop",)),
    )

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            assert app.state.service_id == "test-workflow"

    asyncio.run(run_lifespan())

    assert service.events == [
        ("startup", 12, True),
        ("register", "test-workflow", {"mode": "test"}),
        ("shutdown",),
        ("worker-stop",),
    ]
    assert ready == ["test-workflow"]


def test_create_app_exposes_agent_workflow_api_routes():
    app = create_app(FakeWorkflowService())
    paths = {route.path for route in app.routes}

    assert "/api/v1/operators" in paths
    assert "/api/v1/workflows/validate" in paths
    assert "/api/v1/workflows/{workflow_id}/runs" in paths
    assert "/api/v1/prompt-skills/{skill_id}/apply" in paths


def test_v1_schema_errors_are_machine_readable_and_old_errors_stay_compatible():
    app = create_app(FakeWorkflowService(), worker_health_check=lambda: True)

    with TestClient(app) as client:
        response = client.post("/api/v1/workflows", json={"name": "invalid"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "SCHEMA_VALIDATION_FAILED"

        response = client.post("/api/workflow/missing/inference", json={})
        assert response.status_code == 422
        assert "detail" in response.json()
