import asyncio

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
