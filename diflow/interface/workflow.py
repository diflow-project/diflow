import functools
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import aiohttp
import requests

from diflow.interface.benchmark import BenchmarkSpec
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.region import Region
from diflow.interface.request import InferenceRequest
from diflow.interface.workflow_context import WorkflowContext
from diflow.interface.workflow_node import WorkflowNode


class Workflow:
    def __init__(self, name: str, benchmark: Optional[BenchmarkSpec] = None):
        self.name = name
        self.benchmark = benchmark
        self.workflow_nodes: List[WorkflowNode] = []
        self.inputs: Dict[str, NodeIO] = {}
        self.outputs: Dict[str, str] = {}
        # Control-flow regions, expanded into workflow_nodes per request. Kept in
        # a separate list from workflow_nodes so that expansion happens after all
        # plain nodes have been emitted, regardless of authoring order.
        self.regions: List[Region] = []
        WorkflowContext.set_current_workflow(self)

    def __repr__(self):
        return f"""
        Workflow(
            name={self.name},
            workflow_nodes={self.workflow_nodes},
            inputs={self.inputs},
            outputs={self.outputs},
            regions={self.regions},
        )
        """

    def add_workflow_node(self, workflow_node: WorkflowNode) -> None:
        self.workflow_nodes.append(workflow_node)

    def add_region(self, region: Region) -> None:
        self.regions.append(region)

    def add_input(self, name: str, data_type: type) -> NodeIO:
        self.inputs[name] = NodeIO(
            name=name, data_type=data_type, source_type=SourceType.INPUT
        )
        return self.inputs[name]

    def add_output(self, node_output: NodeIO, name: str) -> None:
        self.outputs[node_output.name] = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.workflow_nodes],
            "outputs": {
                node_output: output_name
                for node_output, output_name in self.outputs.items()
            },
            "regions": [region.to_dict() for region in self.regions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, workflow_dict: Dict[str, Any]) -> "Workflow":
        try:
            workflow = cls(name=workflow_dict["name"])

            # Add workflow nodes
            for node_dict in workflow_dict["nodes"]:
                workflow_node = WorkflowNode.from_dict(node_dict)

                for _, input_io in workflow_node.get_inputs().items():
                    if input_io.source_type == SourceType.INPUT:
                        workflow.add_input(input_io.name, input_io.data_type)

                workflow.add_workflow_node(workflow_node)

            # Add control-flow regions
            for region_dict in workflow_dict.get("regions", []):
                workflow.add_region(Region.from_dict(region_dict))

            # Declare the request inputs read from inside region bodies too.
            # Induction variables look like inputs but are synthesized during
            # expansion, so they are not request inputs.
            placeholders = set()
            for region in workflow.regions:
                placeholders |= region.placeholder_names()
            for region in workflow.regions:
                for program in region.subprograms():
                    for node in program.iter_nodes():
                        for input_io in node.get_inputs().values():
                            if input_io is None:
                                continue
                            if input_io.source_type != SourceType.INPUT:
                                continue
                            if input_io.name in placeholders:
                                continue
                            workflow.add_input(input_io.name, input_io.data_type)

            # Copy the output mapping verbatim. Rediscovering it by scanning node
            # outputs would silently drop any output produced inside a region
            # body, since those nodes do not exist until expansion.
            workflow.outputs = dict(workflow_dict["outputs"])

            if workflow_dict.get("denoise_nodes"):
                raise ValueError(
                    "this workflow was registered by a client that predates "
                    "control flow: it carries denoise_nodes, which are no longer "
                    "expanded. Re-register with a current client."
                )

            return workflow
        except Exception as e:
            print(f"Error creating workflow from dict: {e}")
            raise e


def register_workflow(
    workflow: Workflow,
    server_url: str = "http://localhost:8000",
    service_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Register a workflow directly with the backend service

    Args:
        workflow: The workflow to register
        server_url: The URL of the backend service
        service_config: Optional service configuration

    Returns:
        The service ID of the registered workflow
    """
    # Convert workflow to JSON
    workflow_json = workflow.to_json()
    print(f"Registering workflow: {workflow.name}")

    # Send workflow to backend service
    response = requests.post(
        f"{server_url}/api/workflow/register",
        json={
            "workflow": workflow_json,
            "service_config": service_config or {},
        },
    )

    if response.status_code != 200:
        raise Exception(f"Failed to register workflow: {response.text}")
    return response.json()["service_id"]


def run_inference(
    service_id: str, inputs: Dict[str, Any], server_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Run inference on a registered workflow"""
    response = requests.post(
        f"{server_url}/api/workflow/{service_id}/inference", json={"inputs": inputs}
    )

    if response.status_code != 200:
        raise Exception(f"Failed to run inference: {response.text}")
    return response.json()


async def run_inference_async(
    service_id: str,
    request: InferenceRequest,
    session: aiohttp.ClientSession,
    server_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run inference on a registered workflow asynchronously

    Args:
        service_id: The workflow/service identifier
        request: InferenceRequest instance containing inputs and optionally timeout
        session: aiohttp session
        server_url: Server URL

    Returns:
        Dict containing 'response_json' and 'latency' keys
    """
    try:
        start_time = time.time()

        payload = request.model_dump(exclude_none=True)

        logging.debug(
            f"Starting inference for service {service_id} with request: {payload}"
        )

        async with session.post(
            f"{server_url}/api/workflow/{service_id}/inference",
            json=payload,
        ) as response:
            result = await response.json()
            end_time = time.time()
            latency = end_time - start_time
            logging.debug(
                f"Completed inference for service {service_id} in {latency:.2f}s"
            )
            return {"response_json": result, "latency": latency}
    except Exception as e:
        logging.error(f"Error in inference for service {service_id}: {e}")
        logging.error(traceback.format_exc())
        raise e
