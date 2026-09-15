import pytest
from pydantic import ValidationError

from diflow.api.catalog import operator_catalog
from diflow.api.errors import WorkflowAPIError
from diflow.api.schemas import WorkflowSpec
from diflow.api.workflow_compiler import WorkflowCompiler


def guidance_spec(**overrides):
    payload = {
        "name": "guidance-workflow",
        "inputs": {"guidance_scale": {"type": "number"}},
        "nodes": [
            {
                "id": "guidance",
                "operator": "GuidanceTensor",
                "inputs": {"guidance_scale": {"input": "guidance_scale"}},
            }
        ],
        "outputs": {"guidance": {"node": "guidance", "output": "guidance_tensor"}},
    }
    payload.update(overrides)
    return WorkflowSpec.model_validate(payload)


def test_compiles_public_dag_with_stable_node_names():
    result = WorkflowCompiler({}).compile(guidance_spec())

    assert result.source == "dag"
    assert [node.name for node in result.workflow.workflow_nodes] == [
        "GuidanceTensor_guidance"
    ]
    assert result.workflow.outputs == {
        "GuidanceTensor_guidance:guidance_tensor": "guidance"
    }
    assert result.input_contract["guidance_scale"]["type"] == "float"


def test_rejects_type_mismatch_with_machine_path():
    spec = guidance_spec(inputs={"guidance_scale": {"type": "string"}})

    with pytest.raises(WorkflowAPIError) as error:
        WorkflowCompiler({}).compile(spec)

    assert error.value.code == "TYPE_MISMATCH"
    assert error.value.path == "nodes.guidance.inputs.guidance_scale"


def test_rejects_cycles_before_instantiating_nodes():
    spec = WorkflowSpec.model_validate(
        {
            "name": "cycle",
            "nodes": [
                {
                    "id": "a",
                    "operator": "GuidanceTensor",
                    "inputs": {
                        "guidance_scale": {
                            "node": "b",
                            "output": "guidance_tensor",
                        }
                    },
                },
                {
                    "id": "b",
                    "operator": "GuidanceTensor",
                    "inputs": {
                        "guidance_scale": {
                            "node": "a",
                            "output": "guidance_tensor",
                        }
                    },
                },
            ],
            "outputs": {"guidance": {"node": "b", "output": "guidance_tensor"}},
        }
    )

    with pytest.raises(WorkflowAPIError) as error:
        WorkflowCompiler({}).compile(spec)

    assert error.value.code == "WORKFLOW_CYCLE"


def test_rejects_raw_model_path_in_public_schema():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(
            {
                "name": "unsafe",
                "inputs": {"prompt": {"type": "string"}},
                "nodes": [
                    {
                        "id": "clip",
                        "operator": "CLIP_Flux",
                        "model_path": "/root/models/private",
                        "inputs": {"prompt": {"input": "prompt"}},
                    }
                ],
                "outputs": {"embedding": {"node": "clip", "output": "prompt_embeds"}},
            }
        )


def test_builtin_template_uses_server_model_ref():
    spec = WorkflowSpec.model_validate(
        {
            "name": "agent-flux",
            "template": {"id": "flux-schnell", "model_ref": "flux"},
            "inputs": {
                "negative_prompt": {
                    "type": "string",
                    "required": False,
                    "default": "",
                }
            },
        }
    )

    result = WorkflowCompiler({"flux": "/models/flux"}).compile(spec)

    assert result.source == "template:flux-schnell"
    assert result.workflow.name == "agent-flux"
    assert result.workflow.regions
    assert result.input_contract["negative_prompt"]["default"] == ""


def test_public_operator_catalog_contains_only_available_operators():
    unavailable = [item["id"] for item in operator_catalog() if not item["available"]]
