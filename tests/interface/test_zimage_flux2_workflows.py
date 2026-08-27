import json

import pytest

from diflow.interface.workflow import Workflow
from diflow.interface.workflow_expand import expand_workflow
from tests.interface.hub_workflows import build_workflow, make_inputs


@pytest.mark.parametrize(
    "name",
    [
        "zimage.register_txt2img_workflow",
        "flux2_klein.register_txt2img_workflow",
    ],
)
def test_new_workflows_build_round_trip_and_expand(name):
    workflow = build_workflow(name)
    restored = Workflow.from_dict(json.loads(workflow.to_json()))
    inputs = make_inputs(restored, num_inference_steps=2, guidance_scale=7.5)
    graph = expand_workflow(restored, inputs)
    assert graph.outputs
    assert len(graph.workflow_nodes) > 0


def _expanded_zimage(scale):
    workflow = build_workflow("zimage.register_txt2img_workflow")
    inputs = make_inputs(workflow, num_inference_steps=2, guidance_scale=scale)
    return expand_workflow(workflow, inputs)


def test_zimage_cfg_threshold_and_negative_encoder_are_request_conditional():
    without_cfg = _expanded_zimage(0.0)
    with_cfg = _expanded_zimage(0.5)

    assert sum(node.op.id == "Qwen3_ZImage" for node in without_cfg.workflow_nodes) == 1
    assert sum(node.op.id == "ZImage" for node in without_cfg.workflow_nodes) == 2
    assert sum(node.op.id == "Qwen3_ZImage" for node in with_cfg.workflow_nodes) == 2
    assert sum(node.op.id == "ZImage" for node in with_cfg.workflow_nodes) == 4

    model_nodes = [node for node in with_cfg.workflow_nodes if node.op.id == "ZImage"]
    assert all("encoder_attention_mask" in node.get_inputs() for node in model_nodes)
    assert (
        sum(
            node.mode == "step_classifier_free_guidance"
            for node in with_cfg.workflow_nodes
        )
        == 2
    )


def test_distilled_flux2_klein_is_always_one_transformer_pass_per_step():
    workflow = build_workflow("flux2_klein.register_txt2img_workflow")
    inputs = make_inputs(workflow, num_inference_steps=3, guidance_scale=7.5)
    graph = expand_workflow(workflow, inputs)
    assert sum(node.op.id == "Flux2Klein" for node in graph.workflow_nodes) == 3
    assert all(
        node.mode != "step_classifier_free_guidance" for node in graph.workflow_nodes
    )
