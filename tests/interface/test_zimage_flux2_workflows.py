import base64
import io
import json

import pytest
from PIL import Image

from diflow.interface.benchmark import (
    DEFAULT_BENCHMARK_BATCH_SIZES,
    DEFAULT_BENCHMARK_RESOLUTIONS,
)
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_expand import expand_workflow
from tests.interface.hub_workflows import build_workflow, make_inputs
from workflow_hub import (
    run_flux2_klein_workflow,
    run_zimage_turbo_workflow,
    run_zimage_workflow,
)


@pytest.mark.parametrize(
    "name",
    [
        "zimage.register_txt2img_workflow",
        "zimage_turbo.register_txt2img_workflow",
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


@pytest.mark.parametrize(
    "name",
    [
        "zimage.register_txt2img_workflow",
        "zimage_turbo.register_txt2img_workflow",
        "flux2_klein.register_txt2img_workflow",
    ],
)
def test_new_workflows_profile_supported_shapes_and_dynamic_batches(name):
    benchmark = build_workflow(name).benchmark

    assert benchmark.resolutions == DEFAULT_BENCHMARK_RESOLUTIONS
    assert benchmark.batch_sizes == DEFAULT_BENCHMARK_BATCH_SIZES


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


def test_zimage_turbo_is_always_one_transformer_pass_per_step():
    workflow = build_workflow("zimage_turbo.register_txt2img_workflow")
    inputs = make_inputs(workflow, num_inference_steps=3, guidance_scale=7.5)
    graph = expand_workflow(workflow, inputs)

    assert set(workflow.inputs) == {
        "seed",
        "prompt",
        "height",
        "width",
        "num_inference_steps",
    }
    assert sum(node.op.id == "Qwen3_ZImage" for node in graph.workflow_nodes) == 1
    assert sum(node.op.id == "ZImageTurbo" for node in graph.workflow_nodes) == 3
    assert all(
        node.mode != "step_classifier_free_guidance" for node in graph.workflow_nodes
    )


def test_migrated_clients_preserve_reference_request_parameters():
    assert run_zimage_workflow.build_inputs(4.0) == {
        "prompt": run_zimage_workflow.PROMPT,
        "negative_prompt": "",
        "cfg_guidance_scale": 4.0,
        "num_inference_steps": 50,
        "seed": 0,
        "height": 1024,
        "width": 1024,
    }
    assert run_zimage_turbo_workflow.build_inputs()["num_inference_steps"] == 9
    assert run_flux2_klein_workflow.build_inputs()["num_inference_steps"] == 4


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (run_zimage_workflow, ["zimage_image_0.png", "zimage_image_1.png"]),
        (
            run_zimage_turbo_workflow,
            ["zimage_turbo_image_0.png", "zimage_turbo_image_1.png"],
        ),
        (
            run_flux2_klein_workflow,
            ["flux2_klein_image_0.png", "flux2_klein_image_1.png"],
        ),
    ],
)
def test_migrated_clients_process_all_images_under_output_dir(
    client, expected, tmp_path, monkeypatch
):
    image = Image.new("RGB", (2, 3), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    monkeypatch.setattr(client, "OUTPUT_DIR", tmp_path)

    client.process_response(
        {"status": "success", "results": {"output_img": [encoded, encoded]}}
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == expected
