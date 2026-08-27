# python workflow_hub/flux2_klein/register_txt2img_workflow.py
import argparse

from diflow.interface import BenchmarkSpec, Workflow, denoise_loop, register_workflow
from diflow.operators import (
    Config,
    Flux2FlowMatchEulerDiscreteScheduler,
    Flux2Klein,
    Flux2LatentsGenerator,
    Flux2VAE,
    Qwen3_Flux2Klein,
)
from diflow.operators.utils import default_model_path


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(
        name="flux2klein_txt2img_workflow",
        benchmark=BenchmarkSpec(
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "seed": 0,
                "num_inference_steps": 2,
            },
            resolutions=((1024, 1024),),
            batch_sizes=(1,),
            profile_steps=2,
        ),
    )

    config = Config(model_path=model_path)
    seed = workflow.add_input("seed", int)
    prompt = workflow.add_input("prompt", str)
    height = workflow.add_input("height", int)
    width = workflow.add_input("width", int)
    num_inference_steps = workflow.add_input("num_inference_steps", int)

    latents = Flux2LatentsGenerator()(height=height, width=width, seed=seed)
    prompt_embeds = Qwen3_Flux2Klein(config)(prompt=prompt)
    # The shipped Klein checkpoint has is_distilled=true, so its reference
    # pipeline ignores guidance and performs exactly one transformer call.
    denoised_latents = denoise_loop(
        model=Flux2Klein(config),
        scheduler=Flux2FlowMatchEulerDiscreteScheduler(config),
        latents=latents,
        num_inference_steps=num_inference_steps,
        prompt_embeds=prompt_embeds,
        height=height,
        width=width,
    )
    output_img = Flux2VAE(config)(
        latents=denoised_latents,
        height=height,
        width=width,
        mode="decode_latents",
    )
    workflow.add_output(output_img, name="output_img")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_path_default = default_model_path("Flux.2-klein")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument(
        "--model-path",
        default=model_path_default,
        required=model_path_default is None,
    )
    args = parser.parse_args()
    service_id = register_workflow(
        create_workflow(args.model_path), server_url=args.server_url
    )
    print(f"Registered workflow with service ID: {service_id}")
