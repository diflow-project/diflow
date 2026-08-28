# python workflow_hub/zimage_turbo/register_txt2img_workflow.py
import argparse

from diflow.interface import BenchmarkSpec, Workflow, denoise_loop, register_workflow
from diflow.operators import (
    Config,
    Qwen3_ZImage,
    ZImageLatentsGenerator,
    ZImageTurbo,
    ZImageTurboFlowMatchEulerDiscreteScheduler,
    ZImageVAE,
)
from diflow.operators.utils import default_model_path


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(
        name="zimage_turbo_txt2img_workflow",
        benchmark=BenchmarkSpec(
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "seed": 0,
                "num_inference_steps": 2,
            },
            profile_steps=2,
        ),
    )

    # Define model nodes
    config = Config(model_path=model_path)
    latents_generator = ZImageLatentsGenerator()
    text_encoder = Qwen3_ZImage(config)
    scheduler = ZImageTurboFlowMatchEulerDiscreteScheduler(config)
    transformer = ZImageTurbo(config)
    vae = ZImageVAE(config)

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)

    # Define connections
    latents = latents_generator(height=height, width=width, seed=seed)
    prompt_embeds, attention_mask = text_encoder(prompt=prompt)
    # Turbo is distilled for guidance_scale=0: there is no negative encoder pass
    # or CFG branch, and each requested step performs one transformer call.
    denoised_latents = denoise_loop(
        model=transformer,
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=num_inference_steps,
        prompt_embeds=prompt_embeds,
        encoder_attention_mask=attention_mask,
        height=height,
        width=width,
    )
    output_img = vae(latents=denoised_latents, mode="decode_latents")

    # Define outputs
    workflow.add_output(output_img, name="output_img")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_path_default = default_model_path("Z-Image-Turbo")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--model-path",
        type=str,
        default=model_path_default,
        required=model_path_default is None,
    )
    args = parser.parse_args()
    server_url = args.server_url

    # Register workflow
    service_id = register_workflow(
        workflow=create_workflow(model_path=args.model_path),
        server_url=server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
