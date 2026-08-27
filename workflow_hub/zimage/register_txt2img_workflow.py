# python workflow_hub/zimage/register_txt2img_workflow.py
import argparse

from diflow.interface import (
    BenchmarkSpec,
    DenoiseContext,
    Workflow,
    cond,
    denoise_loop,
    register_workflow,
)
from diflow.operators import (
    Config,
    Qwen3_ZImage,
    ZImage,
    ZImageFlowMatchEulerDiscreteScheduler,
    ZImageLatentsGenerator,
    ZImageVAE,
)
from diflow.operators.utils import default_model_path


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(
        name="zimage_txt2img_workflow",
        benchmark=BenchmarkSpec(
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "negative_prompt": "",
                "cfg_guidance_scale": 5.0,
                "seed": 0,
                "num_inference_steps": 2,
            },
            resolutions=((1024, 1024),),
            batch_sizes=(1,),
            profile_steps=2,
        ),
    )

    # Define model nodes
    config = Config(model_path=model_path)
    latents_generator = ZImageLatentsGenerator()
    text_encoder = Qwen3_ZImage(config)
    scheduler = ZImageFlowMatchEulerDiscreteScheduler(config)
    transformer = ZImage(config)
    vae = ZImageVAE(config)

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    negative_prompt = workflow.add_input(name="negative_prompt", data_type=str)
    cfg_guidance_scale = workflow.add_input(name="cfg_guidance_scale", data_type=float)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)

    # Define connections
    latents = latents_generator(height=height, width=width, seed=seed)
    prompt_embeds, attention_mask = text_encoder(prompt=prompt)
    loop = dict(
        model=transformer,
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=num_inference_steps,
        prompt_embeds=prompt_embeds,
        encoder_attention_mask=attention_mask,
        height=height,
        width=width,
    )

    def with_cfg():
        # Keep negative-prompt encoding inside the request-time branch so a
        # no-CFG request does not spend a second Qwen3 pass.
        negative_embeds, negative_mask = text_encoder(prompt=negative_prompt)

        def cfg_step(context: DenoiseContext):
            noise_pred_text, noise_pred_uncond = transformer(
                latents=context.latents,
                timestep=context.timestep,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                encoder_attention_mask=attention_mask,
                negative_encoder_attention_mask=negative_mask,
                mode="batch_cfg",
            )
            return scheduler(
                latents=context.latents,
                timestep=context.timestep,
                noise_pred_uncond=noise_pred_uncond,
                noise_pred_text=noise_pred_text,
                guidance_scale=cfg_guidance_scale,
                mode="step_classifier_free_guidance",
            )

        return {
            "latents": denoise_loop(
                step_fn=cfg_step,
                **loop,
            )
        }

    def without_cfg():
        return {"latents": denoise_loop(**loop)}

    denoised_latents = cond(cfg_guidance_scale > 0.0, with_cfg, without_cfg)["latents"]
    output_img = vae(latents=denoised_latents, mode="decode_latents")
    # Define outputs
    workflow.add_output(output_img, name="output_img")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_path_default = default_model_path("Z-Image")
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
