# python workflow_hub/flux_schnell/register_txt2img_workflow.py
#
# One registration serves both with and without classifier-free guidance: the
# request's cfg_guidance_scale picks a branch, and only the taken branch's nodes
# reach the served graph.
#
# The whole loop sits inside the cond, not just the per-step choice, because the
# negative prompt's text encoders would otherwise be hoisted above the branch and
# run on every request -- an idle T5-XXL pass whenever guidance is off.
import argparse

from diflow.interface import (
    BenchmarkSpec,
    Workflow,
    cond,
    denoise_loop,
    register_workflow,
)
from diflow.operators import (
    CLIP_Flux,
    Config,
    Flux1Schnell,
    Flux1VAE,
    FluxLatentsGenerator,
    FluxSchnellFlowMatchEulerDiscreteScheduler,
    T5_Flux,
)
from diflow.operators.utils import default_model_path


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(
        name="flux_schnell_txt2img_workflow",
        benchmark=BenchmarkSpec(
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "negative_prompt": "",
                "cfg_guidance_scale": 1.0,
                "seed": 0,
                "num_inference_steps": 2,
                "guidance_scale": 0.0,
            }
        ),
    )

    # Define model nodes
    latents_generator = FluxLatentsGenerator()
    clip_flux = CLIP_Flux(Config(model_path=model_path))
    t5_flux = T5_Flux(Config(model_path=model_path))
    scheduler = FluxSchnellFlowMatchEulerDiscreteScheduler(
        Config(model_path=model_path)
    )
    flux = Flux1Schnell(Config(model_path=model_path))
    vae = Flux1VAE(Config(model_path=model_path))

    # Define inputs
    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    negative_prompt = workflow.add_input(name="negative_prompt", data_type=str)
    cfg_guidance_scale = workflow.add_input(name="cfg_guidance_scale", data_type=float)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    guidance_scale = workflow.add_input(name="guidance_scale", data_type=float)

    # Define connections
    # Generate latents and latent image IDs
    latents = latents_generator(height=height, width=width, seed=seed)

    # Text encoding. The positive prompt is needed either way; the negative one is
    # encoded inside the branch that uses it.
    clip_prompt_embeds = clip_flux(prompt=prompt)
    t5_prompt_embeds = t5_flux(prompt=prompt)

    # Denoising process
    loop = dict(
        model=flux,
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=num_inference_steps,
        prompt_embeds=t5_prompt_embeds,
        pooled_prompt_embeds=clip_prompt_embeds,
        # Flux's distilled guidance embedding, not classifier-free guidance.
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )

    def with_cfg():
        """Two model passes per step, on the negative and positive embeddings.

        denoise_loop builds its own per-step cond on the same predicate, which
        resolves the same way as this one. It emits no node either way.
        """
        return {
            "latents": denoise_loop(
                negative_prompt_embeds=t5_flux(prompt=negative_prompt),
                negative_pooled_prompt_embeds=clip_flux(prompt=negative_prompt),
                cfg_guidance_scale=cfg_guidance_scale,
                **loop,
            )
        }

    def without_cfg():
        """One pass per step, and no negative prompt encoded at all."""
        return {"latents": denoise_loop(**loop)}

    # Resolved when the request arrives. Comparing a request input builds the
    # predicate for you; writing `if cfg_guidance_scale > 1.0:` raises.
    denoised_latents = cond(cfg_guidance_scale > 1.0, with_cfg, without_cfg)["latents"]

    # VAE decode
    output_img = vae(
        latents=denoised_latents, mode="decode_latents", height=height, width=width
    )

    # Define outputs
    workflow.add_output(output_img, name="output_img")

    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_path_default = default_model_path("FLUX.1-schnell")
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
