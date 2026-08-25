# python workflow_hub/flux_schnell/register_txt2img_helper_workflow.py
#
# denoise_loop with a step_fn: you write the per-step computation, the loop keeps
# everything around it. Use it when the standard loop is almost what you want.
#
#   denoise_loop(...)            -> register_txt2img_workflow.py
#   denoise_loop(step_fn=...)    -> here
#   for_range / cond directly    -> register_txt2img_control_flow_workflow.py
#
# What you do not write is the scaffolding: initialising the scheduler, indexing
# the timestep schedule with the loop counter, and threading the latents from one
# iteration to the next. That part is fiddly and has nothing to do with what a
# given pipeline wants to change, so `context` arrives with this step's latents
# and timestep already resolved.
#
# Note where the classifier-free guidance choice is made: once, in a cond around
# the whole loop, so each branch's step is fixed and the negative prompt is only
# encoded when it is used. Deciding it per step instead would put the encoders
# above the branch, where they run on every request.
#
# The steps below are what denoise_loop would have built anyway, spelled out so
# there is something concrete to edit. That they come out identical to the
# standard loop is asserted in tests/interface/test_denoise_helpers.py.
import argparse

from diflow.interface import (
    DenoiseContext,
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
    workflow = Workflow(name="flux_schnell_txt2img_helper_workflow")

    latents_generator = FluxLatentsGenerator()
    clip_flux = CLIP_Flux(Config(model_path=model_path))
    t5_flux = T5_Flux(Config(model_path=model_path))
    scheduler = FluxSchnellFlowMatchEulerDiscreteScheduler(
        Config(model_path=model_path)
    )
    flux = Flux1Schnell(Config(model_path=model_path))
    vae = Flux1VAE(Config(model_path=model_path))

    seed = workflow.add_input(name="seed", data_type=int)
    prompt = workflow.add_input(name="prompt", data_type=str)
    negative_prompt = workflow.add_input(name="negative_prompt", data_type=str)
    cfg_guidance_scale = workflow.add_input(name="cfg_guidance_scale", data_type=float)
    height = workflow.add_input(name="height", data_type=int)
    width = workflow.add_input(name="width", data_type=int)
    num_inference_steps = workflow.add_input(name="num_inference_steps", data_type=int)
    guidance_scale = workflow.add_input(name="guidance_scale", data_type=float)

    latents = latents_generator(height=height, width=width, seed=seed)

    loop = dict(
        model=flux,
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=num_inference_steps,
        prompt_embeds=t5_flux(prompt=prompt),
        pooled_prompt_embeds=clip_flux(prompt=prompt),
        # Flux's distilled guidance embedding, not classifier-free guidance.
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )

    def predict(conditioning: DenoiseContext):
        """One model pass, under whichever text conditioning it is given."""
        return flux(
            latents=conditioning.latents,
            timestep=conditioning.timestep,
            prompt_embeds=conditioning.prompt_embeds,
            pooled_prompt_embeds=conditioning.pooled_prompt_embeds,
            guidance=conditioning.guidance,
            height=conditioning.height,
            width=conditioning.width,
        )

    def with_cfg():
        """Unconditional and conditional passes, combined by the scheduler."""
        negative_prompt_embeds = t5_flux(prompt=negative_prompt)
        negative_pooled_prompt_embeds = clip_flux(prompt=negative_prompt)

        def step(context: DenoiseContext):
            uncond = context.with_conditioning(
                negative_prompt_embeds, negative_pooled_prompt_embeds
            )
            return scheduler(
                latents=context.latents,
                timestep=context.timestep,
                noise_pred_uncond=predict(uncond),
                noise_pred_text=predict(context),
                guidance_scale=cfg_guidance_scale,
                mode="step_classifier_free_guidance",
            )

        return {"latents": denoise_loop(step_fn=step, **loop)}

    def without_cfg():
        """A single conditional pass."""

        def step(context: DenoiseContext):
            return scheduler(
                latents=context.latents,
                timestep=context.timestep,
                noise_pred=predict(context),
                mode="step",
            )

        return {"latents": denoise_loop(step_fn=step, **loop)}

    # Resolved when the request arrives. Comparing a request input builds the
    # predicate for you; writing `if cfg_guidance_scale > 1.0:` raises, because
    # both branches are traced now, before any request exists.
    denoised = cond(cfg_guidance_scale > 1.0, with_cfg, without_cfg)["latents"]

    output_img = vae(
        latents=denoised, mode="decode_latents", height=height, width=width
    )
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

    service_id = register_workflow(
        workflow=create_workflow(model_path=args.model_path),
        server_url=args.server_url,
    )
    print(f"Registered workflow with service ID: {service_id}")
