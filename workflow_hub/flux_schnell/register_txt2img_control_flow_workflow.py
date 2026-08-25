# python workflow_hub/flux_schnell/register_txt2img_control_flow_workflow.py
#
# The same graph as register_txt2img_workflow.py, but with the denoising loop
# written out with for_range instead of denoise_loop().
#
# denoise_loop is a convenience wrapper over exactly this, so the two produce
# isomorphic graphs -- asserted in
# tests/interface/test_denoise_isomorphism.py::TestHandWrittenExampleMatchesHelper.
# Start here when you need a loop the wrapper does not cover: a different step
# count per phase, an extra conditional, a second nested loop. If only the
# per-step computation differs, register_txt2img_helper_workflow.py is the cheaper
# way in.
#
# Everything denoise_loop would have done for you is in `denoise` below: hoisting
# the timestep schedule and the guidance embedding out of the loop, indexing the
# schedule per iteration, and threading the latents through the carry.
#
# Two rules to keep in mind:
#
#   * The body is traced ONCE, at registration. The loop index `i` is a handle,
#     not a number -- pass it to operators, or subscript a tensor with it
#     (`timesteps[i]`), but don't do Python arithmetic on it.
#   * A Python `if` in the body is evaluated at registration, so it can only test
#     things known then (a model id, whether a list is empty). Anything that
#     depends on the request -- like whether guidance is enabled -- has to go
#     through cond(), whose predicate is evaluated when the request arrives.
#     Comparing a request input builds that predicate for you, so
#     `cfg_guidance_scale > 1.0` is an expression rather than a bool; writing
#     `if cfg_guidance_scale > 1.0:` raises and tells you to use cond().
import argparse

from diflow.interface import Workflow, register_workflow
from diflow.interface.control_flow import cond, for_range
from diflow.operators import (
    CLIP_Flux,
    Config,
    Flux1Schnell,
    Flux1VAE,
    FluxLatentsGenerator,
    FluxSchnellFlowMatchEulerDiscreteScheduler,
    T5_Flux,
)

# Not re-exported from diflow.operators.
from diflow.operators.custom.guidance_tensor import GuidanceTensor
from diflow.operators.utils import default_model_path


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(name="flux_schnell_txt2img_control_flow_workflow")

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

    clip_prompt_embeds = clip_flux(prompt=prompt)
    t5_prompt_embeds = t5_flux(prompt=prompt)

    def denoise(step_fn):
        """The scaffolding: schedule, guidance embedding, loop, carry.

        The timestep schedule is computed once and every iteration indexes into
        it, and Flux's distilled guidance embedding is loop-invariant, so both are
        hoisted out. Putting them in the body would emit a redundant node per
        iteration.
        """
        timesteps = scheduler(
            num_inference_steps=num_inference_steps, latents=latents, mode="init"
        )
        guidance = GuidanceTensor()(guidance_scale=guidance_scale)

        def body(i, carry):
            # `timesteps[i]` emits the indexing node for you.
            return {"latents": step_fn(carry["latents"], timesteps[i], guidance)}

        # The carry is what threads state between iterations. It is also what makes
        # them run in order: the executor schedules on data dependencies alone, and
        # the scheduler operator is stateful, so the only thing stopping it from
        # running all the steps at once is that each consumes the previous output.
        return for_range(num_inference_steps, body, carry={"latents": latents})[
            "latents"
        ]

    def predict(
        current_latents, timestep, guidance, prompt_embeds, pooled_prompt_embeds
    ):
        """One model pass."""
        return flux(
            latents=current_latents,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance=guidance,
            height=height,
            width=width,
        )

    def with_cfg():
        """Unconditional and conditional passes, combined by the scheduler."""
        clip_negative_prompt_embeds = clip_flux(prompt=negative_prompt)
        t5_negative_prompt_embeds = t5_flux(prompt=negative_prompt)

        def step(current_latents, timestep, guidance):
            return scheduler(
                latents=current_latents,
                timestep=timestep,
                noise_pred_uncond=predict(
                    current_latents,
                    timestep,
                    guidance,
                    t5_negative_prompt_embeds,
                    clip_negative_prompt_embeds,
                ),
                noise_pred_text=predict(
                    current_latents,
                    timestep,
                    guidance,
                    t5_prompt_embeds,
                    clip_prompt_embeds,
                ),
                guidance_scale=cfg_guidance_scale,
                mode="step_classifier_free_guidance",
            )

        return {"latents": denoise(step)}

    def without_cfg():
        """A single conditional pass, and no negative prompt encoded at all."""

        def step(current_latents, timestep, guidance):
            return scheduler(
                latents=current_latents,
                timestep=timestep,
                noise_pred=predict(
                    current_latents,
                    timestep,
                    guidance,
                    t5_prompt_embeds,
                    clip_prompt_embeds,
                ),
                mode="step",
            )

        return {"latents": denoise(step)}

    # Resolved per request, when cfg_guidance_scale has a value. Both branches are
    # traced now, so both must be buildable even though only one is served.
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
