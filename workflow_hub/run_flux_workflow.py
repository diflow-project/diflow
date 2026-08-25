# python workflow_hub/run_flux_workflow.py --service-id <service_id>
#
# Classifier-free guidance is a request parameter, not a separate service: pass
# --cfg-guidance-scale above 1.0 to enable it, or 1.0 for a single model pass per
# step. It is required either way, since it is what selects the branch.
import argparse
import base64
import io
import time

from diffusers.utils import load_image
from PIL import Image

from diflow.interface import run_inference

TXT2IMG_PROMPT = "A cat holding a sign that says hello world"
CONTROLNET_PROMPT = (
    "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, "
    "capturing every fur detail."
)

# schnell is distilled: four steps, and no distilled guidance embedding.
DEV = {"num_inference_steps": 50, "guidance_scale": 3.5}
SCHNELL = {"num_inference_steps": 4, "guidance_scale": 0.0}

SERVICES = {
    "flux_txt2img_workflow": DEV,
    "flux_txt2img_controlnet_canny_workflow": {
        **DEV,
        "control_image_path": "imgs/flux_canny_image.png",
    },
    "flux_txt2img_controlnet_depth_workflow": {
        **DEV,
        "control_image_path": "imgs/flux_depth_image.png",
    },
    "flux_schnell_txt2img_workflow": SCHNELL,
    "flux_schnell_txt2img_controlnet_canny_workflow": {
        **SCHNELL,
        "control_image_path": "imgs/flux_canny_image.png",
    },
    "flux_schnell_txt2img_controlnet_depth_workflow": {
        **SCHNELL,
        "control_image_path": "imgs/flux_depth_image.png",
    },
}


def decode_image(img_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_str)))


def process_response(response):
    if response["status"] != "success":
        raise ValueError(f"Invalid response: {response}")
    images = response["results"]["output_img"]
    if not isinstance(images, list):
        images = [images]
    for idx, img_str in enumerate(images):
        img = decode_image(img_str)
        print(f"output_img_{idx}.shape: {img.size}")
        img.save(f"output_img_{idx}.png")


def encode_control_image(path: str) -> str:
    control_image = load_image(path).convert("RGB")
    buffered = io.BytesIO()
    control_image.save(buffered, format="PNG")
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_inputs(service_id: str, cfg_guidance_scale: float):
    spec = dict(SERVICES[service_id])
    control_image_path = spec.pop("control_image_path", None)

    inputs = {
        "prompt": TXT2IMG_PROMPT if control_image_path is None else CONTROLNET_PROMPT,
        "negative_prompt": "ugly, blurry, low quality, deformed, text",
        "cfg_guidance_scale": cfg_guidance_scale,
        "seed": 0,
        "height": 1024,
        "width": 1024,
        **spec,
    }
    if control_image_path is not None:
        inputs["control_image"] = encode_control_image(control_image_path)
        # originally named controlnet_conditioning_scale
        inputs["conditioning_scale"] = 0.7
    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-id", type=str, required=True, choices=list(SERVICES))
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=7.0,
        help="1.0 runs one model pass per step; above that enables CFG",
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    start_time = time.time()
    process_response(
        run_inference(
            args.service_id,
            inputs=build_inputs(args.service_id, args.cfg_guidance_scale),
            server_url=args.server_url,
        )
    )
    print(f"Time taken: {time.time() - start_time} seconds")
