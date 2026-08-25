# python workflow_hub/run_workflow.py --service-id <service_id>
#
# Classifier-free guidance is a request parameter, not a separate service: pass
# cfg_guidance_scale > 1.0 to enable it, or 1.0 for a single model pass per step.
# It is required either way, since it is what selects the branch.
import argparse
import base64
import io
import time

import numpy as np
from diffusers.utils import load_image
from PIL import Image

from diflow.interface import run_inference

CONTROL_IMAGES = {
    "flux_txt2img_controlnet_canny_workflow": "imgs/flux_canny_image.png",
    "flux_txt2img_controlnet_depth_workflow": "imgs/flux_depth_image.png",
}


def decode_image(img_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_str)))


def save_images(results) -> None:
    images = results["output_img"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-id",
        type=str,
        required=True,
        choices=[
            "t5_workflow",
            "t5_func_workflow",
            # FLUX
            "flux_txt2img_workflow",
            *CONTROL_IMAGES,
        ],
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=7.0,
        help="1.0 runs one model pass per step; above that enables CFG",
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    service_id = args.service_id
    server_url = args.server_url

    start_time = time.time()

    if service_id == "t5_workflow" or service_id == "t5_func_workflow":
        response = run_inference(
            service_id,
            inputs={
                "text_prompt": "Anime style illustration of a girl wearing a suit.",
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            results = response["results"]
            data = np.array(results["text_embed"])
            print(f"text_embed.shape: {data.shape}")
            print(f"The first 5 elements of text_embed: {data[:5]}")

    elif service_id == "flux_txt2img_workflow":
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A cat holding a sign that says hello world",
                "negative_prompt": "ugly, blurry, low quality, deformed, text",
                "cfg_guidance_scale": args.cfg_guidance_scale,
                "num_inference_steps": 50,
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            save_images(response["results"])

    elif service_id in CONTROL_IMAGES:
        response = run_inference(
            service_id,
            inputs={
                "prompt": "A portrait of a lovely shorthair golden-shaded cat, sitting on a windowsill, capturing every fur detail.",
                "negative_prompt": "ugly, blurry, low quality, deformed, dark, text",
                "cfg_guidance_scale": args.cfg_guidance_scale,
                "control_image": encode_control_image(CONTROL_IMAGES[service_id]),
                "conditioning_scale": 0.7,  # originally named controlnet_conditioning_scale
                "guidance_scale": 3.5,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "seed": 0,
            },
            server_url=server_url,
        )
        if response["status"] == "success":
            save_images(response["results"])

    else:
        raise ValueError(f"Invalid service ID: {service_id}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
