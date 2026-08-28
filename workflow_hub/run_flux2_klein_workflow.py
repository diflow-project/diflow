# python workflow_hub/run_flux2_klein_workflow.py
import argparse
import base64
import io
import time
from pathlib import Path

from PIL import Image

from diflow.interface import run_inference

SERVICE_ID = "flux2klein_txt2img_workflow"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "imgs"
PROMPT = "A cat holding a sign that says hello world"


def decode_image(img_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_str)))


def process_response(response):
    if response["status"] != "success":
        raise ValueError(f"Invalid response: {response}")
    images = response["results"]["output_img"]
    if not isinstance(images, list):
        images = [images]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, img_str in enumerate(images):
        img = decode_image(img_str)
        output_path = OUTPUT_DIR / (
            "flux2_klein_image.png"
            if len(images) == 1
            else f"flux2_klein_image_{idx}.png"
        )
        print(f"output_img_{idx}.shape: {img.size}")
        img.save(output_path)
        print(f"Saved {output_path}")


def build_inputs():
    return {
        "prompt": PROMPT,
        "num_inference_steps": 4,
        "seed": 0,
        "height": 1024,
        "width": 1024,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-id", type=str, default=SERVICE_ID, choices=[SERVICE_ID]
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    start_time = time.time()
    process_response(
        run_inference(
            args.service_id,
            inputs=build_inputs(),
            server_url=args.server_url,
        )
    )
    print(f"Time taken: {time.time() - start_time} seconds")
