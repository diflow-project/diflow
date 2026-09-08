# python workflow_hub/run_zimage_turbo_workflow.py
import argparse
import base64
import io
import time
from pathlib import Path

from PIL import Image

from diflow.interface import run_inference

SERVICE_ID = "zimage_turbo_txt2img_workflow"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "imgs"
PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
    "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
    "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
    "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
    "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
    "blurred colorful distant lights."
)


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
            "zimage_turbo_image.png"
            if len(images) == 1
            else f"zimage_turbo_image_{idx}.png"
        )
        print(f"output_img_{idx}.shape: {img.size}")
        img.save(output_path)
        print(f"Saved {output_path}")


def build_inputs():
    return {
        "prompt": PROMPT,
        "num_inference_steps": 9,
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
