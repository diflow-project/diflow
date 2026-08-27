# python workflow_hub/run_flux2_klein_workflow.py --service-id flux2klein_txt2img_workflow
import argparse
import base64
import io
import time

from PIL import Image

from diflow.interface import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-id", default="flux2klein_txt2img_workflow")
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    started = time.time()
    response = run_inference(
        args.service_id,
        inputs={
            "prompt": "A cat holding a sign that says hello world",
            "num_inference_steps": 4,
            "seed": 0,
            "height": 1024,
            "width": 1024,
        },
        server_url=args.server_url,
    )
    if response["status"] != "success":
        raise RuntimeError(f"Inference failed: {response}")
    encoded = response["results"]["output_img"]
    encoded = encoded[0] if isinstance(encoded, list) else encoded
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.save("flux2_klein_output.png")
    print(
        f"Saved flux2_klein_output.png ({image.size}) "
        f"in {time.time() - started:.2f}s"
    )
