# python workflow_hub/run_zimage_workflow.py --service-id zimage_txt2img_workflow
import argparse
import base64
import io
import time
from pathlib import Path

from PIL import Image

from diflow.interface import run_inference

SERVICE_ID = "zimage_txt2img_workflow"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "imgs"
PROMPT = (
    "两名年轻亚裔女性紧密站在一起，背景为朴素的灰色纹理墙面，"
    "可能是室内地毯地面。左侧女性留着长卷发，身穿藏青色毛衣，"
    "左袖有奶油色褶皱装饰，内搭白色立领衬衫，下身白色裤子；"
    "佩戴小巧金色耳钉，双臂交叉于背后。右侧女性留直肩长发，"
    "身穿奶油色卫衣，胸前印有“Tun the tables”字样，下方为“New ideas”，"
    "搭配白色裤子；佩戴银色小环耳环，双臂交叉于胸前。"
    "两人均面带微笑直视镜头。照片，自然光照明，柔和阴影，"
    "以藏青、奶油白为主的中性色调，休闲时尚摄影，中等景深，"
    "面部和上半身对焦清晰，姿态放松，表情友好，室内环境，"
    "地毯地面，纯色背景。"
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
            "zimage_image.png" if len(images) == 1 else f"zimage_image_{idx}.png"
        )
        print(f"output_img_{idx}.shape: {img.size}")
        img.save(output_path)
        print(f"Saved {output_path}")


def build_inputs(cfg_guidance_scale: float):
    return {
        "prompt": PROMPT,
        "negative_prompt": "",
        "cfg_guidance_scale": cfg_guidance_scale,
        "num_inference_steps": 50,
        "seed": 0,
        "height": 1024,
        "width": 1024,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-id", type=str, default=SERVICE_ID, choices=[SERVICE_ID]
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=4.0,
        help="0.0 runs one model pass per step; above that enables Z-Image CFG",
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    start_time = time.time()
    process_response(
        run_inference(
            args.service_id,
            inputs=build_inputs(args.cfg_guidance_scale),
            server_url=args.server_url,
        )
    )
    print(f"Time taken: {time.time() - start_time} seconds")
