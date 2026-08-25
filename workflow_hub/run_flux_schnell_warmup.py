import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

CMD_FLUX_SCHNELL_TXT2IMG_CFG = [
    "python",
    "workflow_hub/run_flux_workflow.py",
    "--service-id",
    "flux_schnell_txt2img_workflow",
    "--server-url",
    "http://localhost:7777",
]

CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_Canny_CFG = [
    "python",
    "workflow_hub/run_flux_workflow.py",
    "--service-id",
    "flux_schnell_txt2img_controlnet_canny_workflow",
    "--server-url",
    "http://localhost:7777",
]

CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_DEPTH_CFG = [
    "python",
    "workflow_hub/run_flux_workflow.py",
    "--service-id",
    "flux_schnell_txt2img_controlnet_depth_workflow",
    "--server-url",
    "http://localhost:7777",
]


def run_one(CMD, idx: int) -> int:
    print(f"Starting request {idx}, time: {datetime.now()}")
    result = subprocess.run(CMD, text=True)
    print(f"Request {idx} finished with code {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    NUM_REQUESTS_SERIAL = 2
    NUM_REQUESTS_CONCURRENT = 4

    print(f"Serial {NUM_REQUESTS_SERIAL} requests", CMD_FLUX_SCHNELL_TXT2IMG_CFG)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(run_one, CMD_FLUX_SCHNELL_TXT2IMG_CFG, i)
            for i in range(1, NUM_REQUESTS_SERIAL + 1)
        ]
        for f in futures:
            f.result()

    print(
        f"Serial {NUM_REQUESTS_SERIAL} requests",
        CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_Canny_CFG,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(run_one, CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_Canny_CFG, i)
            for i in range(1, NUM_REQUESTS_SERIAL + 1)
        ]
        for f in futures:
            f.result()

    print(
        f"Serial {NUM_REQUESTS_SERIAL} requests",
        CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_DEPTH_CFG,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(run_one, CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_DEPTH_CFG, i)
            for i in range(1, NUM_REQUESTS_SERIAL + 1)
        ]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [
            executor.submit(run_one, CMD_FLUX_SCHNELL_TXT2IMG_CFG, i)
            for i in range(1, NUM_REQUESTS_CONCURRENT + 1)
        ]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [
            executor.submit(run_one, CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_Canny_CFG, i)
            for i in range(1, NUM_REQUESTS_CONCURRENT + 1)
        ]
        for f in futures:
            f.result()

    print(f"Concurrent {NUM_REQUESTS_CONCURRENT} requests")
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS_CONCURRENT) as executor:
        futures = [
            executor.submit(run_one, CMD_FLUX_SCHNELL_TXT2IMG_CONTROLNET_DEPTH_CFG, i)
            for i in range(1, NUM_REQUESTS_CONCURRENT + 1)
        ]
        for f in futures:
            f.result()
