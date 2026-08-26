# Quick Start

## Serve FLUX.1-schnell

Start one local worker, the API server, and the built-in workflow:

```bash
diflow serve \
  --workflow flux-schnell \
  --model-path /path/to/FLUX.1-schnell
```

The equivalent module entrypoint is:

```bash
python -m diflow.launch_server \
  --workflow flux-schnell \
  --model-path /path/to/FLUX.1-schnell
```

The server listens on `http://127.0.0.1:8000` and registers the service ID
`flux_schnell_txt2img_workflow`.

## Send an inference request

The included client builds the complete request and decodes the returned image:

```bash
python workflow_hub/run_flux_workflow.py \
  --service-id flux_schnell_txt2img_workflow \
  --cfg-guidance-scale 1.0 \
  --server-url http://127.0.0.1:8000
```

The output is written to `output_img_0.png`. The corresponding HTTP body is:

```json
{
  "inputs": {
    "prompt": "A cat holding a sign that says hello world",
    "negative_prompt": "ugly, blurry, low quality, deformed, text",
    "cfg_guidance_scale": 1.0,
    "guidance_scale": 0.0,
    "num_inference_steps": 4,
    "seed": 0,
    "height": 1024,
    "width": 1024
  }
}
```

Send it to:

```text
POST /api/workflow/flux_schnell_txt2img_workflow/inference
```

A successful response has this shape:

```json
{
  "status": "success",
  "results": {
    "output_img": "<base64-encoded PNG>"
  }
}
```

## Compose and register a workflow

A Python workflow file exports a type-annotated `create_workflow(...)`
function. The following example composes a compact FLUX.1-schnell text-to-image
workflow; save it as `my_flux_workflow.py`:

```python
from diflow.interface import BenchmarkSpec, Workflow, denoise_loop
from diflow.operators import (
    CLIP_Flux,
    Config,
    Flux1Schnell,
    Flux1VAE,
    FluxLatentsGenerator,
    FluxSchnellFlowMatchEulerDiscreteScheduler,
    T5_Flux,
)


def create_workflow(model_path: str) -> Workflow:
    workflow = Workflow(
        name="my_flux_workflow",
        benchmark=BenchmarkSpec(
            inputs={
                "prompt": "A product photo of a ceramic teapot",
                "seed": 0,
                "num_inference_steps": 2,
                "guidance_scale": 0.0,
            }
        ),
    )

    seed = workflow.add_input("seed", int)
    prompt = workflow.add_input("prompt", str)
    height = workflow.add_input("height", int)
    width = workflow.add_input("width", int)
    num_inference_steps = workflow.add_input("num_inference_steps", int)
    guidance_scale = workflow.add_input("guidance_scale", float)

    config = Config(model_path=model_path)
    latents = FluxLatentsGenerator()(height=height, width=width, seed=seed)
    prompt_embeds = T5_Flux(config)(prompt=prompt)
    pooled_prompt_embeds = CLIP_Flux(config)(prompt=prompt)
    denoised_latents = denoise_loop(
        model=Flux1Schnell(config),
        scheduler=FluxSchnellFlowMatchEulerDiscreteScheduler(config),
        latents=latents,
        num_inference_steps=num_inference_steps,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    output_img = Flux1VAE(config)(
        latents=denoised_latents,
        mode="decode_latents",
        height=height,
        width=width,
    )
    workflow.add_output(output_img, name="output_img")
    return workflow
```

Pass the file to `diflow serve`. DiFlow derives `--model-path` from the factory
signature, builds the workflow, and registers it during startup:

```bash
diflow serve \
  --workflow ./my_flux_workflow.py \
  --model-path /path/to/FLUX.1-schnell
```

The service ID is the workflow name, `my_flux_workflow`. Serving uses one local
worker by default. Use `--num-workers N` for multiple local workers. If the
NVSHMEM extension is unavailable, DiFlow automatically transfers intermediate
tensors through shared host memory. The choice can be made explicit:

```bash
diflow serve \
  --workflow ./my_flux_workflow.py \
  --model-path /path/to/FLUX.1-schnell \
  --num-workers 2 \
  --transfer-backend host
```

The host backend is single-node only and uses `/dev/shm/diflow`. Multi-node
workers configured with `--hostfile PATH` require `--transfer-backend nvshmem`.

To add the workflow to an already running DiFlow server, register it directly:

```python
from diflow import register_workflow
from my_flux_workflow import create_workflow

service_id = register_workflow(
    workflow=create_workflow("/path/to/FLUX.1-schnell"),
    server_url="http://127.0.0.1:8000",
)
print(service_id)
```

See the
[complete FLUX.1-schnell workflow](../workflow_hub/flux_schnell/register_txt2img_workflow.py)
for classifier-free guidance and request-dependent control flow.
