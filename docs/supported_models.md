# Supported Models and Workflows

DiFlow includes the following text-to-image workflows:

| Built-in name | Checkpoint | Local directory | Service ID | Client |
| --- | --- | --- | --- | --- |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | `FLUX.1-schnell` | `flux_schnell_txt2img_workflow` | `workflow_hub/run_flux_workflow.py` |
| `flux-dev` | `black-forest-labs/FLUX.1-dev` | `FLUX.1-dev` | `flux_txt2img_workflow` | `workflow_hub/run_flux_workflow.py` |
| `zimage` | `Tongyi-MAI/Z-Image` | `Z-Image` | `zimage_txt2img_workflow` | `workflow_hub/run_zimage_workflow.py` |
| `zimage-turbo` | `Tongyi-MAI/Z-Image-Turbo` | `Z-Image-Turbo` | `zimage_turbo_txt2img_workflow` | `workflow_hub/run_zimage_turbo_workflow.py` |
| `flux2-klein` | `black-forest-labs/FLUX.2-klein-9B` | `Flux.2-klein` | `flux2klein_txt2img_workflow` | `workflow_hub/run_flux2_klein_workflow.py` |

## Download checkpoints

Download every supported checkpoint into a common model root:

```bash
./scripts/download_models.sh /path/to/checkpoints
```

To download one checkpoint directly, use its repository ID from the table. For
example:

```bash
hf download Tongyi-MAI/Z-Image --local-dir /path/to/checkpoints/Z-Image
```

Set `DIFLOW_MODEL_ROOT` to let built-in workflows derive the local directory
shown in the table:

```bash
export DIFLOW_MODEL_ROOT=/path/to/checkpoints
diflow serve --workflow zimage
```

You can also pass the checkpoint explicitly:

```bash
diflow serve --workflow zimage --model-path /path/to/checkpoints/Z-Image
diflow serve --workflow zimage-turbo --model-path /path/to/checkpoints/Z-Image-Turbo
diflow serve --workflow flux2-klein --model-path /path/to/checkpoints/Flux.2-klein
```

## Workflow behavior

Z-Image uses positive-anchored classifier-free guidance. A positive
`cfg_guidance_scale` enables a negative-prompt encoder branch and separate
positive and negative transformer calls for every denoising step. This differs
from the reference pipeline's single batched CFG call, so compare performance
only with checkpoint-backed latency measurements.

Z-Image Turbo uses the bundled client's 9-step, seed-0, 1024x1024 defaults and
does not use classifier-free guidance. FLUX.2 Klein is step-wise distilled and
does not expose guidance or negative-prompt inputs. Both execute one transformer
call per denoising step.

The bundled clients save decoded responses under `imgs/`:

```bash
python workflow_hub/run_zimage_workflow.py
python workflow_hub/run_zimage_turbo_workflow.py
python workflow_hub/run_flux2_klein_workflow.py
```
