# Runtime profiles

DiFlow automatically measures operator execution latency and activation memory
for each configured resolution and batch size. The coordinator uses the result
for dynamic scheduling, memory planning, and optional SLA admission control.

## What is measured

The profiler:

1. Expands the workflow with representative inputs.
2. Captures the real inputs of every operator.
3. Deduplicates equivalent operator/mode/shape combinations.
4. Measures warmup and repeated execution on the target GPU.
5. Records model loading latency separately from operator execution latency.

`profile_steps` controls how cheaply the denoising graph is expanded. It does
not claim that a complete workflow has that number of production denoising
steps; the profile records per-operator shape and batch latency.

## Partial profiles

Each operator profile point has either latency/memory data or an error. CUDA OOM
and best-effort profiling errors are marked unsupported without discarding
successful points.

The cache manifest has one of these states:

- `complete`: every emitted operator point succeeded.
- `partial`: at least one point failed but usable points remain.
- `invalid`: no usable operator points; the cache entry is not published.

When no SLA admission decision is active, an unprofiled request is allowed to
run. DiFlow logs the gap and continues without proactive memory reservation for
that profile point. When SLA admission is enabled and the request supplies SLA
fields, a missing profile is returned as an explicit profile-unavailable error.
Use a complete profile when predictable memory planning is required.

## Cache identity

The default cache is `~/.cache/diflow/benchmarks`. Entries are fingerprinted by:

- workflow source and factory arguments;
- benchmark inputs and sweep;
- DiFlow profiler/operator source;
- Python, Torch, Transformers, and DiFlow versions;
- GPU model, memory, UUID, and driver;
- strict or best-effort mode.

The manifest records creation time, status, software versions, GPU identity,
benchmark configuration, and an error summary.

## Workflow declaration

Custom workflows attach representative non-shape inputs:

```python
from diflow import BenchmarkSpec, Workflow


def create_workflow(model_path: str) -> Workflow:
    return Workflow(
        "my_workflow",
        benchmark=BenchmarkSpec(
            inputs={
                "prompt": "A product photo of a ceramic teapot",
                "seed": 0,
                "num_inference_steps": 2,
            },
            resolutions=((256, 256), (512, 512)),
            batch_sizes=(1, 2),
        ),
    )
```

The profiler injects `height`, `width`, and the profiling step count.

## Reusing a profile

The profile JSON is stored below the fingerprinted cache directory. Reuse it
without startup profiling:

```bash
diflow serve \
  --workflow flux-schnell \
  --model-path /path/to/FLUX.1-schnell \
  --no-auto-benchmark \
  --runtime-profile /path/to/profile.json
```

Only reuse a profile with the same workflow, model configuration, software
stack, and GPU class. The automatic cache performs these identity checks; a
manually supplied file is the operator's responsibility.
