# Workflow Hub

`workflow_hub` is DiFlow's collection of ready-to-use workflows. It provides a
shared home for workflows from different model families and use cases, so they
can be discovered, registered, served, and exercised through consistent
repository entrypoints.

Each model-family directory, such as `flux_dev/`, `zimage/`, or
`flux2_klein/`, contains the workflow definition and registration entrypoint.
The top-level `run_*_workflow.py` files are runnable clients that construct
requests, call a registered service, and process its responses. Workflows that
are exposed through the built-in loader can also be selected with
`diflow serve --workflow`.

The hub is intended to contain many workflows while keeping each workflow's
model-specific graph separate and its surrounding integration consistent with
the rest of the repository. The following guide describes how to add or update
a workflow in this collection.

# Workflow Development Guide

Adding a workflow means reproducing a reference pipeline inside DiFlow's
operator and graph abstractions. A strong workflow contribution combines a
runnable implementation with repository-consistent code, reference-aligned
numerical behavior, and clear validation evidence. This guide outlines that
process.

## 1. Choose One Reference

Define the single source of truth before writing code. Record:

- the reference repository and exact commit;
- the model repository and checkpoint revision or snapshot;
- Python, PyTorch, Transformers, and Diffusers versions;
- device type and tensor dtypes; and
- every generation input, including prompts, seed, resolution, step count,
  guidance settings, scheduler options, and batch size.

Avoid combining behavior from an older DiFlow or ServerlessT2I implementation,
a newer upstream release, and the pinned reference without explanation. If they
differ, follow the selected reference and document the incompatibility.

## 2. Follow Repository Conventions

Repository consistency applies to every new or modified file, not only to
client scripts. Before implementing a component, find the closest existing
example and use it as the structural template.

Match existing conventions for:

- file placement and module boundaries;
- class, function, variable, execution-mode, and operator-ID names;
- imports, type annotations, comments, logging, and error handling;
- `setup_io()`, `initialize()`, and `execute()` contracts;
- model loading, device placement, and dtype conversion;
- workflow composition, loops, conditions, CFG, and adapters;
- operator registration and `to_dict()`/`from_dict()` symmetry;
- command-line interfaces and response processing;
- benchmark declarations;
- unit, graph, client, and end-to-end tests; and
- documentation and examples.

Use this sequence for every new component:

1. Search for equivalent behavior in the repository.
2. Identify the closest reference file.
3. Reuse its public abstractions and structure.
4. Add only behavior that is specific to the new model family.
5. Explain and test every necessary deviation from existing conventions.

For example, a new text-to-image family should map its components before
implementation:

| New component | Repository precedent |
| --- | --- |
| Request client | Existing `workflow_hub/run_*_workflow.py` client |
| Text-to-image graph | Existing txt2img workflow with the same CFG behavior |
| Latent generator | Existing generator with the same packed/unpacked layout |
| Text encoder | Existing encoder with the same output and mask contract |
| Transformer | An existing `BaseDiffusionModel` implementation |
| Scheduler | Existing scheduler from the same scheduler family |
| VAE | Existing VAE operator with the same latent layout |
| Model download | Existing entries in `scripts/download_models.sh` |
| Built-in loading | Existing entries in `diflow/cli/workflow_loader.py` |
| Tests | Corresponding operator, workflow, client, and golden-graph tests |

Prefer shared abstractions when they already provide the needed capability. If
reuse is not practical, explain why the existing implementation is unsuitable,
which reference behavior motivates the new code, and which regression test
covers that distinction.

### Review Repository Integration Surfaces

A workflow often needs a few repository-level updates in addition to its
operators so that users can download, select, run, profile, and test it through
the usual entrypoints. Review the integration surfaces below and update the ones
that apply:

- implement operators under `diflow/operators/`;
- add operator IDs and public exports following the existing registration
  pattern;
- add the workflow package and registration file under `workflow_hub/`;
- add the built-in workflow entry to `diflow/cli/workflow_loader.py` when the
  workflow should be selectable by `diflow serve --workflow`;
- add the official model repository ID and compatible local directory to
  `scripts/download_models.sh`;
- ensure the download directory matches the default path derived by the
  workflow loader and registration script;
- add or update the matching `workflow_hub/run_*_workflow.py` client;
- declare representative automatic benchmark inputs, resolutions, and batch
  sizes;
- document download and serving commands in `docs/quickstart.md` or the
  relevant existing guide; and
- add operator, serialization, loader, graph, client, benchmark, and model-path
  validation tests following existing test organization.

When a workflow needs a new checkpoint, prefer documenting model acquisition and
adding it to the repository's download script. For example, Z-Image can use an
entry such as `Tongyi-MAI/Z-Image` in `scripts/download_models.sh`, stored under
the same directory name used by `default_model_path(...)`. Integration surfaces
that do not apply can be noted briefly in the pull request when useful.

## 3. Map the Reference Pipeline to Operators

Create an explicit mapping from each reference stage to a DiFlow operator and
workflow node. At minimum, cover:

1. request validation and preprocessing;
2. tokenization and text encoding;
3. initial latent generation;
4. scheduler initialization;
5. each transformer invocation;
6. classifier-free guidance or distilled guidance;
7. each scheduler update;
8. VAE decoding and image postprocessing; and
9. response serialization and client-side image saving.

The mapping should preserve input names, shapes, dtypes, devices, layouts,
ordering, and state transitions. Keep model-specific knowledge in the relevant
operator when possible rather than adding model-ID branches to generic workflow
builders.

## 4. Compare Stages Apple to Apple

Use the same checkpoint, input, device, dtype, and software versions for the
reference and DiFlow runs. Compare stages in order and stop at the first
divergence.

### Request inputs

Compare the positive and negative prompts, seed, height, width, inference step
count, guidance values, scheduler options, image count, and batch size.

### Text encoding

Compare the tokenizer and chat template, padding, truncation, maximum sequence
length, selected hidden-state layer, attention mask, embedding shape, dtype,
and values.

### Initial latents

Compare the generator device, seed, shape, dtype, packed or unpacked layout,
and tensor values. A matching seed is insufficient when generator devices or
sampling dtypes differ.

### Scheduler initialization

Compare the sigma construction, timestep values, dynamic shift or `mu`,
terminal sigma, begin index, and the effective delta for every denoising step.

### Transformer calls

Compare latent and conditioning layouts, timestep conversion, dtype casts,
batch ordering, attention masks, output signs, and output tensors. The number
of transformer calls per step should match the reference. If CFG is batched in
the reference, preserve its positive/negative ordering in the batched call.

### Guidance and scheduler updates

Compare the CFG activation threshold, formula, normalization, truncation,
per-step scale, prediction dtype, scheduler input, and updated latents after
every step.

### VAE and image postprocessing

Compare latent layout and dtype, scaling and shift factors, decode inputs,
image-processor settings, color mode, dimensions, and output encoding.

Fix the first numerical divergence before investigating later stages. A final
image mismatch cannot be diagnosed reliably by changing several stages at
once.

## 5. Compose the Workflow Faithfully

- The graph should express the reference computation, including the same branch
  conditions and transformer invocation count.
- Avoid negative-prompt encoding when the selected request path does not use
  CFG.
- Keep distilled, Turbo, CFG, ControlNet, and adapter paths explicit when their
  contracts differ.
- Keep `NodeIO` names, source types, execution modes, and required outputs
  consistent through serialization and reconstruction.
- Use existing loop, condition, adapter, and denoising helpers when their
  semantics match the reference.
- Treat a change from a batched reference call to separate calls, or the reverse,
  as an explicit design decision rather than a convenience refactor. Include
  numerical evidence and performance results when making this change.

## 6. Follow Existing Client Patterns

Use the closest existing `workflow_hub/run_*_workflow.py` file as the template.
A client should normally:

- expose `--service-id` and `--server-url`;
- parse arguments only under `if __name__ == "__main__"`;
- construct request data in a testable `build_inputs()` function;
- validate and process results in a testable `process_response()` function;
- fail immediately on a non-success response;
- support both a single image and a list of images;
- print output dimensions and elapsed time;
- use stable, predictable output names; and
- avoid personal checkpoint paths or machine-specific defaults.

Match the established client style for imports, constants, argument names,
control-image encoding, response decoding, and output placement. Test
`build_inputs()` and `process_response()` directly.

## 7. Profile the Supported Serving Surface

Automatic benchmarks should represent the shapes the service accepts.

- Include representative supported resolutions, not only one default shape.
- Include useful batch sizes greater than one when dynamic batching is
  supported.
- Reject unsupported resolutions or batch sizes at the API or operator boundary
  instead of silently using an unrelated nearest profile.
- Ensure benchmark inputs contain everything required to expand and execute the
  workflow.
- Record partial or failed profiles separately from successful measurements.

Consider a single-shape, batch-one benchmark when the serving API is explicitly
restricted to that shape and batch size.

## 8. Validate in Layers

Run the cheapest checks first, then increase realism.

### Static checks

```bash
python -m py_compile <changed-python-files>
./scripts/format.sh
git diff --check
```

### Operator tests

Test shapes, dtypes, layouts, devices, serialization round trips, scheduler
values, guidance formulas, conditioning order, and VAE preprocessing and
postprocessing.

### Workflow graph tests

Test node types and counts, branch selection, transformer calls per denoising
step, required inputs, benchmark specifications, and workflow JSON round trips.
Update golden graphs only after reviewing the semantic graph diff.

### Client tests

Test exact request dictionaries, error handling, single- and multi-image
responses, output paths, and image decoding.

### GPU reference comparison

Run the reference pipeline and DiFlow with the same checkpoint and inputs.
Capture and compare intermediate tensors at each mapped stage, then compare the
final image. Use exact equality where deterministic execution permits it;
otherwise report tolerances, maximum and mean errors, the first divergent
stage, and the reason exact equality is not expected.

## 9. State the Evidence Boundary

Report validation gates independently:

- source audit completed;
- static checks passed;
- CPU/operator tests passed;
- workflow graph tests passed;
- GPU job submitted;
- GPU job completed;
- intermediate tensors compared; and
- final image compared.

Keep these evidence levels distinct: a submitted job is not yet a completed run,
graph tests do not establish image correctness, and visual similarity is not
numerical equivalence. When a dependency, checkpoint, GPU, or external service
is unavailable, mark the affected check as not run rather than passed.

## 10. Make the Workflow Reproducible

- Document official model repository IDs and required revisions.
- When applicable, add new checkpoints to `scripts/download_models.sh`, using
  the official repository ID and the local directory expected by the built-in
  workflow.
- Use real workflow names and expected directory layouts in the quickstart.
- Record the pinned reference commit.
- Avoid relying on preinstalled private checkpoints or undocumented local
  paths.
- Keep generated images, logs, checkpoints, and benchmark artifacts out of
  commits unless they are intentional, reviewed fixtures or reports.

## 11. Keep Commits and Pull Requests Accurate

Prefer focused commits that can be reviewed and validated independently, such
as:

1. operators and reference-level tests;
2. workflow composition and graph tests; and
3. clients, benchmarks, and documentation.

Keep the pull-request description synchronized with the final implementation.
It is helpful to state:

- the pinned reference and checkpoint;
- CFG batching and transformer calls per step;
- supported and profiled shapes;
- tests that actually ran;
- GPU and final-image validation status; and
- known behavioral or performance differences.

Review claims from earlier commits when the implementation changes.

## Readiness Checklist

A workflow is generally ready for review when the applicable items below are
covered:

1. reference stages map to reviewed DiFlow code;
2. new files follow the closest repository precedent;
3. model-specific deviations are minimal, explicit, and tested;
4. operator, graph, client, and benchmark contracts are validated;
5. a matched GPU run compares intermediate values and the final image; and
6. documentation and the pull-request description accurately state the
   evidence and remaining limitations.
