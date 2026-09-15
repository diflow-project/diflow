# Agent Workflow API

DiFlow exposes a declarative REST API for agent harnesses. Agents can discover
the server's existing operators and model aliases, validate a workflow, register
it idempotently, and run it asynchronously. The API never accepts Python code,
shell commands, network callbacks, or filesystem model paths.

The existing `/api/workflow/...` endpoints remain available for current Python
clients. New integrations should use `/api/v1/...`.

## Discover capabilities

```bash
curl http://127.0.0.1:8000/api/v1/operators
curl http://127.0.0.1:8000/api/v1/models
curl http://127.0.0.1:8000/api/v1/workflow-templates
curl http://127.0.0.1:8000/api/v1/prompt-skills
```

Model aliases are configured by the server administrator. The workflow started
with `--workflow flux-schnell --model-path /path/to/FLUX.1-schnell` is
automatically published as `model_ref: "flux-schnell"`. Additional aliases can
be supplied with repeatable `--api-model-ref NAME=PATH` options. Only `NAME` is
returned by the API.

## Register a built-in workflow

This form is useful for production workflows with request-dependent denoising
loops. It reuses DiFlow's reviewed control-flow implementation while allowing an
agent to configure inputs and deterministic prompt preprocessing.

```json
{
  "api_version": "diflow/v1",
  "name": "portrait-flux",
  "template": {
    "id": "flux-schnell",
    "model_ref": "flux-schnell"
  },
  "inputs": {
    "negative_prompt": {
      "type": "string",
      "required": false,
      "default": ""
    }
  },
  "preprocessors": [
    {
      "input": "prompt",
      "skill_id": "portrait-composition",
      "version": "1.0.0",
      "parameters": {
        "pose": "walking naturally",
        "framing": "full-body composition"
      }
    }
  ]
}
```

Save the document as `workflow.json`, then validate and register it:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/workflows/validate \
  -H 'content-type: application/json' --data-binary @workflow.json

curl -X POST http://127.0.0.1:8000/api/v1/workflows \
  -H 'content-type: application/json' --data-binary @workflow.json
```

Registration is content-addressed. Sending the same canonical spec again returns
the same `workflow_id` and `created: false`.

## Compose an operator DAG

Agents may connect operators returned by the catalog. Every edge references a
workflow input or a named operator output. The server validates operator IDs,
execution modes, ports, types, cycles, model aliases, and runtime-profile
coverage before registration.

```json
{
  "api_version": "diflow/v1",
  "name": "guidance-value",
  "inputs": {
    "guidance_scale": {"type": "number"}
  },
  "nodes": [
    {
      "id": "guidance",
      "operator": "GuidanceTensor",
      "inputs": {
        "guidance_scale": {"input": "guidance_scale"}
      }
    }
  ],
  "outputs": {
    "guidance": {"node": "guidance", "output": "guidance_tensor"}
  }
}
```

V1 intentionally does not accept arbitrary region expressions. Use a built-in
template for workflows that need loops or branches; this keeps the registration
surface data-only and prevents remote code execution.

## Run and poll

```bash
curl -X POST \
  http://127.0.0.1:8000/api/v1/workflows/WORKFLOW_ID/runs \
  -H 'content-type: application/json' \
  -d '{"inputs":{"prompt":"a traveler","height":512,"width":512,"seed":0,"num_inference_steps":4,"guidance_scale":0.0,"cfg_guidance_scale":1.0}}'

curl http://127.0.0.1:8000/api/v1/runs/RUN_ID
curl -X DELETE http://127.0.0.1:8000/api/v1/runs/RUN_ID
```

Run states are `queued`, `running`, `cancelling`, `succeeded`, `failed`,
`rejected`, or `cancelled`. SLA admission rejection is represented by
`SLO_REJECTED`, separate from an execution failure. Queued runs are cancelled
immediately. GPU kernels that have already been dispatched are not preempted;
their run stays `cancelling` until coordinator cleanup finishes, and the result
is then discarded.

## Define a prompt skill

A user prompt skill is a deterministic format template with declared scalar
parameters. Templates must preserve `{prompt}` and cannot access attributes or
items. They do not execute code, make network calls, or load models.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/prompt-skills \
  -H 'content-type: application/json' \
  -d '{
    "id":"camera-style",
    "version":"1",
    "description":"Adds an approved camera style",
    "template":"{prompt}, photographed with {camera}",
    "parameters":{
      "camera":{"required":true,"choices":["35mm","85mm"]}
    }
  }'
```

The version and digest are pinned when a workflow is registered. Each run keeps
the original prompt, transformed prompt, resolved parameters, skill version, and
skill ID in `prompt_trace`.

## Errors

V1 errors are stable JSON objects suitable for an agent repair loop:

```json
{
  "error": {
    "code": "TYPE_MISMATCH",
    "message": "Input 'guidance_scale' expects float, got str",
    "path": "nodes.guidance.inputs.guidance_scale",
    "retryable": false
  }
}
```
