from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
Scalar = str | int | float | bool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkflowInputSpec(StrictModel):
    type: Literal["string", "integer", "number", "boolean", "image"]
    description: Optional[str] = Field(default=None, max_length=500)
    required: bool = True
    default: Any = None

    @model_validator(mode="after")
    def validate_default(self) -> "WorkflowInputSpec":
        if self.required and self.default is not None:
            raise ValueError("set required=false when declaring a default")
        if not self.required and self.default is None:
            raise ValueError("required=false inputs must declare a non-null default")
        return self


class ValueRef(StrictModel):
    input: Optional[Identifier] = None
    node: Optional[Identifier] = None
    output: Optional[Identifier] = None

    @model_validator(mode="after")
    def validate_reference(self) -> "ValueRef":
        is_input = self.input is not None
        is_node_output = self.node is not None or self.output is not None
        if is_input == is_node_output:
            raise ValueError("set either input or node/output")
        if is_node_output and (self.node is None or self.output is None):
            raise ValueError("node and output must be set together")
        return self


class WorkflowNodeSpec(StrictModel):
    id: Identifier = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    operator: Identifier = Field(min_length=1, max_length=128)
    model_ref: Optional[Identifier] = Field(default=None, max_length=128)
    mode: str = Field(default="default", min_length=1, max_length=64)
    inputs: Dict[str, ValueRef] = Field(default_factory=dict)


class WorkflowTemplateSpec(StrictModel):
    id: Identifier = Field(min_length=1, max_length=128)
    model_ref: Optional[Identifier] = Field(default=None, max_length=128)
    parameters: Dict[str, Scalar] = Field(default_factory=dict)


class PromptSkillUse(StrictModel):
    input: Identifier = Field(min_length=1, max_length=128)
    skill_id: Identifier = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    parameters: Dict[str, Scalar] = Field(default_factory=dict)


class WorkflowSpec(StrictModel):
    api_version: Literal["diflow/v1"] = "diflow/v1"
    name: Identifier = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    description: Optional[str] = Field(default=None, max_length=1000)
    inputs: Dict[str, WorkflowInputSpec] = Field(default_factory=dict)
    preprocessors: List[PromptSkillUse] = Field(default_factory=list, max_length=16)
    nodes: List[WorkflowNodeSpec] = Field(default_factory=list, max_length=64)
    outputs: Dict[str, ValueRef] = Field(default_factory=dict)
    template: Optional[WorkflowTemplateSpec] = None
    metadata: Dict[str, Scalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self) -> "WorkflowSpec":
        if self.template is None and not self.nodes:
            raise ValueError("provide template or nodes")
        if self.template is not None and self.nodes:
            raise ValueError("template and nodes are mutually exclusive")
        if self.template is None and not self.outputs:
            raise ValueError("an explicit DAG must declare outputs")
        return self


class PromptSkillParameter(StrictModel):
    description: Optional[str] = Field(default=None, max_length=300)
    required: bool = False
    default: Optional[Scalar] = None
    choices: Optional[List[Scalar]] = Field(default=None, min_length=1, max_length=64)


class PromptSkillDefinition(StrictModel):
    id: Identifier = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
    description: str = Field(min_length=1, max_length=1000)
    template: str = Field(min_length=1, max_length=4096)
    parameters: Dict[str, PromptSkillParameter] = Field(default_factory=dict)


class PromptSkillApplyRequest(StrictModel):
    prompt: str = Field(max_length=4096)
    version: Optional[str] = Field(default=None, min_length=1, max_length=64)
    parameters: Dict[str, Scalar] = Field(default_factory=dict)


class PublicRunRequest(StrictModel):
    inputs: Dict[str, Any]
    timeout: Optional[float] = Field(default=None, gt=0)
    profiled_latency: Optional[float] = Field(default=None, ge=0)
