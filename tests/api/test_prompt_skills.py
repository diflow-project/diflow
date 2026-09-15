import pytest

from diflow.api.errors import WorkflowAPIError
from diflow.api.prompt_skills import PromptSkillRegistry
from diflow.api.schemas import PromptSkillDefinition, PromptSkillParameter


def test_system_prompt_skill_is_deterministic():
    registry = PromptSkillRegistry()
    parameters = {
        "subject": "a traveler",
        "pose": "looking over one shoulder",
    }

    first = registry.apply("portrait-composition", "cinematic portrait", parameters)
    second = registry.apply("portrait-composition", "cinematic portrait", parameters)

    assert first == second
    assert first["original_prompt"] == "cinematic portrait"
    assert first["skill_version"] == "1.0.0"
    assert "looking over one shoulder" in first["prompt"]


def test_user_prompt_skill_rejects_template_attribute_access():
    registry = PromptSkillRegistry()
    skill = PromptSkillDefinition(
        id="unsafe-template",
        version="1",
        description="Must be rejected",
        template="{prompt.__class__}",
    )

    with pytest.raises(WorkflowAPIError, match="Attribute") as error:
        registry.register(skill)

    assert error.value.code == "INVALID_PROMPT_SKILL"


def test_user_prompt_skill_validates_parameters_and_is_idempotent():
    registry = PromptSkillRegistry()
    skill = PromptSkillDefinition(
        id="camera-style",
        version="1",
        description="Adds a camera style",
        template="{prompt}, photographed with {camera}",
        parameters={
            "camera": PromptSkillParameter(required=True, choices=["35mm", "85mm"])
        },
    )

    first = registry.register(skill)
    second = registry.register(skill)
    assert first == second
    assert (
        registry.apply("camera-style", "a portrait", {"camera": "85mm"})["prompt"]
        == "a portrait, photographed with 85mm"
    )

    with pytest.raises(WorkflowAPIError) as error:
        registry.apply("camera-style", "a portrait", {"camera": "24mm"})
    assert error.value.code == "INVALID_PROMPT_PARAMETER"
