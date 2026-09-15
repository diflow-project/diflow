from __future__ import annotations

import json
from string import Formatter
from typing import Any, Dict, Iterable, Optional, Tuple

from diflow.api.errors import WorkflowAPIError
from diflow.api.schemas import PromptSkillDefinition, PromptSkillParameter, Scalar

SYSTEM_PROMPT_SKILLS = (
    PromptSkillDefinition(
        id="portrait-composition",
        version="1.0.0",
        description=(
            "Expands a portrait prompt with explicit subject, wardrobe, scene, "
            "pose, and framing guidance while preserving the original intent."
        ),
        template=(
            "{prompt}. Subject: {subject}. Wardrobe: {wardrobe}. Scene: {scene}. "
            "Pose: {pose}. Framing: {framing}. Keep the pose natural: turn the "
            "torso for depth, bend joints slightly, align limbs with the visual "
            "flow, and preserve a clear silhouette."
        ),
        parameters={
            "subject": PromptSkillParameter(default="single subject"),
            "wardrobe": PromptSkillParameter(
                default="clothing consistent with the subject"
            ),
            "scene": PromptSkillParameter(default="a coherent environment"),
            "pose": PromptSkillParameter(default="a relaxed, natural pose"),
            "framing": PromptSkillParameter(default="balanced portrait composition"),
        },
    ),
)


class PromptSkillRegistry:
    """In-memory registry for deterministic, data-only prompt transforms."""

    def __init__(
        self,
        system_skills: Iterable[PromptSkillDefinition] = SYSTEM_PROMPT_SKILLS,
        *,
        max_user_skills: int = 64,
    ) -> None:
        self._skills: Dict[Tuple[str, str], PromptSkillDefinition] = {}
        self._system_keys = set()
        self._max_user_skills = max_user_skills
        for skill in system_skills:
            self._validate_template(skill)
            key = (skill.id, skill.version)
            self._skills[key] = skill
            self._system_keys.add(key)

    @staticmethod
    def _validate_template(skill: PromptSkillDefinition) -> None:
        allowed = {"prompt", *skill.parameters}
        try:
            parsed = list(Formatter().parse(skill.template))
        except ValueError as exc:
            raise WorkflowAPIError(
                "INVALID_PROMPT_SKILL",
                f"Invalid template: {exc}",
                path="template",
            ) from exc
        fields = set()
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if format_spec or conversion:
                raise WorkflowAPIError(
                    "INVALID_PROMPT_SKILL",
                    "Format specifiers and conversions are not allowed",
                    path="template",
                )
            if any(token in field_name for token in (".", "[", "]")):
                raise WorkflowAPIError(
                    "INVALID_PROMPT_SKILL",
                    "Attribute and item access are not allowed in templates",
                    path="template",
                )
            fields.add(field_name)
        unknown = sorted(fields - allowed)
        if unknown:
            raise WorkflowAPIError(
                "INVALID_PROMPT_SKILL",
                f"Unknown template parameters: {', '.join(unknown)}",
                path="template",
            )
        if "prompt" not in fields:
            raise WorkflowAPIError(
                "INVALID_PROMPT_SKILL",
                "Template must include {prompt}",
                path="template",
            )

    def register(self, skill: PromptSkillDefinition) -> Dict[str, Any]:
        self._validate_template(skill)
        key = (skill.id, skill.version)
        existing = self._skills.get(key)
        if existing is not None:
            if existing == skill:
                return self.describe(existing)
            raise WorkflowAPIError(
                "PROMPT_SKILL_CONFLICT",
                f"Prompt skill {skill.id!r} version {skill.version!r} already exists",
                status_code=409,
            )
        user_count = len(self._skills) - len(self._system_keys)
        if user_count >= self._max_user_skills:
            raise WorkflowAPIError(
                "PROMPT_SKILL_LIMIT",
                "The server prompt-skill limit has been reached",
                status_code=429,
                retryable=True,
            )
        self._skills[key] = skill
        return self.describe(skill)

    def resolve(
        self, skill_id: str, version: Optional[str] = None
    ) -> PromptSkillDefinition:
        if version is not None:
            skill = self._skills.get((skill_id, version))
            if skill is None:
                raise WorkflowAPIError(
                    "PROMPT_SKILL_NOT_FOUND",
                    f"Prompt skill {skill_id!r} version {version!r} was not found",
                    status_code=404,
                )
            return skill
        candidates = [
            skill
            for (candidate_id, _), skill in self._skills.items()
            if candidate_id == skill_id
        ]
        if not candidates:
            raise WorkflowAPIError(
                "PROMPT_SKILL_NOT_FOUND",
                f"Prompt skill {skill_id!r} was not found",
                status_code=404,
            )
        # Dicts preserve insertion order. "Latest" means the most recently
        # registered version, avoiding surprising lexicographic ordering such
        # as version "10" sorting before version "2".
        return candidates[-1]

    def apply(
        self,
        skill_id: str,
        prompt: str,
        parameters: Dict[str, Scalar],
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        skill = self.resolve(skill_id, version)
        unknown = sorted(set(parameters) - set(skill.parameters))
        if unknown:
            raise WorkflowAPIError(
                "UNKNOWN_PROMPT_PARAMETER",
                f"Unknown prompt-skill parameters: {', '.join(unknown)}",
                path="parameters",
            )
        resolved: Dict[str, Scalar] = {}
        for name, spec in skill.parameters.items():
            value = parameters.get(name, spec.default)
            if value is None and spec.required:
                raise WorkflowAPIError(
                    "MISSING_PROMPT_PARAMETER",
                    f"Prompt-skill parameter {name!r} is required",
                    path=f"parameters.{name}",
                )
            if value is None:
                value = ""
            if spec.choices is not None and value not in spec.choices:
                raise WorkflowAPIError(
                    "INVALID_PROMPT_PARAMETER",
                    f"Prompt-skill parameter {name!r} must be one of {spec.choices!r}",
                    path=f"parameters.{name}",
                )
            resolved[name] = value
        rendered = skill.template.format(prompt=prompt, **resolved).strip()
        if len(rendered) > 8192:
            raise WorkflowAPIError(
                "PROMPT_TOO_LONG",
                "The transformed prompt exceeds 8192 characters",
                path="prompt",
            )
        return {
            "original_prompt": prompt,
            "prompt": rendered,
            "skill_id": skill.id,
            "skill_version": skill.version,
            "parameters": resolved,
        }

    def list(self) -> list[Dict[str, Any]]:
        skills = sorted(self._skills.values(), key=lambda item: (item.id, item.version))
        return [self.describe(skill) for skill in skills]

    def describe(self, skill: PromptSkillDefinition) -> Dict[str, Any]:
        key = (skill.id, skill.version)
        result = skill.model_dump()
        result["source"] = "system" if key in self._system_keys else "user"
        result["digest"] = self.digest(skill)
        return result

    @staticmethod
    def digest(skill: PromptSkillDefinition) -> str:
        import hashlib

        canonical = json.dumps(
            skill.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
