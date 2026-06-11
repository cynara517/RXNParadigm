from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError
from yaml import YAMLError

from skill_schema import Skill


class SkillLoaderError(ValueError):
    """Raised when a Skill file cannot be loaded or validated."""


def load_skill(path: Union[str, Path]) -> Skill:
    skill_path = Path(path)
    if not skill_path.exists():
        raise SkillLoaderError(f"Skill file does not exist: {skill_path}")
    if not skill_path.is_file():
        raise SkillLoaderError(f"Skill path is not a file: {skill_path}")

    try:
        with skill_path.open("r", encoding="utf-8") as skill_file:
            raw_skill = yaml.safe_load(skill_file)
    except YAMLError as exc:
        raise SkillLoaderError(f"Invalid YAML in Skill file {skill_path}: {exc}") from exc

    if not isinstance(raw_skill, dict):
        raise SkillLoaderError(
            f"Skill file must contain a YAML mapping/object: {skill_path}"
        )

    try:
        return Skill.parse_obj(raw_skill)
    except ValidationError as exc:
        raise SkillLoaderError(f"Invalid Skill schema in {skill_path}: {exc}") from exc
