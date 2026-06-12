from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


class SkillRegistry:
    def __init__(self, skills_dir: str | Path = "skills") -> None:
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, dict[str, Any]] = {}

    def discover(self) -> dict[str, dict[str, Any]]:
        self._skills.clear()
        for identity_path in sorted(self.skills_dir.glob("*/identity.yaml")):
            skill_dir = identity_path.parent
            tool_path = skill_dir / "tool.py"
            if not tool_path.exists():
                continue
            identity = yaml.safe_load(identity_path.read_text(encoding="utf-8"))
            name = identity.get("name", skill_dir.name)
            self._skills[name] = {
                "name": name,
                "identity": identity,
                "tool_path": str(tool_path),
                "skill_dir": str(skill_dir),
            }
        return self._skills

    def get(self, name: str) -> dict[str, Any]:
        if not self._skills:
            self.discover()
        if name not in self._skills:
            known = ", ".join(sorted(self._skills))
            raise KeyError(f"Unknown skill '{name}'. Known skills: {known}")
        return self._skills[name]

    def load_tool(self, name: str) -> ModuleType:
        info = self.get(name)
        module_name = f"_workflow_skill_{name}"
        spec = importlib.util.spec_from_file_location(module_name, info["tool_path"])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import skill tool: {info['tool_path']}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "run"):
            raise AttributeError(f"Skill '{name}' must expose run(context)")
        return module
