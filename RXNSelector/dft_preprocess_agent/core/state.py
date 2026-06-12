from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkflowState:
    run_id: str
    run_dir: str
    input_dataset_path: str
    current_dataset_path: str
    config_path: str
    target_column: str | None = None
    selected_features: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.run_dir) / "workflow_state.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def add_step(self, skill_name: str, result: dict[str, Any]) -> None:
        self.steps.append(
            {
                "skill": skill_name,
                "status": result.get("status", "success"),
                "message": result.get("message", ""),
                "artifacts": result.get("artifacts", {}),
            }
        )
        for key, value in result.get("artifacts", {}).items():
            self.artifacts[key] = value
        if result.get("output_dataset_path"):
            self.current_dataset_path = result["output_dataset_path"]
        if result.get("target_column"):
            self.target_column = result["target_column"]
        if result.get("selected_features"):
            self.selected_features = result["selected_features"]
