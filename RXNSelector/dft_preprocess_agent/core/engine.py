from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dft_preprocess_agent.core.registry import SkillRegistry
from dft_preprocess_agent.core.state import WorkflowState


DEFAULT_SKILL_SEQUENCE = [
    "dataset_profile",
    "grouped_correlation",
    "mutual_information",
    "ridge_dependency_graph",
    "llm_chain_decomposition",
    "feature_selection_from_chains",
    "residual_collinearity_check",
    "final_feature_export",
]


class WorkflowEngine:
    def __init__(
        self,
        skills_dir: str | Path = "skills",
        runs_dir: str | Path = "runs",
    ) -> None:
        self.registry = SkillRegistry(skills_dir)
        self.runs_dir = Path(runs_dir)

    def create_run(
        self,
        dataset_path: str | Path,
        config_path: str | Path,
        run_id: str | None = None,
    ) -> WorkflowState:
        dataset_path = Path(dataset_path).resolve()
        config_path = Path(config_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)

        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (self.runs_dir / run_id).resolve()
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        copied_dataset = input_dir / dataset_path.name
        if dataset_path != copied_dataset:
            shutil.copy2(dataset_path, copied_dataset)

        state = WorkflowState(
            run_id=run_id,
            run_dir=str(run_dir),
            input_dataset_path=str(copied_dataset),
            current_dataset_path=str(copied_dataset),
            config_path=str(config_path),
        )
        state.save()
        return state

    def load_config(self, state: WorkflowState) -> dict[str, Any]:
        return yaml.safe_load(Path(state.config_path).read_text(encoding="utf-8"))

    def run_skill(
        self,
        state: WorkflowState,
        skill_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.load_config(state)
        tool = self.registry.load_tool(skill_name)
        context = {
            "run_id": state.run_id,
            "run_dir": state.run_dir,
            "state": state.to_dict(),
            "config": config,
            "params": params or {},
        }
        result = tool.run(context)
        state.add_step(skill_name, result)
        state.save()
        return result

    def run_workflow(
        self,
        state: WorkflowState,
        skill_sequence: list[str] | None = None,
        params_by_skill: dict[str, dict[str, Any]] | None = None,
    ) -> WorkflowState:
        for skill_name in skill_sequence or DEFAULT_SKILL_SEQUENCE:
            self.run_skill(
                state,
                skill_name,
                (params_by_skill or {}).get(skill_name, {}),
            )
        return state
