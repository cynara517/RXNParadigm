from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dft_preprocess_agent.core.engine import DEFAULT_SKILL_SEQUENCE, WorkflowEngine


def main() -> None:
    engine = WorkflowEngine(
        skills_dir=ROOT / "skills",
        runs_dir=ROOT / "runs",
    )
    state = engine.create_run(
        dataset_path=ROOT / "examples" / "00_01_data.csv",
        config_path=ROOT / "configs" / "reaction_dft_screening.yaml",
        run_id="reaction_dft_benchmark",
    )
    engine.run_workflow(state, skill_sequence=list(DEFAULT_SKILL_SEQUENCE))
    print(f"Workflow state: {Path(state.run_dir) / 'workflow_state.json'}")
    print(f"Selected features: {len(state.selected_features)}")


if __name__ == "__main__":
    main()
