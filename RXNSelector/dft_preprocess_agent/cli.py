from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dft_preprocess_agent.core.engine import DEFAULT_SKILL_SEQUENCE, WorkflowEngine


ROOT = Path(__file__).resolve().parents[1]


def project_path(name: str) -> Path:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find '{name}' at {path}. RXNSelector currently expects a "
            "source-tree or editable install so that project folders such as "
            "'skills', 'configs', and 'ui' are available."
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RXNSelector Skill workflow. Input datasets must use "
            "the strict format: first N-1 columns are features and the final "
            "column is the prediction target. Header names are used for feature "
            "names, group parsing, and LLM reaction reasoning."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a complete workflow.")
    run_parser.add_argument("--dataset", required=True, help="Input CSV/XLSX/Parquet dataset.")
    run_parser.add_argument("--config", required=True, help="Workflow YAML config.")
    run_parser.add_argument("--run-id", default=None, help="Optional run id.")
    run_parser.add_argument(
        "--expected",
        default=None,
        help="Optional expected feature YAML/JSON for benchmark validation.",
    )

    subparsers.add_parser("ui", help="Launch the RXNSelector Streamlit UI.")
    return parser


def ui_main() -> None:
    ui_dir = project_path("ui")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_dir / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        engine = WorkflowEngine(skills_dir=project_path("skills"), runs_dir=ROOT / "runs")
        state = engine.create_run(args.dataset, args.config, run_id=args.run_id)
        params_by_skill = {}
        skill_sequence = list(DEFAULT_SKILL_SEQUENCE)
        if args.expected:
            params_by_skill["benchmark_validate"] = {"expected_features_path": args.expected}
            skill_sequence.append("benchmark_validate")
        engine.run_workflow(state, skill_sequence, params_by_skill=params_by_skill)
        print(f"Run finished: {state.run_id}")
        print(f"State: {Path(state.run_dir) / 'workflow_state.json'}")
        print(f"Selected features: {len(state.selected_features)}")
    elif args.command == "ui":
        ui_main()


if __name__ == "__main__":
    main()
