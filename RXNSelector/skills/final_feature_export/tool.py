from __future__ import annotations

from pathlib import Path

import pandas as pd

from dft_preprocess_agent.screening.analysis import ensure_dir, load_dataset, profile_from_state


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_08_final_feature_export")

    profile = profile_from_state(state)
    df = load_dataset(state["current_dataset_path"])
    target_column = profile["target_column"]
    selected = state.get("selected_features", [])
    if not selected:
        raise ValueError("No selected features in workflow state.")
    missing = [feature for feature in selected if feature not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing from dataset: {missing}")

    selected_dataset_path = output_dir / "selected_dataset.csv"
    df[selected + [target_column]].to_csv(selected_dataset_path, index=False)

    report_path = output_dir / "final_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Final Feature Screening Report",
                "",
                f"- Target column: `{target_column}`",
                f"- Final feature count: {len(selected)}",
                f"- Selected dataset: `{selected_dataset_path}`",
                "",
                "## Final Features",
                "",
                "```python",
                repr(selected),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "target_column": target_column,
        "selected_features": selected,
        "output_dataset_path": str(selected_dataset_path),
        "artifacts": {
            "selected_dataset": str(selected_dataset_path),
            "final_report": str(report_path),
        },
        "message": f"Exported screened dataset with {len(selected)} features.",
    }
