from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dft_preprocess_agent.screening.analysis import (
    ensure_dir,
    get_feature_columns,
    get_target_column,
    load_dataset,
)
from dft_preprocess_agent.screening.selection import pairwise_dependency_screen


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_04_dependency_screening")

    df = load_dataset(state["current_dataset_path"])
    target_column = get_target_column(df, config)
    feature_columns = get_feature_columns(df, config)

    mi_path = state["artifacts"].get("mi_scores")
    if not mi_path:
        raise ValueError("MI scores artifact missing. Run mutual_information first.")
    mi_df = pd.read_csv(mi_path)
    mi_scores = mi_df.set_index("feature")["mi_score"]

    candidates, audit_df, components_df = pairwise_dependency_screen(
        df,
        feature_columns,
        mi_scores,
        config,
    )

    candidates_path = output_dir / "candidate_features.json"
    candidates_path.write_text(
        json.dumps({"features": candidates}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit_path = output_dir / "dependency_audit.csv"
    components_path = output_dir / "dependency_components.csv"
    audit_df.to_csv(audit_path, index=False)
    components_df.to_csv(components_path, index=False)

    return {
        "status": "success",
        "target_column": target_column,
        "selected_features": candidates,
        "artifacts": {
            "candidate_features": str(candidates_path),
            "dependency_audit": str(audit_path),
            "dependency_components": str(components_path),
        },
        "message": f"First-pass dependency screening retained {len(candidates)} features.",
    }
