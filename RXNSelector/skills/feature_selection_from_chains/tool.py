from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dft_preprocess_agent.screening.analysis import ensure_dir, profile_from_state


def run(context: dict) -> dict:
    state = context["state"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_06_feature_selection_from_chains")

    profile = profile_from_state(state)
    ridge_path = state["artifacts"].get("ridge_dependency_report")
    chain_path = state["artifacts"].get("llm_chain_decomposition")
    mi_path = state["artifacts"].get("mi_scores_by_group")
    if not ridge_path or not chain_path or not mi_path:
        raise ValueError("Ridge report, chain decomposition, and MI scores are required.")

    ridge_report = json.loads(Path(ridge_path).read_text(encoding="utf-8"))
    chain_report = json.loads(Path(chain_path).read_text(encoding="utf-8"))
    mi_df = pd.read_csv(mi_path)
    mi_scores = {row.feature: float(row.mi_score) for row in mi_df.itertuples()}
    retained_from_chains = {
        item["feature"]
        for group in chain_report["groups"].values()
        for chain in group["chains"]
        for item in chain["retained_features"]
    }
    removed_from_chains = {
        item["feature"]
        for group in chain_report["groups"].values()
        for chain in group["chains"]
        for item in chain["removed_features"]
    }

    selected: list[str] = []
    rows: list[dict] = []
    for group_name, features in profile["groups"].items():
        for feature in features:
            variable_type = ridge_report["variable_types"][feature]["variable_type"]
            if variable_type == "completely_independent_variable":
                decision = "retain"
                reason = "Completely independent variables are retained by rule."
            elif feature in retained_from_chains:
                decision = "retain"
                reason = "Retained by chain decomposition using ridge structure and MI."
            elif feature in removed_from_chains:
                decision = "remove"
                reason = "Removed by chain decomposition using ridge structure and MI."
            else:
                decision = "needs_review"
                reason = "No chain decision was available for this non-independent variable."

            if decision == "retain":
                selected.append(feature)
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "variable_type": variable_type,
                    "decision": decision,
                    "mi_score": mi_scores.get(feature, 0.0),
                    "reason": reason,
                }
            )

    selected_path = output_dir / "initial_selected_features.json"
    table_path = output_dir / "feature_decision_table.csv"
    selected_path.write_text(
        json.dumps({"features": selected}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(table_path, index=False)

    return {
        "status": "success",
        "selected_features": selected,
        "artifacts": {
            "initial_selected_features": str(selected_path),
            "feature_decision_table": str(table_path),
        },
        "message": f"Initial chain-based selection retained {len(selected)} features.",
    }
