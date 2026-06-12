from __future__ import annotations

from pathlib import Path

from dft_preprocess_agent.screening.analysis import (
    compute_mi_scores,
    ensure_dir,
    load_dataset,
    profile_from_state,
)


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_03_mutual_information")

    profile = profile_from_state(state)
    df = load_dataset(state["current_dataset_path"])
    target_column = profile["target_column"]
    feature_columns = profile["feature_names"]
    mi_scores = compute_mi_scores(
        df,
        feature_columns,
        target_column,
        random_state=int(config.get("random_state", 42)),
    )
    grouped = profile["groups"]
    rows = []
    global_ranks = mi_scores.rank(ascending=False, method="dense").astype(int)
    for group_name, columns in grouped.items():
        group_ranks = mi_scores[columns].rank(ascending=False, method="dense").astype(int)
        for feature in columns:
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "mi_score": float(mi_scores[feature]),
                    "mi_rank_in_group": int(group_ranks[feature]),
                    "mi_rank_global": int(global_ranks[feature]),
                    "target_column": target_column,
                }
            )
    output_path = output_dir / "mi_scores_by_group.csv"
    import pandas as pd

    pd.DataFrame(rows).sort_values(["group", "mi_rank_in_group"]).to_csv(
        output_path,
        index=False,
    )
    return {
        "status": "success",
        "target_column": target_column,
        "artifacts": {"mi_scores_by_group": str(output_path)},
        "message": f"Calculated MI scores for {len(feature_columns)} features.",
    }
