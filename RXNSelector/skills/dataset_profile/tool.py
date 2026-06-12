from __future__ import annotations

from pathlib import Path

from dft_preprocess_agent.screening.analysis import (
    ensure_dir,
    get_feature_columns,
    get_target_column,
    group_features,
    load_dataset,
    write_json,
)


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_01_dataset_profile")

    df = load_dataset(state["current_dataset_path"])
    target_column = get_target_column(df, config)
    feature_columns = get_feature_columns(df, config)
    grouped = group_features(feature_columns, config)
    numeric_features = df[feature_columns].select_dtypes(include="number").columns.tolist()

    profile = {
        "dataset_rule": (
            "The first N-1 columns are feature columns and the final column is "
            "the prediction target. Header names are used as feature names for "
            "group parsing and LLM reaction reasoning."
        ),
        "dataset_path": state["current_dataset_path"],
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_features),
        "feature_names": feature_columns,
        "groups": grouped,
        "group_counts": {name: len(cols) for name, cols in grouped.items()},
    }
    profile_path = write_json(output_dir / "dataset_profile.json", profile)
    return {
        "status": "success",
        "target_column": target_column,
        "artifacts": {"dataset_profile": profile_path},
        "message": (
            f"Detected {len(feature_columns)} feature columns and final-column "
            f"target '{target_column}' across {len(grouped)} groups."
        ),
    }
