from __future__ import annotations

import json
from pathlib import Path
from html import escape

import pandas as pd

from dft_preprocess_agent.screening.analysis import ensure_dir, load_dataset
from dft_preprocess_agent.screening.selection import load_feature_list

try:
    from dft_preprocess_agent.screening.analysis import write_correlation_svg
except ImportError:
    def write_correlation_svg(path: str | Path, corr: pd.DataFrame, title: str) -> str:
        path = Path(path)
        labels = [str(label) for label in corr.columns]
        cell = 24
        left = 220
        top = 160
        width = left + len(labels) * cell + 40
        height = top + len(labels) * cell + 80

        def color(value: float) -> str:
            if pd.isna(value):
                return "#f3f4f6"
            value = max(-1.0, min(1.0, float(value)))
            if value >= 0:
                return f"rgb(255,{round(255 - 180 * value)},{round(255 - 180 * value)})"
            return f"rgb({round(255 - 180 * abs(value))},{round(255 - 120 * abs(value))},255)"

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="20" y="32" font-family="Arial" font-size="18" font-weight="700">{escape(title)}</text>',
        ]
        for i, label in enumerate(labels):
            parts.append(
                f'<text x="{left - 8}" y="{top + i * cell + 16}" font-family="Arial" '
                f'font-size="10" text-anchor="end">{escape(label)}</text>'
            )
            x = left + i * cell + 12
            parts.append(
                f'<text x="{x}" y="{top - 8}" font-family="Arial" font-size="10" '
                f'text-anchor="start" transform="rotate(-55 {x} {top - 8})">{escape(label)}</text>'
            )
        for row_index, row_label in enumerate(labels):
            for col_index, col_label in enumerate(labels):
                value = corr.loc[row_label, col_label]
                parts.append(
                    f'<rect x="{left + col_index * cell}" y="{top + row_index * cell}" '
                    f'width="{cell}" height="{cell}" fill="{color(value)}" '
                    'stroke="#e5e7eb" stroke-width="0.5"/>'
                )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return str(path)


def _high_pairs(df: pd.DataFrame, features: list[str], threshold: float) -> pd.DataFrame:
    if len(features) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r", "abs_r"])
    corr = df[features].corr(method="pearson")
    rows = []
    for idx, feature_1 in enumerate(features):
        for feature_2 in features[idx + 1 :]:
            value = corr.loc[feature_1, feature_2]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "pearson_r": float(value),
                        "abs_r": float(abs(value)),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r", "abs_r"])
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def _choose_removal(pairs: pd.DataFrame, mi_scores: dict[str, float]) -> tuple[str, str]:
    counts: dict[str, int] = {}
    for row in pairs.itertuples():
        counts[row.feature_1] = counts.get(row.feature_1, 0) + 1
        counts[row.feature_2] = counts.get(row.feature_2, 0) + 1
    max_count = max(counts.values())
    conflict_centers = [feature for feature, count in counts.items() if count == max_count]
    if max_count > 1:
        removed = min(conflict_centers, key=lambda feature: mi_scores.get(feature, 0.0))
        return (
            removed,
            f"Removed conflict-center variable causing {max_count} over-threshold pairs; tied candidates were ranked by lower MI.",
        )

    top_pair = pairs.iloc[0]
    candidates = [top_pair["feature_1"], top_pair["feature_2"]]
    removed = min(candidates, key=lambda feature: mi_scores.get(feature, 0.0))
    kept = candidates[0] if removed == candidates[1] else candidates[1]
    return (
        removed,
        f"Removed lower-MI variable from pair with abs(r)={top_pair['abs_r']:.3f}; retained {kept}.",
    )


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_07_residual_collinearity_check")

    df = load_dataset(state["current_dataset_path"])
    selected_path = state["artifacts"].get("initial_selected_features")
    mi_path = state["artifacts"].get("mi_scores_by_group")
    if not selected_path:
        raise ValueError("Initial selected features missing. Run feature_selection_from_chains first.")
    if not mi_path:
        raise ValueError("MI scores missing. Run mutual_information first.")

    selected = load_feature_list(selected_path)
    mi_df = pd.read_csv(mi_path)
    mi_scores = {row.feature: float(row.mi_score) for row in mi_df.itertuples()}
    threshold = float(config.get("residual_check", {}).get("threshold", 0.8))
    max_iterations = int(config.get("residual_check", {}).get("max_iterations", 100))
    iterations: list[dict] = []

    current = list(selected)
    for iteration in range(1, max_iterations + 1):
        pairs = _high_pairs(df, current, threshold)
        if pairs.empty:
            iterations.append(
                {
                    "iteration": iteration,
                    "over_threshold_pair_count": 0,
                    "removed_feature": "",
                    "remove_reason": "All retained feature pairs are below the threshold.",
                    "max_abs_corr_after": 0.0,
                }
            )
            break
        removed, reason = _choose_removal(pairs, mi_scores)
        current = [feature for feature in current if feature != removed]
        next_pairs = _high_pairs(df, current, threshold)
        iterations.append(
            {
                "iteration": iteration,
                "over_threshold_pair_count": int(len(pairs)),
                "removed_feature": removed,
                "remove_reason": reason,
                "max_abs_corr_after": float(next_pairs["abs_r"].max()) if not next_pairs.empty else 0.0,
            }
        )
    else:
        iterations.append(
            {
                "iteration": max_iterations,
                "over_threshold_pair_count": int(len(_high_pairs(df, current, threshold))),
                "removed_feature": "",
                "remove_reason": "Maximum iterations reached; manual review may be required.",
                "max_abs_corr_after": 0.0,
            }
        )

    final_path = output_dir / "final_features.json"
    iterations_path = output_dir / "residual_collinearity_iterations.csv"
    matrix_path = output_dir / "final_correlation_matrix.csv"
    image_path = output_dir / "final_correlation_graph.svg"

    final_path.write_text(
        json.dumps({"features": current}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(iterations).to_csv(iterations_path, index=False)
    corr = df[current].corr(method="pearson") if current else pd.DataFrame()
    corr.to_csv(matrix_path)
    write_correlation_svg(image_path, corr, "Final Feature Correlation")

    return {
        "status": "success",
        "selected_features": current,
        "artifacts": {
            "final_features": str(final_path),
            "residual_collinearity_iterations": str(iterations_path),
            "final_correlation_matrix": str(matrix_path),
            "final_correlation_graph": str(image_path),
        },
        "message": f"Residual compliance retained {len(current)} final features.",
    }
