from __future__ import annotations

from pathlib import Path
from html import escape

import numpy as np
import pandas as pd

from dft_preprocess_agent.screening.analysis import (
    ensure_dir,
    high_correlation_pairs,
    load_dataset,
    profile_from_state,
)

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


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_02_grouped_correlation")

    profile = profile_from_state(state)
    df = load_dataset(state["current_dataset_path"])
    target_column = profile["target_column"]
    grouped = profile["groups"]
    thresholds = config.get("correlation", {}).get("report_thresholds", [0.8])
    primary_threshold = float(config.get("correlation", {}).get("primary_threshold", 0.8))

    all_pairs = []
    summary_rows = []
    for group_name, columns in grouped.items():
        corr = df[columns].corr(method="pearson")
        corr_path = output_dir / f"matrix_{group_name}.csv"
        corr.to_csv(corr_path)
        image_path = output_dir / f"corr_{group_name}.svg"
        write_correlation_svg(image_path, corr, f"Internal Pearson Correlation: {group_name}")

        for threshold in thresholds:
            pairs = high_correlation_pairs(df, columns, float(threshold))
            off_diagonal = corr.abs().where(~np.eye(len(corr), dtype=bool)) if len(columns) >= 2 else None
            summary_rows.append(
                {
                    "group": group_name,
                    "feature_count": len(columns),
                    "threshold": threshold,
                    "high_pair_count": len(pairs),
                    "max_abs_r": float(off_diagonal.max().max()) if off_diagonal is not None else 0.0,
                    "mean_abs_r": float(off_diagonal.stack().mean())
                    if off_diagonal is not None
                    else 0.0,
                }
            )
        primary_pairs = high_correlation_pairs(df, columns, primary_threshold)
        if not primary_pairs.empty:
            primary_pairs.insert(0, "group", group_name)
            all_pairs.append(primary_pairs)

    pairs_df = (
        pd.concat(all_pairs, ignore_index=True)
        if all_pairs
        else pd.DataFrame(columns=["group", "feature_1", "feature_2", "pearson_r", "abs_r"])
    )
    pairs_path = output_dir / "high_correlation_pairs.csv"
    summary_path = output_dir / "correlation_summary.csv"
    pairs_df.to_csv(pairs_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return {
        "status": "success",
        "target_column": target_column,
        "artifacts": {
            "high_correlation_pairs": str(pairs_path),
            "correlation_summary": str(summary_path),
        },
        "message": f"Found {len(pairs_df)} high-correlation pairs at |r| >= {primary_threshold}.",
    }
