from __future__ import annotations

import re
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def get_target_column(df: pd.DataFrame, config: dict[str, Any]) -> str:
    if df.empty or len(df.columns) < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")
    return str(df.columns[-1])


def get_feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    target = get_target_column(df, config)
    return [str(col) for col in df.columns if col != target]


def group_features(columns: list[str], config: dict[str, Any]) -> dict[str, list[str]]:
    groups_config = config.get("groups", {})
    grouped: dict[str, list[str]] = {name: [] for name in groups_config}
    other: list[str] = []
    for col in columns:
        matched = False
        for group_name, group_spec in groups_config.items():
            prefixes = group_spec.get("prefixes", [])
            patterns = group_spec.get("patterns", [])
            if any(col.startswith(prefix) for prefix in prefixes) or any(
                re.search(pattern, col) for pattern in patterns
            ):
                grouped[group_name].append(col)
                matched = True
                break
        if not matched:
            inferred = infer_group_name(col)
            if inferred:
                grouped.setdefault(inferred, []).append(col)
            else:
                other.append(col)
    if other:
        grouped["other"] = other
    return {name: cols for name, cols in grouped.items() if cols}


def infer_group_name(feature_name: str) -> str:
    known_multiword = ("aryl_halide",)
    for group_name in known_multiword:
        if feature_name.startswith(f"{group_name}_"):
            return group_name
    if "_" in feature_name:
        return feature_name.split("_", 1)[0]
    return "ungrouped"


def load_profile(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile_from_state(state: dict[str, Any]) -> dict[str, Any]:
    profile_path = state.get("artifacts", {}).get("dataset_profile")
    if not profile_path:
        raise ValueError("Dataset profile artifact missing. Run dataset_profile first.")
    return load_profile(profile_path)


def feature_columns_from_profile(profile: dict[str, Any]) -> list[str]:
    return list(profile["feature_names"])


def target_from_profile(profile: dict[str, Any]) -> str:
    return str(profile["target_column"])


def compute_mi_scores(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int = 42,
) -> pd.Series:
    X = df[feature_columns]
    y = df[target_column]
    scores = mutual_info_regression(
        X,
        y,
        discrete_features=False,
        random_state=random_state,
    )
    return pd.Series(scores, index=feature_columns, name="mi_score")


def high_correlation_pairs(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    if len(columns) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r", "abs_r"])
    corr = df[columns].corr(method="pearson")
    rows: list[dict[str, Any]] = []
    for idx, feature_1 in enumerate(columns):
        for feature_2 in columns[idx + 1 :]:
            value = corr.loc[feature_1, feature_2]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "pearson_r": value,
                        "abs_r": abs(value),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r", "abs_r"])
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def write_json(path: str | Path, data: dict[str, Any]) -> str:
    import json

    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def write_correlation_svg(path: str | Path, corr: pd.DataFrame, title: str) -> str:
    path = Path(path)
    if corr.empty:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120">'
            f'<text x="20" y="50" font-family="Arial" font-size="18">{escape(title)}</text>'
            '<text x="20" y="82" font-family="Arial" font-size="13">No correlation values available.</text>'
            "</svg>",
            encoding="utf-8",
        )
        return str(path)

    labels = [str(label) for label in corr.columns]
    n = len(labels)
    cell = 18 if n > 40 else 24 if n > 20 else 34
    ruler = 24
    left = 250
    top = 205
    right = 70
    bottom = 180
    width = left + n * cell + right
    height = top + n * cell + bottom

    def color(value: float) -> str:
        if pd.isna(value):
            return "#f3f4f6"
        value = max(-1.0, min(1.0, float(value)))
        if value >= 0:
            ratio = value
            r1, g1, b1 = 255, 255, 255
            r2, g2, b2 = 185, 28, 28
        else:
            ratio = abs(value)
            r1, g1, b1 = 255, 255, 255
            r2, g2, b2 = 37, 99, 235
        r = round(r1 + (r2 - r1) * ratio)
        g = round(g1 + (g2 - g1) * ratio)
        b = round(b1 + (b2 - b1) * ratio)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="32" font-family="Arial" font-size="20" font-weight="700">{escape(title)}</text>',
        '<text x="20" y="56" font-family="Arial" font-size="12" fill="#4b5563">'
        "Red = positive correlation; blue = negative correlation; darker = stronger absolute value."
        "</text>",
        '<text x="20" y="76" font-family="Arial" font-size="12" fill="#4b5563">'
        "Numbered rulers show row/column coordinates; rows and columns use the same feature order."
        "</text>",
    ]

    ruler_x = left - ruler
    ruler_y = top - ruler
    parts.append(
        f'<rect x="{ruler_x}" y="{ruler_y}" width="{ruler}" height="{ruler}" '
        'fill="#f9fafb" stroke="#9ca3af" stroke-width="0.8"/>'
    )
    parts.append(
        f'<text x="{ruler_x + ruler / 2}" y="{ruler_y + 16}" font-family="Arial" '
        'font-size="10" text-anchor="middle" fill="#374151">#</text>'
    )

    for idx, label in enumerate(labels):
        x = left + idx * cell + cell / 2
        y = top - 8
        safe_label = escape(label)
        parts.append(
            f'<rect x="{left + idx * cell}" y="{ruler_y}" width="{cell}" height="{ruler}" '
            'fill="#f9fafb" stroke="#9ca3af" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{ruler_y + 16}" font-family="Arial" font-size="10" '
            f'text-anchor="middle" fill="#374151">{idx + 1}</text>'
        )
        parts.append(
            f'<rect x="{ruler_x}" y="{top + idx * cell}" width="{ruler}" height="{cell}" '
            'fill="#f9fafb" stroke="#9ca3af" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{ruler_x + ruler / 2}" y="{top + idx * cell + cell * 0.65:.1f}" '
            f'font-family="Arial" font-size="10" text-anchor="middle" fill="#374151">{idx + 1}</text>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y}" font-family="Arial" font-size="10" '
            f'text-anchor="start" transform="rotate(-55 {x:.1f} {y})">{safe_label}</text>'
        )
        row_y = top + idx * cell + cell * 0.7
        parts.append(
            f'<text x="{left - 8}" y="{row_y:.1f}" font-family="Arial" font-size="10" '
            f'text-anchor="end">{safe_label}</text>'
        )

    for row_index, row_label in enumerate(labels):
        for col_index, col_label in enumerate(labels):
            value = corr.loc[row_label, col_label]
            x = left + col_index * cell
            y = top + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{color(value)}" stroke="#e5e7eb" stroke-width="0.5">'
                f'<title>{escape(row_label)} vs {escape(col_label)}: {float(value):.4f}</title>'
                "</rect>"
            )

    mapping_x = left + n * cell + 18
    mapping_y = top
    parts.append(
        f'<text x="{mapping_x}" y="{mapping_y - 10}" font-family="Arial" '
        'font-size="12" font-weight="700">Index</text>'
    )
    visible_mapping = labels[: min(n, 35)]
    for index, label in enumerate(visible_mapping, start=1):
        y = mapping_y + (index - 1) * 15
        parts.append(
            f'<text x="{mapping_x}" y="{y}" font-family="Arial" font-size="10" fill="#374151">'
            f'{index}. {escape(label)}</text>'
        )
    if n > len(visible_mapping):
        parts.append(
            f'<text x="{mapping_x}" y="{mapping_y + len(visible_mapping) * 15}" '
            'font-family="Arial" font-size="10" fill="#6b7280">... see axis labels for remaining features</text>'
        )

    legend_x = left
    legend_y = top + n * cell + 38
    parts.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-family="Arial" font-size="12">Scale</text>')
    legend_values = [-1, -0.5, 0, 0.5, 1]
    for index, value in enumerate(legend_values):
        x = legend_x + index * 58
        parts.append(
            f'<rect x="{x}" y="{legend_y}" width="52" height="16" fill="{color(value)}" '
            'stroke="#d1d5db" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x + 26}" y="{legend_y + 32}" font-family="Arial" font-size="11" '
            f'text-anchor="middle">{value}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)
