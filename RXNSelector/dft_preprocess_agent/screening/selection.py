from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from dft_preprocess_agent.screening.analysis import group_features


def _abs_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame(index=cols, columns=cols, dtype=float)
    return df[cols].corr(method="pearson").abs()


def _components(corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    columns = list(corr.columns)
    if not columns:
        return []
    adjacency: dict[str, set[str]] = {col: set() for col in columns}
    for idx, feature_1 in enumerate(columns):
        for feature_2 in columns[idx + 1 :]:
            value = corr.loc[feature_1, feature_2]
            if pd.notna(value) and value >= threshold:
                adjacency[feature_1].add(feature_2)
                adjacency[feature_2].add(feature_1)

    seen: set[str] = set()
    components: list[list[str]] = []
    for col in columns:
        if col in seen:
            continue
        queue: deque[str] = deque([col])
        seen.add(col)
        comp: list[str] = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(comp, key=columns.index))
    return components


def _preference_score(feature: str, patterns: list[str]) -> int:
    for idx, pattern in enumerate(patterns):
        if re.search(pattern, feature):
            return len(patterns) - idx
    return 0


def _component_centrality(feature: str, members: list[str], corr: pd.DataFrame) -> float:
    return sum(float(corr.loc[feature, other]) for other in members if other != feature)


def _choose_representative(
    members: list[str],
    corr: pd.DataFrame,
    mi_scores: pd.Series,
    preference_patterns: list[str],
) -> str:
    return max(
        members,
        key=lambda feature: (
            _preference_score(feature, preference_patterns),
            _component_centrality(feature, members, corr),
            float(mi_scores.get(feature, 0.0)),
        ),
    )


def _dense_group_select(
    columns: list[str],
    mi_scores: pd.Series,
    group_config: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    audit: list[dict[str, Any]] = []
    family_patterns = group_config.get("family_patterns", [])
    max_features = int(group_config.get("max_features", len(columns)))

    for family in family_patterns:
        candidates = [col for col in columns if re.search(family, col)]
        candidates = [col for col in candidates if col not in selected]
        if not candidates:
            continue
        picked = max(candidates, key=lambda col: float(mi_scores.get(col, 0.0)))
        selected.append(picked)
        audit.append(
            {
                "feature": picked,
                "status": "candidate",
                "reason": f"dense_group_family_representative:{family}",
                "representative": picked,
            }
        )
        if len(selected) >= max_features:
            break

    if len(selected) < max_features:
        for col in mi_scores[columns].sort_values(ascending=False).index:
            if col not in selected:
                selected.append(col)
                audit.append(
                    {
                        "feature": col,
                        "status": "candidate",
                        "reason": "dense_group_mi_fill",
                        "representative": col,
                    }
                )
            if len(selected) >= max_features:
                break

    for col in columns:
        if col in selected:
            continue
        audit.append(
            {
                "feature": col,
                "status": "removed",
                "reason": "dense_group_factor_collapse",
                "representative": ";".join(selected),
            }
        )
    return selected, audit


def pairwise_dependency_screen(
    df: pd.DataFrame,
    feature_columns: list[str],
    mi_scores: pd.Series,
    config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    grouped = group_features(feature_columns, config)
    selection_config = config.get("selection", {})
    threshold = float(selection_config.get("pairwise_threshold", 0.9))
    group_methods = selection_config.get("group_methods", {})
    preferences = selection_config.get("representative_preferences", {})

    selected: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for group_name, columns in grouped.items():
        method = group_methods.get(group_name, group_methods.get("default", "components"))
        if method == "dense_factor":
            group_config = selection_config.get("dense_groups", {}).get(group_name, {})
            group_selected, group_audit = _dense_group_select(columns, mi_scores, group_config)
            selected.extend(group_selected)
            for row in group_audit:
                row["group"] = group_name
                audit_rows.append(row)
            continue

        corr = _abs_corr(df, columns)
        components = _components(corr, threshold)
        preference_patterns = preferences.get(group_name, [])
        for comp_index, members in enumerate(components, start=1):
            representative = _choose_representative(
                members,
                corr,
                mi_scores,
                preference_patterns,
            )
            selected.append(representative)
            component_rows.append(
                {
                    "group": group_name,
                    "component_id": f"{group_name}_{comp_index}",
                    "threshold": threshold,
                    "representative": representative,
                    "members": json.dumps(members, ensure_ascii=False),
                    "size": len(members),
                }
            )
            for member in members:
                status = "candidate" if member == representative else "removed"
                reason = "component_representative" if status == "candidate" else "pairwise_dependency"
                audit_rows.append(
                    {
                        "group": group_name,
                        "feature": member,
                        "status": status,
                        "reason": reason,
                        "representative": representative,
                    }
                )

    ordered_selected = [col for col in feature_columns if col in set(selected)]
    return ordered_selected, pd.DataFrame(audit_rows), pd.DataFrame(component_rows)


def residual_collinearity_screen(
    df: pd.DataFrame,
    candidate_features: list[str],
    mi_scores: pd.Series,
    config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    grouped = group_features(candidate_features, config)
    residual_config = config.get("residual_check", {})
    threshold = float(residual_config.get("threshold", 0.8))
    skip_groups = set(residual_config.get("skip_groups", []))
    preferences = residual_config.get("representative_preferences", {})

    selected: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for group_name, columns in grouped.items():
        if group_name in skip_groups or len(columns) < 2:
            selected.extend(columns)
            for col in columns:
                audit_rows.append(
                    {
                        "group": group_name,
                        "feature": col,
                        "status": "selected",
                        "reason": "residual_check_skipped_or_singleton",
                        "representative": col,
                    }
                )
            continue

        corr = _abs_corr(df, columns)
        components = _components(corr, threshold)
        preference_patterns = preferences.get(group_name, [])
        for comp_index, members in enumerate(components, start=1):
            if len(members) == 1:
                selected.append(members[0])
                audit_rows.append(
                    {
                        "group": group_name,
                        "feature": members[0],
                        "status": "selected",
                        "reason": "no_residual_high_correlation",
                        "representative": members[0],
                    }
                )
                continue
            representative = _choose_representative(
                members,
                corr,
                mi_scores,
                preference_patterns,
            )
            selected.append(representative)
            for idx, feature_1 in enumerate(members):
                for feature_2 in members[idx + 1 :]:
                    value = corr.loc[feature_1, feature_2]
                    if pd.notna(value) and value >= threshold:
                        pair_rows.append(
                            {
                                "group": group_name,
                                "component_id": f"{group_name}_residual_{comp_index}",
                                "feature_1": feature_1,
                                "feature_2": feature_2,
                                "abs_r": value,
                                "threshold": threshold,
                            }
                        )
            for member in members:
                status = "selected" if member == representative else "removed"
                audit_rows.append(
                    {
                        "group": group_name,
                        "feature": member,
                        "status": status,
                        "reason": "residual_representative"
                        if status == "selected"
                        else "residual_collinearity",
                        "representative": representative,
                    }
                )

    ordered_selected = [col for col in candidate_features if col in set(selected)]
    return ordered_selected, pd.DataFrame(audit_rows), pd.DataFrame(pair_rows)


def load_feature_list(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("features", []))
    return list(data)
