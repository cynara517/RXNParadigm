from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from dft_preprocess_agent.screening.analysis import ensure_dir, load_dataset, profile_from_state


def _standardize(values: pd.DataFrame | pd.Series) -> np.ndarray:
    data = values.to_frame() if isinstance(values, pd.Series) else values
    return StandardScaler().fit_transform(data.astype(float))


def _components(nodes: list[str], edges: list[dict]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        adjacency[edge["feature_1"]].add(edge["feature_2"])
        adjacency[edge["feature_2"]].add(edge["feature_1"])
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        queue: deque[str] = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(component) > 1:
            components.append(sorted(component, key=nodes.index))
    return components


def _formula(intercept: float, coefficients: dict[str, float]) -> str:
    parts = [f"{intercept:.4f}"]
    for feature, coef in coefficients.items():
        sign = "+" if coef >= 0 else "-"
        parts.append(f"{sign} {abs(coef):.4f} * [{feature}]")
    return " ".join(parts)


def _short_name(feature: str, group_name: str) -> str:
    prefix = f"{group_name}_"
    return feature[len(prefix) :] if feature.startswith(prefix) else feature


def _render_relation(features: list[str], edges: list[dict], group_name: str) -> str:
    if not features:
        return ""
    if len(features) == 1:
        return _short_name(features[0], group_name)
    feature_set = set(features)
    component_edges = [
        edge
        for edge in edges
        if edge["feature_1"] in feature_set and edge["feature_2"] in feature_set
    ]
    if not component_edges:
        return " ; ".join(_short_name(feature, group_name) for feature in features)
    relation_parts = []
    for edge in sorted(component_edges, key=lambda item: -item["abs_ridge_value"]):
        left = _short_name(edge["feature_1"], group_name)
        right = _short_name(edge["feature_2"], group_name)
        value = float(edge["ridge_value"])
        relation_parts.append(f"{left} ↔<sup>{value:.3f}</sup> {right}")
    return " ; ".join(relation_parts)


def run(context: dict) -> dict:
    state = context["state"]
    config = context["config"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_04_ridge_dependency_graph")

    profile = profile_from_state(state)
    df = load_dataset(state["current_dataset_path"])
    grouped = profile["groups"]
    ridge_config = config.get("ridge_dependency", {})
    alpha = float(ridge_config.get("alpha", 1.0))
    edge_threshold = float(ridge_config.get("edge_threshold", 0.9))
    multicollinearity_r2_threshold = float(
        ridge_config.get("multicollinearity_r2_threshold", 0.9)
    )
    contributor_threshold = float(ridge_config.get("contributor_threshold", 0.05))
    min_contributors = int(ridge_config.get("min_multicollinearity_contributors", 2))

    report = {
        "parameters": {
            "alpha": alpha,
            "edge_threshold": edge_threshold,
            "multicollinearity_r2_threshold": multicollinearity_r2_threshold,
            "contributor_threshold": contributor_threshold,
            "min_multicollinearity_contributors": min_contributors,
        },
        "groups": {},
        "variable_types": {},
    }
    markdown_lines = ["# Ridge Dependency Report", ""]

    for group_name, columns in grouped.items():
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        group_edges: list[dict] = []
        formulas: dict[str, dict] = {}

        for idx, feature_1 in enumerate(numeric_columns):
            for feature_2 in numeric_columns[idx + 1 :]:
                x = _standardize(df[feature_1])
                y = _standardize(df[feature_2]).ravel()
                model = Ridge(alpha=alpha)
                model.fit(x, y)
                coef = float(model.coef_[0])
                if abs(coef) >= edge_threshold:
                    group_edges.append(
                        {
                            "group": group_name,
                            "feature_1": feature_1,
                            "feature_2": feature_2,
                            "ridge_value": coef,
                            "abs_ridge_value": abs(coef),
                            "direction": "bidirectional",
                        }
                    )

        components = _components(numeric_columns, group_edges)
        chain_members = {feature for component in components for feature in component}
        multicollinear_members: set[str] = set()

        for target_feature in numeric_columns:
            predictors = [col for col in numeric_columns if col != target_feature]
            if not predictors:
                continue
            x = _standardize(df[predictors])
            y = _standardize(df[target_feature]).ravel()
            model = Ridge(alpha=alpha)
            model.fit(x, y)
            prediction = model.predict(x)
            r2 = float(r2_score(y, prediction))
            coefficients = {
                feature: float(coef)
                for feature, coef in zip(predictors, model.coef_)
                if abs(float(coef)) >= contributor_threshold
            }
            sorted_coefficients = dict(
                sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)
            )
            is_multicollinear = (
                r2 >= multicollinearity_r2_threshold
                and len(sorted_coefficients) >= min_contributors
            )
            if is_multicollinear:
                multicollinear_members.add(target_feature)
            formulas[target_feature] = {
                "ridge_r2": r2,
                "is_multicollinear": is_multicollinear,
                "contributors": sorted_coefficients,
                "formula": _formula(0.0, sorted_coefficients),
            }

        feature_types = {}
        for feature in columns:
            if feature in multicollinear_members:
                variable_type = "multicollinear_variable"
            elif feature in chain_members:
                variable_type = "chain_correlated_variable"
            else:
                variable_type = "completely_independent_variable"
            feature_types[feature] = variable_type
            report["variable_types"][feature] = {
                "group": group_name,
                "variable_type": variable_type,
            }

        raw_components = [
            {
                "component_id": f"{group_name}_component_{index}",
                "features": component,
                "rendered_relation": _render_relation(component, group_edges, group_name),
            }
            for index, component in enumerate(components, start=1)
        ]

        report["groups"][group_name] = {
            "features": columns,
            "numeric_features": numeric_columns,
            "ridge_edges": group_edges,
            "raw_components": raw_components,
            "formulas": formulas,
            "feature_types": feature_types,
        }

        markdown_lines.extend([f"## {group_name}", ""])
        markdown_lines.append(f"- Features: {len(columns)}")
        markdown_lines.append(f"- Ridge edges: {len(group_edges)}")
        markdown_lines.append(f"- Raw components: {len(components)}")
        markdown_lines.append("")
        for index, component in enumerate(raw_components, start=1):
            markdown_lines.append(f"{index}. {component['rendered_relation']}")
        markdown_lines.append("")

    json_path = output_dir / "ridge_dependency_report.json"
    md_path = output_dir / "ridge_dependency_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    return {
        "status": "success",
        "artifacts": {
            "ridge_dependency_report": str(json_path),
            "ridge_dependency_report_md": str(md_path),
        },
        "message": "Built ridge dependency structures and base variable types.",
    }
