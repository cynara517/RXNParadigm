from __future__ import annotations

import csv
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_ROLE_PREFIXES = [
    "amine",
    "halide",
    "aryl_halide",
    "boronic Acid",
    "organoboron",
    "ligand",
    "base",
    "solvent",
    "additive",
]


def build_agent_quality_report(
    dataset_keys: list[str],
    graph_csv_dir: Path,
    dataset_sources: dict[str, Path],
    report_dir: Path,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    review_items = []
    for dataset_key in dataset_keys:
        dataset_profile = profile_dataset_source(
            dataset_key=dataset_key,
            source_csv=dataset_sources[dataset_key],
        )
        graph_summary = summarize_graph_csv(graph_csv_dir, dataset_key)
        dataset_report = {
            "dataset_key": dataset_key,
            "source_profile": dataset_profile,
            "graph_summary": graph_summary,
        }
        datasets.append(dataset_report)
        review_items.extend(build_review_items(dataset_report))

    result = {
        "artifact_id": "reaction_graph_agent_quality_report_v1",
        "description": (
            "Controlled graph-construction report. Review items mark graph "
            "decisions that require human approval before being treated as locked."
        ),
        "datasets": datasets,
        "review_queue": review_items,
    }
    write_yaml(report_dir / "graph_quality_report.yaml", result)
    write_yaml(
        report_dir / "review_queue.yaml",
        {
            "artifact_id": "reaction_graph_agent_review_queue_v1",
            "review_items": review_items,
        },
    )
    write_csv(report_dir / "dataset_profile.csv", flatten_dataset_profiles(datasets))
    write_csv(report_dir / "graph_summary.csv", flatten_graph_summaries(datasets))
    write_csv(report_dir / "review_queue.csv", review_items)
    return result


def profile_dataset_source(dataset_key: str, source_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(source_csv, nrows=0)
    columns = list(df.columns)
    inferred_prefixes = infer_prefixes(columns)
    prefixes = sorted(set(DEFAULT_ROLE_PREFIXES) | inferred_prefixes)
    role_profiles = []
    for prefix in prefixes:
        matching = [column for column in columns if starts_with_prefix(column, prefix)]
        if not matching:
            continue
        atom_columns = [column for column in matching if "_." in column]
        global_columns = [column for column in matching if "_." not in column]
        role_profiles.append(
            {
                "role_prefix": prefix,
                "total_column_count": len(matching),
                "atom_descriptor_column_count": len(atom_columns),
                "global_descriptor_column_count": len(global_columns),
                "atom_descriptor_sample": atom_columns[:12],
                "global_descriptor_sample": global_columns[:12],
            }
        )

    return {
        "dataset_key": dataset_key,
        "source_csv": str(source_csv),
        "column_count": len(columns),
        "has_yield": "yield" in columns,
        "role_profiles": role_profiles,
    }


def infer_prefixes(columns: list[str]) -> set[str]:
    prefixes = set()
    for column in columns:
        if "_." in column:
            prefixes.add(column.split("_.", 1)[0])
    return prefixes


def starts_with_prefix(column: str, prefix: str) -> bool:
    return column.startswith(f"{prefix}_.") or column.startswith(f"{prefix}_")


def summarize_graph_csv(graph_csv_dir: Path, dataset_key: str) -> dict[str, Any]:
    nodes = read_csv(graph_csv_dir / f"{dataset_key}_nodes.csv")
    edges = read_csv(graph_csv_dir / f"{dataset_key}_edges.csv")
    node_role = {node["node_id"]: node["role"] for node in nodes}
    node_element = {node["node_id"]: node["element"] for node in nodes}
    degree: Counter[str] = Counter()
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        degree[source] += 1
        degree[target] += 1
        adjacency[source].append(target)
        adjacency[target].append(source)

    components = connected_components([node["node_id"] for node in nodes], adjacency)
    isolated_nodes = [node["node_id"] for node in nodes if degree[node["node_id"]] == 0]
    return {
        "dataset_key": dataset_key,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "role_counts": dict(Counter(node["role"] for node in nodes)),
        "element_counts": dict(Counter(node["element"] for node in nodes)),
        "edge_scope_counts": dict(Counter(edge["edge_scope"] for edge in edges)),
        "edge_feature_counts": dict(Counter(edge["edge_feature_full_name"] for edge in edges)),
        "isolated_nodes": isolated_nodes,
        "component_count": len(components),
        "components": [
            {
                "size": len(component),
                "roles": dict(Counter(node_role[node_id] for node_id in component)),
                "nodes": component,
            }
            for component in components
        ],
        "top_degree_nodes": [
            {
                "node_id": node_id,
                "role": node_role[node_id],
                "element": node_element[node_id],
                "degree": count,
            }
            for node_id, count in sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
        "cross_role_edges": [
            {
                "edge_index": edge["edge_index"],
                "source_node_id": edge["source_node_id"],
                "target_node_id": edge["target_node_id"],
                "relation_type": edge["relation_type"],
                "weight": edge["weight"],
                "evidence_sources": edge.get("evidence_sources", ""),
            }
            for edge in edges
            if edge["edge_scope"] == "cross_role"
        ],
    }


def connected_components(
    node_ids: list[str],
    adjacency: dict[str, list[str]],
) -> list[list[str]]:
    visited = set()
    components = []
    for node_id in node_ids:
        if node_id in visited:
            continue
        queue: deque[str] = deque([node_id])
        visited.add(node_id)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def build_review_items(dataset_report: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_key = dataset_report["dataset_key"]
    source_profile = dataset_report["source_profile"]
    graph_summary = dataset_report["graph_summary"]
    items = []

    for edge in graph_summary["cross_role_edges"]:
        items.append(
            review_item(
                dataset_key=dataset_key,
                item_type="cross_role_edge",
                severity="high",
                target_id=f"{edge['source_node_id']}--{edge['target_node_id']}",
                reason="Cross-role candidate relation requires human/mechanistic approval.",
                recommended_action="Approve, reject, or adjust relation_type/weight.",
                payload=edge,
            )
        )

    if graph_summary["isolated_nodes"]:
        items.append(
            review_item(
                dataset_key=dataset_key,
                item_type="isolated_nodes",
                severity="medium",
                target_id=f"{dataset_key}:isolated_nodes",
                reason="Some feature-bearing atoms do not pass messages to other atoms.",
                recommended_action=(
                    "Confirm they should remain readout-only nodes, or add reviewed "
                    "role/cross-role relations."
                ),
                payload={"isolated_nodes": graph_summary["isolated_nodes"]},
            )
        )

    if graph_summary["node_count"] <= 5:
        items.append(
            review_item(
                dataset_key=dataset_key,
                item_type="minimal_reaction_center_graph",
                severity="medium",
                target_id=f"{dataset_key}:node_count",
                reason="Graph is very small and may miss substituent or ligand environment.",
                recommended_action=(
                    "Consider expanding topology with MoleCode/RDKit structural nodes "
                    "and missing-DFT masks."
                ),
                payload={"node_count": graph_summary["node_count"]},
            )
        )

    role_counts = graph_summary["role_counts"]
    source_roles = {role["role_prefix"]: role for role in source_profile["role_profiles"]}
    for role_name, role_count in role_counts.items():
        role_profile = source_roles.get(role_name) or role_alias_profile(role_name, source_roles)
        if not role_profile:
            continue
        if role_count <= 1 and role_profile["global_descriptor_column_count"] > 0:
            items.append(
                review_item(
                    dataset_key=dataset_key,
                    item_type="sparse_role_topology",
                    severity="medium",
                    target_id=f"{dataset_key}:{role_name}",
                    reason=(
                        f"Role {role_name} has only {role_count} graph node(s), but "
                        "source data contains global descriptors for this role."
                    ),
                    recommended_action=(
                        "Decide whether to add role-level/global descriptor nodes, "
                        "or expand topology with missing-DFT atom nodes."
                    ),
                    payload={
                        "role": role_name,
                        "graph_node_count": role_count,
                        "source_profile": role_profile,
                    },
                )
            )
    return items


def role_alias_profile(
    role_name: str,
    source_roles: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    aliases = {
        "aryl_halide": ["halide"],
        "organoboron": ["boronic Acid"],
    }
    for alias in aliases.get(role_name, []):
        if alias in source_roles:
            return source_roles[alias]
    return None


def review_item(
    dataset_key: str,
    item_type: str,
    severity: str,
    target_id: str,
    reason: str,
    recommended_action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "review_item_id": f"{dataset_key}:{item_type}:{target_id}",
        "dataset_key": dataset_key,
        "item_type": item_type,
        "severity": severity,
        "target_id": target_id,
        "review_required": True,
        "review_status": "pending",
        "review_reason": reason,
        "recommended_action": recommended_action,
        "human_decision": "",
        "locked": False,
        "payload": yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip(),
    }


def flatten_dataset_profiles(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        profile = dataset["source_profile"]
        for role in profile["role_profiles"]:
            rows.append(
                {
                    "dataset_key": dataset["dataset_key"],
                    "source_csv": profile["source_csv"],
                    "column_count": profile["column_count"],
                    "has_yield": profile["has_yield"],
                    **role,
                    "atom_descriptor_sample": ";".join(role["atom_descriptor_sample"]),
                    "global_descriptor_sample": ";".join(role["global_descriptor_sample"]),
                }
            )
    return rows


def flatten_graph_summaries(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        summary = dataset["graph_summary"]
        rows.append(
            {
                "dataset_key": dataset["dataset_key"],
                "node_count": summary["node_count"],
                "edge_count": summary["edge_count"],
                "component_count": summary["component_count"],
                "isolated_node_count": len(summary["isolated_nodes"]),
                "role_counts": yaml.safe_dump(summary["role_counts"], sort_keys=True).strip(),
                "edge_scope_counts": yaml.safe_dump(summary["edge_scope_counts"], sort_keys=True).strip(),
                "edge_feature_counts": yaml.safe_dump(summary["edge_feature_counts"], sort_keys=True).strip(),
                "isolated_nodes": ";".join(summary["isolated_nodes"]),
            }
        )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
