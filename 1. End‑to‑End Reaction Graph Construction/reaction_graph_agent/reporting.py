from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from reaction_graph_agent.llm import LLMConfig, LLMRationaleClient
from reaction_graph_agent.profiler import (
    build_review_items,
    profile_dataset_source,
    summarize_graph_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROSS_ROLE_PATH = ROOT / "generated" / "literature_cross_role_bonds.yaml"
DEFAULT_EVIDENCE_PATH = ROOT / "generated" / "literature_retrieval_evidence.yaml"


def build_structured_graph_report(
    dataset_keys: list[str],
    graph_csv_dir: Path,
    sample_output_dir: Path,
    dataset_sources: dict[str, Path],
    report_dir: Path,
    llm_config: LLMConfig | dict[str, Any] | None = None,
    cross_role_path: Path = DEFAULT_CROSS_ROLE_PATH,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMRationaleClient(llm_config)
    evidence_index = load_evidence_index(evidence_path)
    cross_role_index = load_cross_role_bond_index(cross_role_path)
    datasets = []
    human_control_points = []

    for dataset_key in dataset_keys:
        dataset_profile = profile_dataset_source(dataset_key, dataset_sources[dataset_key])
        graph_summary = summarize_graph_csv(graph_csv_dir, dataset_key)
        nodes = read_csv(graph_csv_dir / f"{dataset_key}_nodes.csv")
        edges = read_csv(graph_csv_dir / f"{dataset_key}_edges.csv")
        cross_role_relations = build_cross_role_relation_reports(
            dataset_key=dataset_key,
            edges=edges,
            evidence=evidence_index.get(dataset_key, []),
            cross_role_bonds=cross_role_index.get(dataset_key, []),
            llm=llm,
        )
        dataset_report = {
            "dataset_key": dataset_key,
            "input_dataset": dataset_profile,
            "gnn_data": {
                "format": "npz",
                "path": str(sample_output_dir / f"{dataset_key}_samples.npz"),
                "fields": [
                    "node_features",
                    "node_feature_mask",
                    "edge_index",
                    "edge_attr",
                    "adjacency",
                    "y",
                    "splits",
                    "node_ids",
                    "node_feature_names",
                    "edge_feature_names",
                ],
            },
            "nodes": nodes,
            "edges": edges,
            "connectivity": graph_summary,
            "cross_role_relations": cross_role_relations,
        }
        datasets.append(dataset_report)
        human_control_points.extend(
            build_human_control_points(
                {
                    "dataset_key": dataset_key,
                    "source_profile": dataset_profile,
                    "graph_summary": graph_summary,
                }
            )
        )

    result = {
        "artifact_id": "reaction_graph_agent_structured_report_v1",
        "interface_contract": {
            "inputs": [
                "datasets: mapping of dataset_key to user CSV path",
                "llm: optional user LLM API config for evidence/rationale text",
                "output_format: currently npz for GNN-readable data",
            ],
            "llm_boundary": (
                "The LLM may explain/retrieve rationale for cross-role relations, "
                "but it is not allowed to create atom IDs, covalent bonds, "
                "cross-role atom edges, or adjacency matrices."
            ),
        },
        "llm": llm.config.public_dict() if llm.config else {"enabled": False},
        "graph_outputs": {
            "gnn_format": "npz",
            "sample_output_dir": str(sample_output_dir),
            "graph_csv_dir": str(graph_csv_dir),
        },
        "report_outputs": {
            "structured_report_yaml": str(report_dir / "structured_graph_report.yaml"),
            "structured_report_json": str(report_dir / "structured_graph_report.json"),
            "human_control_points_csv": str(report_dir / "human_control_points.csv"),
            "cross_role_relation_report_csv": str(report_dir / "cross_role_relation_report.csv"),
        },
        "datasets": datasets,
        "human_control_points": human_control_points,
    }
    write_yaml(report_dir / "structured_graph_report.yaml", result)
    (report_dir / "structured_graph_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(report_dir / "human_control_points.csv", human_control_points)
    write_csv(
        report_dir / "cross_role_relation_report.csv",
        flatten_cross_role_relations(datasets),
    )
    return result


def build_cross_role_relation_reports(
    dataset_key: str,
    edges: list[dict[str, str]],
    evidence: list[dict[str, Any]],
    cross_role_bonds: list[dict[str, Any]],
    llm: LLMRationaleClient,
) -> list[dict[str, Any]]:
    reports = []
    for edge in edges:
        if edge["edge_scope"] != "cross_role":
            continue
        matched_evidence = match_evidence(edge, evidence)
        compiled_relation = match_cross_role_bond(edge, cross_role_bonds)
        relation = {
            "edge_index": edge["edge_index"],
            "source_node_id": edge["source_node_id"],
            "target_node_id": edge["target_node_id"],
            "relation_type": edge["relation_type"],
            "edge_feature_full_name": edge["edge_feature_full_name"],
            "weight": edge["weight"],
            "evidence_sources": split_semicolon(edge.get("evidence_sources", "")),
        }
        evidence_records = merge_evidence_records(
            retrieved_evidence=matched_evidence,
            compiled_relation=compiled_relation,
            relation=relation,
        )
        reports.append(
            {
                **relation,
                "llm_search_and_parse_reason": llm.explain_cross_role_relation(
                    relation=relation,
                    evidence=evidence_records,
                ),
                "compiled_cross_role_record": compiled_relation or {},
                "evidence_records": evidence_records,
            }
        )
    return reports


def match_evidence(
    edge: dict[str, str],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_ids = set(split_semicolon(edge.get("evidence_sources", "")))
    if source_ids:
        matched = [item for item in evidence if item.get("source_id") in source_ids]
        if matched:
            return matched
        curated = match_curated_literature(source_ids)
        if curated:
            return curated
    source_label = edge["source_node_id"].split(":")[-1]
    target_label = edge["target_node_id"].split(":")[-1]
    relation_type = edge["relation_type"]
    matched = []
    for item in evidence:
        supports = item.get("supports", [])
        support_text = " ".join(str(value) for value in supports)
        if relation_type in support_text and (
            source_label in support_text or target_label in support_text
        ):
            matched.append(item)
    return matched


def load_cross_role_bond_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    index: dict[str, list[dict[str, Any]]] = {}
    for dataset in payload.get("datasets", []):
        dataset_key = dataset.get("dataset_key")
        if not dataset_key:
            continue
        records = []
        for bond in dataset.get("cross_role_bonds", []):
            record = dict(bond)
            record["dataset_key"] = dataset_key
            record["reaction_type"] = dataset.get("reaction_type", "")
            records.append(record)
        index[dataset_key] = records
    return index


def match_cross_role_bond(
    edge: dict[str, str],
    cross_role_bonds: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source = edge["source_node_id"]
    target = edge["target_node_id"]
    relation_type = edge["relation_type"]
    for bond in cross_role_bonds:
        if bond.get("relation_type") != relation_type:
            continue
        bond_source = bond.get("source_fixed_atom_id")
        bond_target = bond.get("target_fixed_atom_id")
        same_direction = bond_source == source and bond_target == target
        reverse_direction = bond_source == target and bond_target == source
        if same_direction or reverse_direction:
            return summarize_cross_role_bond_record(bond)
    return None


def summarize_cross_role_bond_record(bond: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": bond.get("edge_id", ""),
        "dataset_key": bond.get("dataset_key", ""),
        "reaction_type": bond.get("reaction_type", ""),
        "source_fixed_atom_id": bond.get("source_fixed_atom_id", ""),
        "target_fixed_atom_id": bond.get("target_fixed_atom_id", ""),
        "source_role": bond.get("source_role", ""),
        "target_role": bond.get("target_role", ""),
        "source_dft_atom_label": bond.get("source_dft_atom_label", ""),
        "target_dft_atom_label": bond.get("target_dft_atom_label", ""),
        "relation_type": bond.get("relation_type", ""),
        "mechanism_step": bond.get("mechanism_step", ""),
        "evidence_summary": bond.get("evidence_summary", ""),
        "evidence_sources": bond.get("evidence_sources", []),
        "created_by": bond.get("created_by", ""),
        "validation_status": bond.get("validation_status", ""),
    }


def merge_evidence_records(
    retrieved_evidence: list[dict[str, Any]],
    compiled_relation: dict[str, Any] | None,
    relation: dict[str, Any],
) -> list[dict[str, Any]]:
    records = [dict(item) for item in retrieved_evidence]
    if compiled_relation:
        records.append(
            compiled_relation_evidence_record(
                compiled_relation=compiled_relation,
                relation=relation,
            )
        )
    return records


def compiled_relation_evidence_record(
    compiled_relation: dict[str, Any],
    relation: dict[str, Any],
) -> dict[str, Any]:
    source = compiled_relation.get("source_role") or relation["source_node_id"]
    target = compiled_relation.get("target_role") or relation["target_node_id"]
    relation_type = relation["relation_type"]
    source_label = compiled_relation.get("source_dft_atom_label", "")
    target_label = compiled_relation.get("target_dft_atom_label", "")
    support = f"{source}:{source_label} -> {target}:{target_label} {relation_type}"
    return {
        "source_id": compiled_relation.get("edge_id", ""),
        "record_type": "compiled_cross_role_parse",
        "title": "Compiled cross-role relation parse",
        "url": "",
        "retrieval_query": build_compiled_relation_query(compiled_relation, relation),
        "evidence_summary": compiled_relation.get("evidence_summary", ""),
        "supports": [support],
        "mechanism_step": compiled_relation.get("mechanism_step", ""),
        "evidence_source_ids": compiled_relation.get("evidence_sources", []),
        "created_by": compiled_relation.get("created_by", ""),
    }


def build_compiled_relation_query(
    compiled_relation: dict[str, Any],
    relation: dict[str, Any],
) -> str:
    parts = [
        compiled_relation.get("reaction_type", ""),
        relation.get("relation_type", ""),
        compiled_relation.get("mechanism_step", ""),
        compiled_relation.get("source_role", ""),
        compiled_relation.get("target_role", ""),
    ]
    return " ".join(part for part in parts if part)


def match_curated_literature(source_ids: set[str]) -> list[dict[str, Any]]:
    try:
        from literature_evidence_retriever import CURATED_LITERATURE
    except ImportError:
        return []
    return [
        dict(record)
        for record in CURATED_LITERATURE
        if record.get("source_id") in source_ids
    ]


def build_human_control_points(dataset_report: dict[str, Any]) -> list[dict[str, Any]]:
    control_points = []
    for item in build_review_items(dataset_report):
        control_points.append(
            {
                "control_point_id": item["review_item_id"],
                "dataset_key": item["dataset_key"],
                "control_type": item["item_type"],
                "severity": item["severity"],
                "target_id": item["target_id"],
                "why_it_matters": item["review_reason"],
                "suggested_user_check": item["recommended_action"],
                "agent_action": "reported_not_blocking",
                "status": "open",
                "payload": item["payload"],
            }
        )
    return control_points


def flatten_cross_role_relations(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for relation in dataset["cross_role_relations"]:
            rationale = relation["llm_search_and_parse_reason"]
            rows.append(
                {
                    "dataset_key": dataset["dataset_key"],
                    "edge_index": relation["edge_index"],
                    "source_node_id": relation["source_node_id"],
                    "target_node_id": relation["target_node_id"],
                    "relation_type": relation["relation_type"],
                    "weight": relation["weight"],
                    "evidence_source_ids": ";".join(relation.get("evidence_sources", [])),
                    "compiled_mechanism_step": relation.get(
                        "compiled_cross_role_record",
                        {},
                    ).get("mechanism_step", ""),
                    "llm_status": rationale.get("llm_status", ""),
                    "search_queries": ";".join(rationale.get("search_queries", [])),
                    "evidence_summary": rationale.get("evidence_summary", ""),
                    "mechanistic_rationale": rationale.get("mechanistic_rationale", ""),
                    "confidence": rationale.get("confidence", ""),
                    "limitations": ";".join(rationale.get("limitations", [])),
                }
            )
    return rows


def load_evidence_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        dataset["dataset_key"]: dataset.get("evidence", [])
        for dataset in payload.get("datasets", [])
    }


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


def split_semicolon(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(";") if item]
