from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CANONICAL_PATH = ROOT / "generated" / "role_atom_canonicalization.yaml"
DEFAULT_ALIGNMENT_PATH = ROOT / "generated" / "role_graph_dft_alignment.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "role_canonical_graphs.yaml"

EDGE_RE = re.compile(
    r"^\s*(?P<source>[A-Za-z0-9_]+)\s+"
    r"(?P<operator><-->|-->|===\|[EZ]\||---|===|-\.-)\s+"
    r"(?P<target>[A-Za-z0-9_]+)\s*$"
)

BOND_TYPE_BY_OPERATOR = {
    "---": "SINGLE",
    "===": "DOUBLE",
    "-.-": "TRIPLE",
    "-->": "DATIVE",
    "<-->": "AROMATIC",
    "===|E|": "DOUBLE_E",
    "===|Z|": "DOUBLE_Z",
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def build_role_canonical_graphs(
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
    alignment_path: Path = DEFAULT_ALIGNMENT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    canonical = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    alignment = yaml.safe_load(alignment_path.read_text(encoding="utf-8"))
    alignment_index = index_alignment_roles(alignment)
    result = {
        "artifact_id": "role_canonical_graphs_v1",
        "description": (
            "Role-level canonical graphs whose nodes are DFT-backed canonical "
            "atoms and whose edges are candidate bonds lifted from representative "
            "MoleCode graphs. Feature values are not extracted here."
        ),
        "datasets": [
            build_dataset_graphs(dataset, alignment_index)
            for dataset in canonical.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(result, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def build_dataset_graphs(
    dataset: dict[str, Any],
    alignment_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    dataset_key = dataset["dataset_key"]
    return {
        "dataset_key": dataset_key,
        "roles": [
            build_role_graph(dataset_key, role, alignment_index)
            for role in dataset.get("roles", [])
        ],
    }


def build_role_graph(
    dataset_key: str,
    canonical_role: dict[str, Any],
    alignment_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    role = canonical_role["role"]
    alignment_role = alignment_index.get((dataset_key, role), {})
    representative_graph = alignment_role.get("representative_molecode_graph")
    representative_edges = parse_molecode_edges(representative_graph or "")
    canonical_atoms = canonical_role.get("canonical_role_atoms", [])
    graph_nodes = canonical_role.get("canonical_role_graph", {}).get("nodes", [])
    graph_edges = build_canonical_edges(canonical_atoms, representative_edges)
    return {
        "role": role,
        "representative_component_id": canonical_role.get("representative_component_id"),
        "representative_source_smiles": canonical_role.get("representative_source_smiles"),
        "descriptor_coverage": canonical_role.get("descriptor_coverage", {}),
        "mapping_status_summary": canonical_role.get("mapping_status_summary", {}),
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
            "edge_status": summarize_edge_status(graph_edges, representative_graph),
        },
        "edge_status_summary": dict(
            sorted(Counter(edge["edge_status"] for edge in graph_edges).items())
        ),
    }


def parse_molecode_edges(graph: str) -> list[dict[str, str]]:
    edges = []
    for line in graph.splitlines():
        match = EDGE_RE.match(line)
        if not match:
            continue
        operator = match.group("operator")
        edges.append(
            {
                "source_node_id": match.group("source"),
                "target_node_id": match.group("target"),
                "operator": operator,
                "bond_type": BOND_TYPE_BY_OPERATOR[operator],
            }
        )
    return edges


def build_canonical_edges(
    canonical_atoms: list[dict[str, Any]],
    representative_edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    edge_lookup = build_edge_lookup(representative_edges)
    canonical_edges = []
    for source_index, source_atom in enumerate(canonical_atoms):
        for target_atom in canonical_atoms[source_index + 1 :]:
            evidence = find_edge_evidence(source_atom, target_atom, edge_lookup)
            if not evidence:
                continue
            canonical_edges.append(
                {
                    "edge_id": make_edge_id(
                        source_atom["canonical_atom_id"],
                        target_atom["canonical_atom_id"],
                    ),
                    "source_canonical_atom_id": source_atom["canonical_atom_id"],
                    "target_canonical_atom_id": target_atom["canonical_atom_id"],
                    "edge_category": "CANDIDATE_COVALENT_BOND",
                    "edge_status": determine_edge_status(source_atom, target_atom),
                    "confidence": determine_edge_confidence(source_atom, target_atom),
                    "bond_types": sorted({item["bond_type"] for item in evidence}),
                    "evidence_molecode_node_pairs": evidence,
                }
            )
    return canonical_edges


def build_edge_lookup(
    representative_edges: list[dict[str, str]]
) -> dict[frozenset[str], list[dict[str, str]]]:
    lookup: dict[frozenset[str], list[dict[str, str]]] = {}
    for edge in representative_edges:
        key = frozenset([edge["source_node_id"], edge["target_node_id"]])
        lookup.setdefault(key, []).append(edge)
    return lookup


def find_edge_evidence(
    source_atom: dict[str, Any],
    target_atom: dict[str, Any],
    edge_lookup: dict[frozenset[str], list[dict[str, str]]],
) -> list[dict[str, str]]:
    evidence = []
    for source_node in source_atom.get("candidate_molecode_nodes", []):
        for target_node in target_atom.get("candidate_molecode_nodes", []):
            source_node_id = source_node["node_id"]
            target_node_id = target_node["node_id"]
            if source_node_id == target_node_id:
                continue
            key = frozenset([source_node_id, target_node_id])
            for edge in edge_lookup.get(key, []):
                evidence.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "source_component_id": source_node["component_id"],
                        "target_component_id": target_node["component_id"],
                        "operator": edge["operator"],
                        "bond_type": edge["bond_type"],
                    }
                )
    return deduplicate_evidence(evidence)


def deduplicate_evidence(evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for item in evidence:
        key = tuple(sorted([item["source_node_id"], item["target_node_id"]])) + (
            item["operator"],
            item["bond_type"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def determine_edge_status(
    source_atom: dict[str, Any],
    target_atom: dict[str, Any],
) -> str:
    statuses = {source_atom["mapping_status"], target_atom["mapping_status"]}
    if statuses == {"exact_element_singleton"}:
        return "confirmed_representative_bond"
    return "candidate_representative_bond"


def determine_edge_confidence(
    source_atom: dict[str, Any],
    target_atom: dict[str, Any],
) -> str:
    confidences = {source_atom["confidence"], target_atom["confidence"]}
    if confidences == {"high"}:
        return "high"
    if "none" in confidences:
        return "none"
    if "low" in confidences:
        return "low"
    return "medium"


def make_edge_id(source_id: str, target_id: str) -> str:
    safe_source = source_id.replace(":", "_")
    safe_target = target_id.replace(":", "_")
    return f"edge:{safe_source}--{safe_target}"


def summarize_edge_status(
    graph_edges: list[dict[str, Any]],
    representative_graph: str | None,
) -> str:
    if graph_edges:
        return "candidate_edges_lifted_from_representative_molecode_graph"
    if representative_graph:
        return "no_canonical_edges_found_in_representative_molecode_graph"
    return "missing_representative_molecode_graph"


def index_alignment_roles(alignment: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (dataset["dataset_key"], role["role"]): role
        for dataset in alignment.get("datasets", [])
        for role in dataset.get("roles", [])
    }


def main() -> None:
    build_role_canonical_graphs()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
