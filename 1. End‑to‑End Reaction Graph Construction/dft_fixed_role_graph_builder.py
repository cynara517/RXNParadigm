from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CANONICAL_PATH = ROOT / "generated" / "role_atom_canonicalization.yaml"
DEFAULT_GRAPH_PATH = ROOT / "generated" / "role_canonical_graphs.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "dft_fixed_role_graphs.yaml"

BOND_ORDER_BY_TYPE = {
    "SINGLE": 1.0,
    "DOUBLE": 2.0,
    "TRIPLE": 3.0,
    "AROMATIC": 1.5,
    "DATIVE": 1.0,
    "DOUBLE_E": 2.0,
    "DOUBLE_Z": 2.0,
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def build_dft_fixed_role_graphs(
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    canonical = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
    role_graphs = yaml.safe_load(graph_path.read_text(encoding="utf-8"))
    graph_index = index_role_graphs(role_graphs)
    result = {
        "artifact_id": "dft_fixed_role_graphs_v1",
        "description": (
            "Feature-ready role graphs whose atom identities are fixed by DFT "
            "atom labels. MoleCode nodes are attached as structural anchors when "
            "available; descriptor values are not extracted here."
        ),
        "datasets": [
            build_dataset(dataset, graph_index)
            for dataset in canonical.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(result, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def build_dataset(
    canonical_dataset: dict[str, Any],
    graph_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    dataset_key = canonical_dataset["dataset_key"]
    return {
        "dataset_key": dataset_key,
        "roles": [
            build_role(dataset_key, role, graph_index)
            for role in canonical_dataset.get("roles", [])
        ],
    }


def build_role(
    dataset_key: str,
    canonical_role: dict[str, Any],
    graph_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    role = canonical_role["role"]
    role_graph = graph_index.get((dataset_key, role), {})
    fixed_atoms = [
        build_fixed_atom(atom)
        for atom in canonical_role.get("canonical_role_atoms", [])
    ]
    fixed_edges = build_fixed_edges_from_selected_anchors(
        fixed_atoms,
        role_graph.get("graph", {}).get("edges", []),
    )
    descriptor_index = build_descriptor_index(fixed_atoms)
    return {
        "role": role,
        "graph_identity_policy": "dft_atom_label_is_primary_atom_id",
        "representative_component_id": canonical_role.get("representative_component_id"),
        "representative_source_smiles": canonical_role.get("representative_source_smiles"),
        "fixed_atoms": fixed_atoms,
        "fixed_edges": fixed_edges,
        "descriptor_to_atom": descriptor_index,
        "coverage": {
            "fixed_atom_count": len(fixed_atoms),
            "descriptor_column_count": len(descriptor_index),
            "molecode_anchor_summary": summarize_anchor_methods(fixed_atoms),
            "bond_type_summary": summarize_bond_types(fixed_edges),
            "edge_count": len(fixed_edges),
        },
    }


def build_fixed_atom(canonical_atom: dict[str, Any]) -> dict[str, Any]:
    selected_anchor = select_molecode_anchor(canonical_atom)
    return {
        "fixed_atom_id": canonical_atom["canonical_atom_id"],
        "role": canonical_atom["role"],
        "dft_atom_label": canonical_atom["dft_atom_label"],
        "element": canonical_atom["element"],
        "atom_index": canonical_atom["atom_index"],
        "descriptor_columns": canonical_atom["descriptor_columns"],
        "descriptor_names": canonical_atom["descriptor_names"],
        "likely_site_types": canonical_atom.get("likely_site_types", []),
        "feature_anchor_status": "fixed_by_dft_atom_label",
        "molecode_anchor": selected_anchor,
    }


def select_molecode_anchor(canonical_atom: dict[str, Any]) -> dict[str, Any]:
    candidates = sorted(
        canonical_atom.get("candidate_molecode_nodes", []),
        key=lambda node: (node.get("node_index", 10**9), node.get("node_id", "")),
    )
    if not candidates:
        return {
            "anchor_status": "fixed_dft_only",
            "anchor_method": "no_molecode_candidate",
            "confidence": "none",
            "fixed_molecode_node": None,
            "candidate_count": 0,
        }

    atom_index = canonical_atom["atom_index"]
    element = canonical_atom["element"]
    exact = [
        candidate
        for candidate in candidates
        if candidate.get("element") == element
        and candidate.get("node_index") == atom_index
    ]
    if exact:
        return {
            "anchor_status": "fixed_to_molecode",
            "anchor_method": "element_and_index_match",
            "confidence": "high",
            "fixed_molecode_node": exact[0],
            "candidate_count": len(candidates),
        }

    same_element = [
        candidate for candidate in candidates if candidate.get("element") == element
    ]
    if atom_index <= len(same_element):
        return {
            "anchor_status": "fixed_to_molecode",
            "anchor_method": "same_element_ordinal_fallback",
            "confidence": "medium",
            "fixed_molecode_node": same_element[atom_index - 1],
            "candidate_count": len(candidates),
        }

    fallback = same_element[0] if same_element else candidates[0]
    return {
        "anchor_status": "fixed_to_molecode",
        "anchor_method": "first_candidate_fallback",
        "confidence": "low",
        "fixed_molecode_node": fallback,
        "candidate_count": len(candidates),
    }


def build_descriptor_index(fixed_atoms: list[dict[str, Any]]) -> dict[str, str]:
    descriptor_index = {}
    for atom in fixed_atoms:
        for column in atom.get("descriptor_columns", []):
            descriptor_index[column] = atom["fixed_atom_id"]
    return dict(sorted(descriptor_index.items()))


def build_fixed_edges_from_selected_anchors(
    fixed_atoms: list[dict[str, Any]],
    candidate_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    atoms_by_id = {atom["fixed_atom_id"]: atom for atom in fixed_atoms}
    selected_node_to_atom = {
        atom["molecode_anchor"]["fixed_molecode_node"]["node_id"]: atom
        for atom in fixed_atoms
        if atom["molecode_anchor"].get("fixed_molecode_node")
    }
    fixed_edges = []
    for candidate_edge in candidate_edges:
        source_atom = atoms_by_id.get(candidate_edge["source_canonical_atom_id"])
        target_atom = atoms_by_id.get(candidate_edge["target_canonical_atom_id"])
        if source_atom is None or target_atom is None:
            continue
        selected_evidence = select_evidence_for_fixed_atoms(
            source_atom,
            target_atom,
            candidate_edge.get("evidence_molecode_node_pairs", []),
            selected_node_to_atom,
        )
        if not selected_evidence:
            continue
        bond_records = [build_bond_record(evidence) for evidence in selected_evidence]
        fixed_edges.append(
            {
                "edge_id": make_fixed_edge_id(
                    source_atom["fixed_atom_id"],
                    target_atom["fixed_atom_id"],
                ),
                "source_fixed_atom_id": source_atom["fixed_atom_id"],
                "target_fixed_atom_id": target_atom["fixed_atom_id"],
                "edge_category": "COVALENT_BOND",
                "edge_status": "fixed_from_selected_molecode_anchors",
                "confidence": fixed_edge_confidence(source_atom, target_atom),
                "bond_records": bond_records,
                "bond_types": sorted({record["bond_type"] for record in bond_records}),
                "bond_orders": sorted({record["bond_order"] for record in bond_records}),
            }
        )
    return fixed_edges


def select_evidence_for_fixed_atoms(
    source_atom: dict[str, Any],
    target_atom: dict[str, Any],
    evidence_pairs: list[dict[str, Any]],
    selected_node_to_atom: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_source = source_atom["molecode_anchor"].get("fixed_molecode_node")
    selected_target = target_atom["molecode_anchor"].get("fixed_molecode_node")
    if not selected_source or not selected_target:
        return []

    selected_ids = {
        selected_source["node_id"],
        selected_target["node_id"],
    }
    selected_evidence = []
    for evidence in evidence_pairs:
        evidence_ids = {
            evidence["source_node_id"],
            evidence["target_node_id"],
        }
        if evidence_ids != selected_ids:
            continue
        if evidence["source_node_id"] not in selected_node_to_atom:
            continue
        if evidence["target_node_id"] not in selected_node_to_atom:
            continue
        selected_evidence.append(evidence)
    return selected_evidence


def build_bond_record(evidence: dict[str, Any]) -> dict[str, Any]:
    bond_type = evidence["bond_type"]
    return {
        "source_molecode_node_id": evidence["source_node_id"],
        "target_molecode_node_id": evidence["target_node_id"],
        "source_component_id": evidence["source_component_id"],
        "target_component_id": evidence["target_component_id"],
        "molecode_operator": evidence["operator"],
        "bond_type": bond_type,
        "bond_order": BOND_ORDER_BY_TYPE[bond_type],
        "is_aromatic": bond_type == "AROMATIC",
        "stereochemistry": bond_stereochemistry(bond_type),
    }


def bond_stereochemistry(bond_type: str) -> str | None:
    if bond_type == "DOUBLE_E":
        return "E"
    if bond_type == "DOUBLE_Z":
        return "Z"
    return None


def fixed_edge_confidence(
    source_atom: dict[str, Any],
    target_atom: dict[str, Any],
) -> str:
    confidences = {
        source_atom["molecode_anchor"]["confidence"],
        target_atom["molecode_anchor"]["confidence"],
    }
    if confidences == {"high"}:
        return "high"
    if "none" in confidences:
        return "none"
    if "low" in confidences:
        return "low"
    return "medium"


def make_fixed_edge_id(source_atom_id: str, target_atom_id: str) -> str:
    safe_source = source_atom_id.replace(":", "_")
    safe_target = target_atom_id.replace(":", "_")
    return f"fixed_edge:{safe_source}--{safe_target}"


def summarize_bond_types(fixed_edges: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for edge in fixed_edges:
        for bond_type in edge.get("bond_types", []):
            summary[bond_type] = summary.get(bond_type, 0) + 1
    return dict(sorted(summary.items()))


def summarize_anchor_methods(fixed_atoms: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for atom in fixed_atoms:
        method = atom["molecode_anchor"]["anchor_method"]
        summary[method] = summary.get(method, 0) + 1
    return dict(sorted(summary.items()))


def index_role_graphs(role_graphs: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (dataset["dataset_key"], role["role"]): role
        for dataset in role_graphs.get("datasets", [])
        for role in dataset.get("roles", [])
    }


def main() -> None:
    build_dft_fixed_role_graphs()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
