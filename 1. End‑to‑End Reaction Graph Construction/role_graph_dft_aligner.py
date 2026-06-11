from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_MOLECODE_PATH = ROOT / "generated" / "role_molecode_summary.yaml"
DEFAULT_DFT_PATH = ROOT / "generated" / "dft_atom_variables.yaml"
DEFAULT_ONTOLOGY_PATH = ROOT / "ontology" / "site_ontology.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "role_graph_dft_alignment.yaml"

NODE_RE = re.compile(
    r"^\s+(?P<node_id>[A-Za-z0-9_]+_(?P<element>[A-Z][a-z]?)_(?P<node_index>\d+)(?:_[RS])?)\[(?P<label>[^\]]+)\]\s*$"
)

ROLE_SITE_TYPES = {
    "aryl_halide": ["ARYL_C_IPSO", "LEAVING_GROUP_X"],
    "amine": ["AMINE_N", "AMINE_N_H"],
    "organoboron": ["BORON_SITE", "BORON_ATTACHED_CARBON"],
    "ligand": ["LIGAND_DONOR_ATOM", "LIGAND_STERIC_ENVIRONMENT"],
    "base": ["BASE_BASIC_SITE"],
    "additive": [],
    "solvent": [],
    "product": [],
    "counterion": [],
    "ligand_fragment": ["LIGAND_STERIC_ENVIRONMENT"],
}


def align_role_graphs_to_dft_atoms(
    molecode_path: Path = DEFAULT_MOLECODE_PATH,
    dft_path: Path = DEFAULT_DFT_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    molecode = yaml.safe_load(molecode_path.read_text(encoding="utf-8"))
    dft = yaml.safe_load(dft_path.read_text(encoding="utf-8"))
    ontology = load_yaml_allowing_code_fence(ontology_path)
    site_types = ontology.get("site_types", {}) if isinstance(ontology, dict) else {}

    dft_index = index_dft_roles(dft)
    result = {
        "artifact_id": "role_graph_dft_alignment_v1",
        "description": (
            "Auditable candidate alignment between DFT atom descriptor labels "
            "and MoleCode nodes grouped by dataset and role. Feature values are "
            "not extracted in this artifact."
        ),
        "datasets": [
            align_dataset(dataset, dft_index, site_types)
            for dataset in molecode.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def align_dataset(
    molecode_dataset: dict[str, Any],
    dft_index: dict[tuple[str, str], dict[str, Any]],
    site_types: dict[str, Any],
) -> dict[str, Any]:
    dataset_key = molecode_dataset["dataset_key"]
    aligned_roles = []
    for role_record in molecode_dataset.get("roles", []):
        role = role_record["role"]
        dft_role = dft_index.get((dataset_key, role))
        if dft_role is None:
            continue
        aligned_roles.append(
            align_role(
                dataset_key=dataset_key,
                role=role,
                molecode_role=role_record,
                dft_role=dft_role,
                site_types=site_types,
            )
        )
    return {
        "dataset_key": dataset_key,
        "roles": aligned_roles,
    }


def align_role(
    dataset_key: str,
    role: str,
    molecode_role: dict[str, Any],
    dft_role: dict[str, Any],
    site_types: dict[str, Any],
) -> dict[str, Any]:
    components = molecode_role.get("molecode_components", [])
    parsed_components = [
        component
        for component in components
        if component.get("molecode_graph")
        and component.get("molecode_parser", {}).get("status") in {"parsed", "roundtrip_mismatch"}
    ]
    component_nodes = {
        component["component_id"]: parse_molecode_nodes(component)
        for component in parsed_components
    }
    representative = choose_representative_component(parsed_components, component_nodes)
    representative_nodes = component_nodes.get(
        representative.get("component_id") if representative else "", []
    )
    role_site_candidates = build_role_site_candidates(role, site_types)
    dft_atoms = [
        align_dft_atom(atom, representative_nodes, role_site_candidates)
        for atom in dft_role.get("atoms", [])
    ]
    return {
        "role": role,
        "molecode_component_count": len(components),
        "parsed_component_count": len(parsed_components),
        "representative_component_id": (
            representative.get("component_id") if representative else None
        ),
        "representative_source_smiles": (
            representative.get("source_smiles") if representative else None
        ),
        "representative_molecode_graph": (
            representative.get("molecode_graph") if representative else None
        ),
        "common_atom_summary": summarize_component_nodes(component_nodes),
        "role_site_candidates": role_site_candidates,
        "dft_atom_anchor_map": dft_atoms,
    }


def parse_molecode_nodes(component: dict[str, Any]) -> list[dict[str, Any]]:
    graph = component.get("molecode_graph") or ""
    nodes = []
    for line in graph.splitlines():
        match = NODE_RE.match(line)
        if not match:
            continue
        nodes.append(
            {
                "node_id": match.group("node_id"),
                "component_id": component["component_id"],
                "element": match.group("element"),
                "node_index": int(match.group("node_index")),
                "display_label": match.group("label"),
            }
        )
    return nodes


def choose_representative_component(
    components: list[dict[str, Any]],
    component_nodes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    parsed_components = [
        component
        for component in components
        if component.get("molecode_parser", {}).get("status") == "parsed"
    ]
    candidates = parsed_components or components
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda component: (
            len(component_nodes.get(component["component_id"], [])),
            component["component_id"],
        ),
    )


def summarize_component_nodes(
    component_nodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    per_component_counts = []
    aggregate = Counter()
    for component_id, nodes in sorted(component_nodes.items()):
        counts = Counter(node["element"] for node in nodes)
        aggregate.update(counts)
        per_component_counts.append(
            {
                "component_id": component_id,
                "atom_count": len(nodes),
                "element_counts": dict(sorted(counts.items())),
            }
        )

    element_ranges = {}
    all_elements = sorted(
        {
            element
            for record in per_component_counts
            for element in record["element_counts"]
        }
    )
    for element in all_elements:
        counts = [
            record["element_counts"].get(element, 0)
            for record in per_component_counts
        ]
        element_ranges[element] = {
            "min": min(counts),
            "max": max(counts),
        }

    return {
        "element_count_ranges": element_ranges,
        "component_count": len(component_nodes),
    }


def build_role_site_candidates(
    role: str,
    site_types: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []
    for site_type in ROLE_SITE_TYPES.get(role, []):
        site_record = site_types.get(site_type, {})
        candidates.append(
            {
                "site_type": site_type,
                "matching_strategy": site_record.get("matching_strategy"),
                "pattern_hint": site_record.get("pattern_hint"),
            }
        )
    return candidates


def align_dft_atom(
    dft_atom: dict[str, Any],
    representative_nodes: list[dict[str, Any]],
    role_site_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    element = dft_atom["element"]
    candidates = [
        node for node in representative_nodes if node["element"] == element
    ]
    if not candidates:
        status = "unmapped"
        confidence = "none"
        reason = "No representative MoleCode node has the same element."
    elif len(candidates) == 1:
        status = "exact_element_singleton"
        confidence = "high"
        reason = "Exactly one representative MoleCode node has the same element."
    elif has_site_candidate_for_element(element, role_site_candidates):
        status = "site_rule_candidate"
        confidence = "medium"
        reason = (
            "Multiple same-element MoleCode nodes exist, but role site ontology "
            "contains a chemically relevant candidate for this element."
        )
    else:
        status = "ambiguous_same_element"
        confidence = "low"
        reason = "Multiple representative MoleCode nodes have the same element."

    return {
        "atom_label": dft_atom["atom_label"],
        "element": element,
        "atom_index": dft_atom["atom_index"],
        "descriptor_columns": dft_atom["descriptor_columns"],
        "descriptor_names": dft_atom["descriptor_names"],
        "alignment_status": status,
        "confidence": confidence,
        "reason": reason,
        "candidate_molecode_nodes": candidates,
    }


def has_site_candidate_for_element(
    element: str,
    role_site_candidates: list[dict[str, Any]],
) -> bool:
    hints = " ".join(
        str(candidate.get("pattern_hint") or "")
        for candidate in role_site_candidates
    )
    if element == "B" and "B" in hints:
        return True
    if element == "P" and "P" in hints:
        return True
    if element == "N" and "N" in hints:
        return True
    if element == "C" and ("[C" in hints or "[c" in hints or "carbon" in hints.lower()):
        return True
    if element in {"Cl", "Br", "I"} and element in hints:
        return True
    return False


def index_dft_roles(dft: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (dataset["dataset_key"], role["role"]): role
        for dataset in dft.get("datasets", [])
        for role in dataset.get("roles", [])
    }


def load_yaml_allowing_code_fence(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    loaded = yaml.safe_load(text)
    return loaded or {}


def main() -> None:
    align_role_graphs_to_dft_atoms()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
