from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "generated" / "role_graph_dft_alignment.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "role_atom_canonicalization.yaml"

ROLE_SITE_HINTS = {
    "aryl_halide": {
        "C": ["ARYL_C_IPSO", "ARYL_ELECTRONIC_SUBSTITUENT", "ARYL_STERIC_SUBSTITUENT"],
        "Cl": ["LEAVING_GROUP_X"],
        "Br": ["LEAVING_GROUP_X"],
        "I": ["LEAVING_GROUP_X"],
    },
    "amine": {
        "N": ["AMINE_N", "AMINE_N_H"],
    },
    "organoboron": {
        "B": ["BORON_SITE"],
        "C": ["BORON_ATTACHED_CARBON"],
    },
    "ligand": {
        "P": ["LIGAND_DONOR_ATOM"],
        "N": ["LIGAND_DONOR_ATOM"],
        "O": ["LIGAND_DONOR_ATOM"],
        "S": ["LIGAND_DONOR_ATOM"],
        "C": ["LIGAND_STERIC_ENVIRONMENT"],
        "H": ["LIGAND_STERIC_ENVIRONMENT"],
    },
    "base": {
        "N": ["BASE_BASIC_SITE"],
        "O": ["BASE_BASIC_SITE"],
    },
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def canonicalize_role_atoms(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    alignment = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    result = {
        "artifact_id": "role_atom_canonicalization_v1",
        "description": (
            "Canonical role-level atom schemas derived from DFT atom labels and "
            "MoleCode alignment candidates. Feature values are not extracted here."
        ),
        "datasets": [
            canonicalize_dataset(dataset)
            for dataset in alignment.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(result, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def canonicalize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_key": dataset["dataset_key"],
        "roles": [
            canonicalize_role(dataset["dataset_key"], role)
            for role in dataset.get("roles", [])
        ],
    }


def canonicalize_role(dataset_key: str, role_record: dict[str, Any]) -> dict[str, Any]:
    role = role_record["role"]
    canonical_atoms = [
        build_canonical_atom(dataset_key, role, atom)
        for atom in role_record.get("dft_atom_anchor_map", [])
    ]
    return {
        "role": role,
        "representative_component_id": role_record.get("representative_component_id"),
        "representative_source_smiles": role_record.get("representative_source_smiles"),
        "descriptor_coverage": build_descriptor_coverage(canonical_atoms),
        "mapping_status_summary": dict(
            sorted(Counter(atom["mapping_status"] for atom in canonical_atoms).items())
        ),
        "canonical_role_atoms": canonical_atoms,
        "canonical_role_graph": {
            "nodes": [
                build_graph_node(atom)
                for atom in canonical_atoms
            ],
            "edges": [],
            "edge_status": "pending_markush_role_graph_unification",
        },
    }


def build_canonical_atom(
    dataset_key: str,
    role: str,
    aligned_atom: dict[str, Any],
) -> dict[str, Any]:
    atom_label = aligned_atom["atom_label"]
    element = aligned_atom["element"]
    canonical_atom_id = f"{dataset_key.lower()}:{role}:{atom_label}"
    candidates = aligned_atom.get("candidate_molecode_nodes", [])
    return {
        "canonical_atom_id": canonical_atom_id,
        "role": role,
        "dft_atom_label": atom_label,
        "element": element,
        "atom_index": aligned_atom["atom_index"],
        "descriptor_columns": aligned_atom["descriptor_columns"],
        "descriptor_names": aligned_atom["descriptor_names"],
        "likely_site_types": infer_likely_site_types(role, element),
        "mapping_status": aligned_atom["alignment_status"],
        "confidence": aligned_atom["confidence"],
        "mapping_reason": aligned_atom["reason"],
        "candidate_molecode_nodes": candidates,
        "candidate_count": len(candidates),
    }


def infer_likely_site_types(role: str, element: str) -> list[str]:
    return ROLE_SITE_HINTS.get(role, {}).get(element, [])


def build_descriptor_coverage(canonical_atoms: list[dict[str, Any]]) -> dict[str, Any]:
    descriptor_columns = [
        column
        for atom in canonical_atoms
        for column in atom.get("descriptor_columns", [])
    ]
    return {
        "canonical_atom_count": len(canonical_atoms),
        "descriptor_column_count": len(descriptor_columns),
        "descriptor_columns": descriptor_columns,
    }


def build_graph_node(canonical_atom: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_atom_id": canonical_atom["canonical_atom_id"],
        "dft_atom_label": canonical_atom["dft_atom_label"],
        "element": canonical_atom["element"],
        "likely_site_types": canonical_atom["likely_site_types"],
        "mapping_status": canonical_atom["mapping_status"],
        "confidence": canonical_atom["confidence"],
    }


def main() -> None:
    canonicalize_role_atoms()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
