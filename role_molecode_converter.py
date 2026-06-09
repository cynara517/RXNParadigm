from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem
from molecode import mermaid_to_mol, mol_to_mermaid, mol_to_smiles


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "generated" / "role_smiles_summary.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "role_molecode_summary.yaml"


def convert_role_smiles_to_molecode(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    source = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    result = {
        "artifact_id": "role_molecode_summary_v1",
        "molecode_format_version": "mermaid_graph_v0_1",
        "note": (
            "Each role SMILES is parsed with RDKit and serialized through the "
            "MoleCode mol_to_mermaid API. Invalid or unsupported SMILES are "
            "retained with structured parse errors."
        ),
        "datasets": [
            convert_dataset(dataset)
            for dataset in source.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def convert_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    dataset_key = dataset["dataset_key"]
    return {
        "dataset_key": dataset_key,
        "roles": [
            convert_role(dataset_key, role_record)
            for role_record in dataset.get("roles", [])
        ],
    }


def convert_role(dataset_key: str, role_record: dict[str, Any]) -> dict[str, Any]:
    role = role_record["role"]
    return {
        "role": role,
        "molecode_components": [
            make_molecode_component(dataset_key, role, index, smiles)
            for index, smiles in enumerate(role_record.get("smiles", []))
        ],
    }


def make_molecode_component(
    dataset_key: str,
    role: str,
    index: int,
    smiles: str,
) -> dict[str, Any]:
    component_id = make_component_id(dataset_key, role, index)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return failed_molecode_component(
            component_id=component_id,
            role=role,
            smiles=smiles,
            error_type="rdkit_parse_failure",
            message="RDKit could not parse source_smiles.",
        )

    try:
        canonical_smiles = Chem.CanonSmiles(smiles)
        molecode_graph = mol_to_mermaid(mol, name=component_id)
        roundtrip_mol = mermaid_to_mol(molecode_graph)
        roundtrip_smiles = (
            mol_to_smiles(roundtrip_mol)
            if roundtrip_mol is not None
            else None
        )
    except Exception as exc:
        return failed_molecode_component(
            component_id=component_id,
            role=role,
            smiles=smiles,
            error_type="molecode_parse_failure",
            message=str(exc),
        )

    roundtrip_ok = roundtrip_smiles == canonical_smiles
    return {
        "component_id": component_id,
        "role": role,
        "source_smiles": smiles,
        "molecode_parser": {
            "engine": "MoleCode",
            "status": "parsed" if roundtrip_ok else "roundtrip_mismatch",
        },
        "canonical_smiles": canonical_smiles,
        "roundtrip_smiles": roundtrip_smiles,
        "roundtrip_ok": roundtrip_ok,
        "molecode_graph": molecode_graph,
        "persistent_atom_id_prefix": f"{component_id}:a",
        "errors": [] if roundtrip_ok else [
            {
                "error_type": "roundtrip_mismatch",
                "message": "MoleCode round-trip SMILES did not match RDKit canonical SMILES.",
            }
        ],
    }


def make_component_id(dataset_key: str, role: str, index: int) -> str:
    safe_dataset = dataset_key.lower()
    safe_role = role.lower().replace(" ", "_")
    return f"{safe_dataset}:{safe_role}:m{index:04d}"


def failed_molecode_component(
    component_id: str,
    role: str,
    smiles: str,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "role": role,
        "source_smiles": smiles,
        "molecode_parser": {
            "engine": "MoleCode",
            "status": "failed",
        },
        "canonical_smiles": None,
        "roundtrip_smiles": None,
        "roundtrip_ok": False,
        "molecode_graph": None,
        "persistent_atom_id_prefix": f"{component_id}:a",
        "errors": [
            {
                "error_type": error_type,
                "message": message,
            }
        ],
    }


def main() -> None:
    convert_role_smiles_to_molecode()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
