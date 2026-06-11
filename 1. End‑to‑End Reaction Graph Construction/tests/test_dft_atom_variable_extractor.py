from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dft_atom_variable_extractor import (  # noqa: E402
    extract_dft_atom_variables,
    normalize_role,
    parse_atom_descriptor_column,
)


def test_parse_atom_descriptor_column() -> None:
    parsed = parse_atom_descriptor_column("aryl_halide_.C1_NMR_shift")

    assert parsed == {
        "raw_role": "aryl_halide",
        "atom_label": "C1",
        "element": "C",
        "atom_index": "1",
        "descriptor_name": "NMR_shift",
    }


def test_parse_column_with_spaced_role() -> None:
    parsed = parse_atom_descriptor_column("boronic Acid_.B1_electrostatic_charge")

    assert parsed["raw_role"] == "boronic Acid"
    assert parsed["atom_label"] == "B1"
    assert parsed["descriptor_name"] == "electrostatic_charge"


def test_ignores_non_atom_descriptor_column() -> None:
    assert parse_atom_descriptor_column("ligand_E_HOMO") is None


def test_normalizes_dataset_role_aliases() -> None:
    assert normalize_role("halide") == "aryl_halide"
    assert normalize_role("boronic Acid") == "organoboron"
    assert normalize_role("ligand") == "ligand"


def test_extracts_dft_atom_variables_from_all_datasets(tmp_path: Path) -> None:
    output_path = tmp_path / "dft_atom_variables.yaml"

    result = extract_dft_atom_variables(output_path)

    assert output_path.exists()
    assert [dataset["dataset_key"] for dataset in result["datasets"]] == [
        "AZ",
        "SU_NO",
        "DY",
    ]
    assert dataset_by_key(result, "AZ")["row_count"] == 750
    assert dataset_by_key(result, "SU_NO")["row_count"] == 4543
    assert dataset_by_key(result, "DY")["row_count"] == 3955

    az_roles = roles_by_name(dataset_by_key(result, "AZ"))
    assert set(az_roles) == {"amine", "aryl_halide", "ligand"}
    assert atom_by_label(az_roles["amine"], "N1")["descriptor_columns"] == [
        "amine_.N1_NMR_shift",
        "amine_.N1_electrostatic_charge",
    ]

    su_roles = roles_by_name(dataset_by_key(result, "SU_NO"))
    assert "organoboron" in su_roles
    assert atom_by_label(su_roles["organoboron"], "B1")["descriptor_columns"] == [
        "boronic Acid_.B1_electrostatic_charge"
    ]

    dy_roles = roles_by_name(dataset_by_key(result, "DY"))
    assert {"additive", "aryl_halide", "base", "ligand"} <= set(dy_roles)
    assert atom_by_label(dy_roles["ligand"], "P1")["descriptor_columns"] == [
        "ligand_.P1_electrostatic_charge"
    ]

    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert loaded == result


def dataset_by_key(result: dict, dataset_key: str) -> dict:
    return next(
        dataset for dataset in result["datasets"] if dataset["dataset_key"] == dataset_key
    )


def roles_by_name(dataset: dict) -> dict:
    return {role["role"]: role for role in dataset["roles"]}


def atom_by_label(role: dict, atom_label: str) -> dict:
    return next(atom for atom in role["atoms"] if atom["atom_label"] == atom_label)
