from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generic_smiles_extractor import extract_all, extract_dataset  # noqa: E402
from generic_smiles_extractor import (  # noqa: E402
    DatasetSpec,
    DATA_DIR,
    extract_az_components,
    extract_dy_components,
    extract_su_components,
)


def test_extract_all_reads_full_datasets(tmp_path: Path) -> None:
    output_path = tmp_path / "generic_smiles_extraction.yaml"

    result = extract_all(output_path)

    assert output_path.exists()
    assert [dataset["dataset_key"] for dataset in result["datasets"]] == [
        "AZ",
        "SU_NO",
        "DY",
    ]
    assert dataset_by_key(result, "AZ")["row_count"] == 750
    assert dataset_by_key(result, "SU_NO")["row_count"] == 4543
    assert dataset_by_key(result, "DY")["row_count"] == 3955


def test_output_yaml_is_serializable(tmp_path: Path) -> None:
    output_path = tmp_path / "generic_smiles_extraction.yaml"

    extract_all(output_path)
    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert loaded["artifact_id"] == "generic_smiles_extraction_v1"
    assert dataset_by_key(loaded, "AZ")["first_three_reaction_examples"]


def test_az_roles_are_extracted_from_first_rows() -> None:
    dataset = extract_dataset(
        DatasetSpec(
            dataset_key="AZ",
            source_path=DATA_DIR / "az_no_rdkit.csv",
            file_type="csv",
            reaction_family="Buchwald-Hartwig amination",
            generic_reaction_template="Ar-X.R-NH2_or_R2NH.L.Base.Solvent>>Ar-NR",
            extractor=extract_az_components,
        )
    )

    roles = role_names(dataset)

    assert {"amine", "aryl_halide", "ligand", "base", "solvent", "product"} <= roles
    assert len(dataset["first_three_reaction_examples"]) == 3


def test_su_roles_are_extracted_from_first_rows() -> None:
    dataset = extract_dataset(
        DatasetSpec(
            dataset_key="SU_NO",
            source_path=DATA_DIR / "su_no_rdkit.csv",
            file_type="csv",
            reaction_family="Suzuki-Miyaura cross-coupling",
            generic_reaction_template="Ar-X.Ar-B(OH)2_or_Ar-B(OR)2.L.Base.Solvent>>Ar-Ar",
            extractor=extract_su_components,
        )
    )

    roles = role_names(dataset)

    assert {"aryl_halide", "organoboron", "ligand", "base", "solvent"} <= roles
    assert len(dataset["first_three_reaction_examples"]) == 3


def test_dy_roles_are_extracted_from_first_rows() -> None:
    dataset = extract_dataset(
        DatasetSpec(
            dataset_key="DY",
            source_path=DATA_DIR / "dy.xlsx",
            file_type="xlsx",
            reaction_family="Buchwald-Hartwig amination",
            generic_reaction_template="Ar-X.L.Additive.Base>>C-N_coupling_product",
            extractor=extract_dy_components,
        )
    )

    roles = role_names(dataset)

    assert {"aryl_halide", "ligand", "additive", "base"} <= roles
    assert len(dataset["first_three_reaction_examples"]) == 3


def dataset_by_key(result: dict, dataset_key: str) -> dict:
    return next(
        dataset for dataset in result["datasets"] if dataset["dataset_key"] == dataset_key
    )


def role_names(dataset: dict) -> set[str]:
    return {role["role"] for role in dataset["component_roles"]}
