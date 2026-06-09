from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "generic_smiles_extraction.yaml"


GENERIC_SYMBOLS = {
    "aryl_halide": "Ar-X",
    "amine": "R-NH2_or_R2NH",
    "organoboron": "Ar-B(OH)2_or_Ar-B(OR)2",
    "ligand": "L",
    "base": "Base",
    "solvent": "Solvent",
    "additive": "Additive",
    "counterion": "Counterion",
    "unknown_reactant": "UnknownReactant",
    "ligand_fragment": "L_fragment",
    "product": "Product",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    source_path: Path
    file_type: str
    reaction_family: str
    generic_reaction_template: str
    extractor: Callable[[pd.Series], Dict[str, List[str]]]


def extract_all(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    specs = [
        DatasetSpec(
            dataset_key="AZ",
            source_path=DATA_DIR / "az_no_rdkit.csv",
            file_type="csv",
            reaction_family="Buchwald-Hartwig amination",
            generic_reaction_template="Ar-X.R-NH2_or_R2NH.L.Base.Solvent>>Ar-NR",
            extractor=extract_az_components,
        ),
        DatasetSpec(
            dataset_key="SU_NO",
            source_path=DATA_DIR / "su_no_rdkit.csv",
            file_type="csv",
            reaction_family="Suzuki-Miyaura cross-coupling",
            generic_reaction_template="Ar-X.Ar-B(OH)2_or_Ar-B(OR)2.L.Base.Solvent>>Ar-Ar",
            extractor=extract_su_components,
        ),
        DatasetSpec(
            dataset_key="DY",
            source_path=DATA_DIR / "dy.xlsx",
            file_type="xlsx",
            reaction_family="Buchwald-Hartwig amination",
            generic_reaction_template="Ar-X.L.Additive.Base>>C-N_coupling_product",
            extractor=extract_dy_components,
        ),
    ]
    result = {
        "artifact_id": "generic_smiles_extraction_v1",
        "description": (
            "Concrete component SMILES and role-level generic symbols extracted "
            "from full datasets, with the first three reactions retained as examples."
        ),
        "datasets": [extract_dataset(spec) for spec in specs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def extract_dataset(spec: DatasetSpec) -> dict[str, Any]:
    dataframe = read_dataset(spec)
    role_values: dict[str, Counter[str]] = defaultdict(Counter)
    role_examples: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    first_three_examples: list[dict[str, Any]] = []

    for row_position, (_, row) in enumerate(dataframe.iterrows()):
        components = spec.extractor(row)
        normalized_components = {
            role: sorted(set(values))
            for role, values in components.items()
            if values
        }

        for role, values in normalized_components.items():
            for smiles in values:
                role_values[role][smiles] += 1
                if len(role_examples[role][smiles]) < 5:
                    role_examples[role][smiles].append(row_position)

        if row_position < 3:
            first_three_examples.append(
                {
                    "dataset_row_index": row_position,
                    "source_id": clean_value(row.get("id")),
                    "concrete_components": normalized_components,
                    "generic_components": {
                        role: GENERIC_SYMBOLS.get(role, role)
                        for role in normalized_components
                    },
                    "generic_reaction_template": spec.generic_reaction_template,
                }
            )

    return {
        "dataset_key": spec.dataset_key,
        "source_path": str(spec.source_path.relative_to(ROOT)),
        "file_type": spec.file_type,
        "reaction_family": spec.reaction_family,
        "row_count": int(len(dataframe)),
        "generic_reaction_template": spec.generic_reaction_template,
        "component_roles": build_component_roles(role_values, role_examples),
        "first_three_reaction_examples": first_three_examples,
    }


def read_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if spec.file_type == "csv":
        return pd.read_csv(spec.source_path)
    if spec.file_type == "xlsx":
        return pd.read_excel(spec.source_path)
    raise ValueError(f"Unsupported dataset type: {spec.file_type}")


def build_component_roles(
    role_values: dict[str, Counter[str]],
    role_examples: dict[str, dict[str, list[int]]],
) -> list[dict[str, Any]]:
    roles = []
    for role in sorted(role_values):
        values = role_values[role]
        roles.append(
            {
                "role": role,
                "generic_symbol": GENERIC_SYMBOLS.get(role, role),
                "unique_smiles_count": len(values),
                "components": [
                    {
                        "smiles": smiles,
                        "count": count,
                        "example_row_indices": role_examples[role][smiles],
                    }
                    for smiles, count in sorted(values.items())
                ],
            }
        )
    return roles


def extract_az_components(row: pd.Series) -> Dict[str, List[str]]:
    components: dict[str, list[str]] = defaultdict(list)
    for smiles in split_reactant_components(row.get("reactant_smiles")):
        components[classify_az_reactant(smiles)].append(smiles)
    add_single_value(components, "base", row.get("base_smiles"))
    add_single_value(components, "solvent", row.get("solvent_smiles"))
    add_single_value(components, "product", row.get("product_smiles"))
    return dict(components)


def extract_su_components(row: pd.Series) -> Dict[str, List[str]]:
    components: dict[str, list[str]] = defaultdict(list)
    for smiles in split_reactant_components(row.get("reactant_smiles")):
        components[classify_su_reactant(smiles)].append(smiles)
    add_single_value(components, "base", row.get("base_smiles"))
    add_single_value(components, "solvent", row.get("solvent_smiles"))
    add_single_value(components, "product", row.get("product_smiles"))
    return dict(components)


def extract_dy_components(row: pd.Series) -> Dict[str, List[str]]:
    components: dict[str, list[str]] = defaultdict(list)
    add_single_value(components, "ligand", row.get("Ligand"))
    add_single_value(components, "additive", row.get("Additive"))
    add_single_value(components, "base", row.get("Base"))
    add_single_value(components, "aryl_halide", row.get("Aryl halide"))
    return dict(components)


def classify_az_reactant(smiles: str) -> str:
    if is_counterion(smiles):
        return "counterion"
    if contains_phosphorus(smiles):
        return "ligand"
    if smiles == "[Fe]":
        return "ligand_fragment"
    if smiles == "C1CCCC1":
        return "ligand_fragment"
    if is_small_acid_or_additive(smiles):
        return "additive"
    if contains_aryl_leaving_group(smiles):
        return "aryl_halide"
    if "N" in smiles or "n" in smiles:
        return "amine"
    return "unknown_reactant"


def classify_su_reactant(smiles: str) -> str:
    if is_counterion(smiles):
        return "counterion"
    if contains_phosphorus(smiles):
        return "ligand"
    if contains_boron(smiles):
        return "organoboron"
    if contains_aryl_leaving_group(smiles):
        return "aryl_halide"
    return "unknown_reactant"


def contains_phosphorus(smiles: str) -> bool:
    return "P" in smiles


def contains_boron(smiles: str) -> bool:
    return "B" in smiles


def contains_aryl_leaving_group(smiles: str) -> bool:
    return (
        "Cl" in smiles
        or "Br" in smiles
        or "I" in smiles
        or "S(=O)" in smiles
        or "S(=O)(=O)" in smiles
    )


def is_counterion(smiles: str) -> bool:
    return smiles in {"[K+]", "[Na+]", "[Cs+]", "[Li+]"}


def is_small_acid_or_additive(smiles: str) -> bool:
    return smiles in {"C(=O)(C(F)(F)F)O"}


def split_reactant_components(value: Any) -> list[str]:
    text = clean_value(value)
    if text is None:
        return []
    return [part.strip() for part in text.split(".") if part.strip()]


def add_single_value(
    components: dict[str, list[str]], role: str, value: Any
) -> None:
    text = clean_value(value)
    if text is not None:
        components[role].append(text)


def clean_value(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def main() -> None:
    extract_all()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
