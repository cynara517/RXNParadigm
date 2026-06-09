from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "dft_atom_variables.yaml"

ATOM_DESCRIPTOR_RE = re.compile(
    r"^(?P<raw_role>.+)_\.(?P<atom_label>[A-Z][a-z]?\d+)_(?P<descriptor_name>.+)$"
)
ATOM_LABEL_RE = re.compile(r"^(?P<element>[A-Z][a-z]?)(?P<atom_index>\d+)$")

ROLE_ALIASES = {
    "halide": "aryl_halide",
    "boronic Acid": "organoboron",
}


@dataclass(frozen=True)
class DFTDatasetSpec:
    dataset_key: str
    source_path: Path


def extract_dft_atom_variables(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    specs = [
        DFTDatasetSpec("AZ", DATA_DIR / "az_no_rdkit.csv"),
        DFTDatasetSpec("SU_NO", DATA_DIR / "su_no_rdkit.csv"),
        DFTDatasetSpec("DY", DATA_DIR / "dy_dft.csv"),
    ]
    result = {
        "artifact_id": "dft_atom_variables_v1",
        "description": (
            "Atom-level DFT descriptor variables grouped by dataset, role, and "
            "atom label. Descriptor values are intentionally not extracted here."
        ),
        "datasets": [extract_dataset_atom_variables(spec) for spec in specs],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def extract_dataset_atom_variables(spec: DFTDatasetSpec) -> dict[str, Any]:
    dataframe = pd.read_csv(spec.source_path)
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for column_name in dataframe.columns:
        parsed = parse_atom_descriptor_column(column_name)
        if parsed is None:
            continue
        role = normalize_role(parsed["raw_role"])
        grouped[role][parsed["atom_label"]].append(
            {
                "descriptor_name": parsed["descriptor_name"],
                "column_name": column_name,
            }
        )

    return {
        "dataset_key": spec.dataset_key,
        "source_path": str(spec.source_path.relative_to(ROOT)),
        "row_count": int(len(dataframe)),
        "roles": build_role_records(grouped),
    }


def parse_atom_descriptor_column(column_name: str) -> dict[str, str] | None:
    match = ATOM_DESCRIPTOR_RE.match(column_name)
    if not match:
        return None
    atom_label = match.group("atom_label")
    atom_match = ATOM_LABEL_RE.match(atom_label)
    if not atom_match:
        return None
    return {
        "raw_role": match.group("raw_role"),
        "atom_label": atom_label,
        "element": atom_match.group("element"),
        "atom_index": atom_match.group("atom_index"),
        "descriptor_name": match.group("descriptor_name"),
    }


def normalize_role(raw_role: str) -> str:
    return ROLE_ALIASES.get(raw_role, raw_role)


def build_role_records(
    grouped: dict[str, dict[str, list[dict[str, str]]]]
) -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "atoms": [
                build_atom_record(atom_label, descriptors)
                for atom_label, descriptors in sorted(
                    atoms.items(), key=lambda item: atom_sort_key(item[0])
                )
            ],
        }
        for role, atoms in sorted(grouped.items())
    ]


def build_atom_record(
    atom_label: str,
    descriptors: list[dict[str, str]],
) -> dict[str, Any]:
    atom_match = ATOM_LABEL_RE.match(atom_label)
    if atom_match is None:
        raise ValueError(f"Invalid atom label: {atom_label}")
    sorted_descriptors = sorted(
        descriptors,
        key=lambda descriptor: descriptor["descriptor_name"],
    )
    return {
        "atom_label": atom_label,
        "element": atom_match.group("element"),
        "atom_index": int(atom_match.group("atom_index")),
        "descriptor_columns": [
            descriptor["column_name"] for descriptor in sorted_descriptors
        ],
        "descriptor_names": [
            descriptor["descriptor_name"] for descriptor in sorted_descriptors
        ],
    }


def atom_sort_key(atom_label: str) -> tuple[str, int, str]:
    atom_match = ATOM_LABEL_RE.match(atom_label)
    if atom_match is None:
        return (atom_label, 0, atom_label)
    return (
        atom_match.group("element"),
        int(atom_match.group("atom_index")),
        atom_label,
    )


def main() -> None:
    extract_dft_atom_variables()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
