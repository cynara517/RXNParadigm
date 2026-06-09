from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "generated" / "generic_smiles_extraction.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "role_smiles_summary.yaml"


def summarize_role_smiles(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    source = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    summary = {
        "artifact_id": "role_smiles_summary_v1",
        "datasets": [
            {
                "dataset_key": dataset["dataset_key"],
                "roles": [
                    {
                        "role": role["role"],
                        "smiles": [
                            component["smiles"]
                            for component in role.get("components", [])
                        ],
                    }
                    for role in dataset.get("component_roles", [])
                ],
            }
            for dataset in source.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summarize_role_smiles()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
