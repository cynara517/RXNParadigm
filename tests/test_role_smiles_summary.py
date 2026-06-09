from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from role_smiles_summary import summarize_role_smiles  # noqa: E402


def test_summarizes_only_roles_and_smiles(tmp_path: Path) -> None:
    input_path = tmp_path / "generic_smiles_extraction.yaml"
    output_path = tmp_path / "role_smiles_summary.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "artifact_id": "generic_smiles_extraction_v1",
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "component_roles": [
                            {
                                "role": "amine",
                                "generic_symbol": "R-NH2_or_R2NH",
                                "unique_smiles_count": 2,
                                "components": [
                                    {"smiles": "CN", "count": 3},
                                    {"smiles": "CCN", "count": 1},
                                ],
                            }
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = summarize_role_smiles(input_path, output_path)

    assert summary == {
        "artifact_id": "role_smiles_summary_v1",
        "datasets": [
            {
                "dataset_key": "AZ",
                "roles": [
                    {
                        "role": "amine",
                        "smiles": ["CN", "CCN"],
                    }
                ],
            }
        ],
    }
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == summary
