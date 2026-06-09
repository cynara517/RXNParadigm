from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from role_molecode_converter import convert_role_smiles_to_molecode  # noqa: E402


def test_converts_role_smiles_to_molecode_component_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "role_smiles_summary.yaml"
    output_path = tmp_path / "role_molecode_summary.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = convert_role_smiles_to_molecode(input_path, output_path)

    components = result["datasets"][0]["roles"][0]["molecode_components"]
    assert [component["component_id"] for component in components] == [
        "az:amine:m0000",
        "az:amine:m0001",
    ]
    assert [component["source_smiles"] for component in components] == ["CN", "CCN"]
    assert components[0]["molecode_parser"]["status"] == "parsed"
    assert components[0]["canonical_smiles"] == "CN"
    assert components[0]["roundtrip_smiles"] == "CN"
    assert components[0]["roundtrip_ok"] is True
    assert components[0]["molecode_graph"].startswith("graph TB")
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result


def test_invalid_smiles_is_retained_with_parse_error(tmp_path: Path) -> None:
    input_path = tmp_path / "role_smiles_summary.yaml"
    output_path = tmp_path / "role_molecode_summary.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "artifact_id": "role_smiles_summary_v1",
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "smiles": ["not_a_smiles"],
                            }
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = convert_role_smiles_to_molecode(input_path, output_path)

    component = result["datasets"][0]["roles"][0]["molecode_components"][0]
    assert component["molecode_parser"]["status"] == "failed"
    assert component["molecode_graph"] is None
    assert component["roundtrip_ok"] is False
    assert component["errors"][0]["error_type"] == "rdkit_parse_failure"
