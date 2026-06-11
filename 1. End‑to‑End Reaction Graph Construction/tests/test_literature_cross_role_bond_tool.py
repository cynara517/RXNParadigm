from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from literature_cross_role_bond_tool import (  # noqa: E402
    build_literature_bond_prompt,
    build_literature_cross_role_bonds,
    parse_llm_bond_response,
)


def minimal_fixed_graphs() -> dict:
    return {
        "datasets": [
            {
                "dataset_key": "AZ",
                "roles": [
                    {
                        "role": "aryl_halide",
                        "fixed_atoms": [
                            fixed_atom("az:aryl_halide:C1", "aryl_halide", "C1", "C")
                        ],
                    },
                    {
                        "role": "amine",
                        "fixed_atoms": [
                            fixed_atom("az:amine:N1", "amine", "N1", "N")
                        ],
                    },
                ],
            }
        ]
    }


def fixed_atom(
    fixed_atom_id: str,
    role: str,
    atom_label: str,
    element: str,
) -> dict:
    return {
        "fixed_atom_id": fixed_atom_id,
        "role": role,
        "dft_atom_label": atom_label,
        "element": element,
        "descriptor_columns": [f"{role}_.{atom_label}_NMR_shift"],
        "likely_site_types": [],
        "molecode_anchor": {"anchor_method": "element_and_index_match"},
    }


def test_prompt_lists_allowed_atoms() -> None:
    prompt = build_literature_bond_prompt(
        fixed_graphs=minimal_fixed_graphs(),
        reaction_types={"AZ": "Buchwald-Hartwig amination"},
        evidence_texts=[{"source_id": "paper_1", "text": "C-N bond formation."}],
    )

    assert "Buchwald-Hartwig amination" in prompt
    assert "az:aryl_halide:C1" in prompt
    assert "paper_1" in prompt
    assert "Return only JSON" in prompt


def test_parse_llm_json_response_from_code_fence() -> None:
    response = """```json
{"bonds": [{"dataset_key": "AZ", "source_role": "aryl_halide"}]}
```"""

    assert parse_llm_bond_response(response) == [
        {"dataset_key": "AZ", "source_role": "aryl_halide"}
    ]


def test_builds_default_cross_role_bond(tmp_path: Path) -> None:
    input_path = tmp_path / "dft_fixed_role_graphs.yaml"
    output_path = tmp_path / "literature_cross_role_bonds.yaml"
    input_path.write_text(
        yaml.safe_dump(minimal_fixed_graphs(), sort_keys=False),
        encoding="utf-8",
    )

    result = build_literature_cross_role_bonds(
        input_path=input_path,
        output_path=output_path,
        reaction_types={"AZ": "Buchwald-Hartwig amination"},
        retrieve_evidence=False,
    )

    bond = result["datasets"][0]["cross_role_bonds"][0]
    assert bond["source_fixed_atom_id"] == "az:aryl_halide:C1"
    assert bond["target_fixed_atom_id"] == "az:amine:N1"
    assert bond["relation_type"] == "FORMING_BOND"
    assert bond["validation_status"] == "grounded"
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result


def test_llm_response_is_validated_and_grounded(tmp_path: Path) -> None:
    input_path = tmp_path / "dft_fixed_role_graphs.yaml"
    output_path = tmp_path / "literature_cross_role_bonds.yaml"
    input_path.write_text(
        yaml.safe_dump(minimal_fixed_graphs(), sort_keys=False),
        encoding="utf-8",
    )
    llm_response = {
        "bonds": [
            {
                "dataset_key": "AZ",
                "source_role": "aryl_halide",
                "source_atom_label": "C1",
                "target_role": "amine",
                "target_atom_label": "N1",
                "relation_type": "FORMING_BOND",
                "mechanism_step": "reductive_elimination",
                "evidence_summary": "Aryl amination forms C-N bonds.",
                "evidence_sources": ["paper_1"],
            }
        ]
    }

    result = build_literature_cross_role_bonds(
        input_path=input_path,
        output_path=output_path,
        reaction_types={"AZ": "Buchwald-Hartwig amination"},
        llm_response_text=__import__("json").dumps(llm_response),
        retrieve_evidence=False,
    )

    bond = result["datasets"][0]["cross_role_bonds"][0]
    assert bond["created_by"] == "llm"
    assert bond["evidence_sources"] == ["paper_1"]


def test_invalid_llm_atom_is_reported(tmp_path: Path) -> None:
    input_path = tmp_path / "dft_fixed_role_graphs.yaml"
    output_path = tmp_path / "literature_cross_role_bonds.yaml"
    input_path.write_text(
        yaml.safe_dump(minimal_fixed_graphs(), sort_keys=False),
        encoding="utf-8",
    )
    llm_response = {
        "bonds": [
            {
                "dataset_key": "AZ",
                "source_role": "aryl_halide",
                "source_atom_label": "C99",
                "target_role": "amine",
                "target_atom_label": "N1",
                "relation_type": "FORMING_BOND",
            }
        ]
    }

    result = build_literature_cross_role_bonds(
        input_path=input_path,
        output_path=output_path,
        reaction_types={"AZ": "Buchwald-Hartwig amination"},
        llm_response_text=__import__("json").dumps(llm_response),
        retrieve_evidence=False,
    )

    assert result["datasets"][0]["cross_role_bonds"] == []
    assert result["datasets"][0]["validation_errors"][0]["error_type"] == (
        "unknown_source_atom"
    )


def test_retrieved_evidence_sources_are_attached(tmp_path: Path) -> None:
    input_path = tmp_path / "dft_fixed_role_graphs.yaml"
    output_path = tmp_path / "literature_cross_role_bonds.yaml"
    input_path.write_text(
        yaml.safe_dump(minimal_fixed_graphs(), sort_keys=False),
        encoding="utf-8",
    )

    result = build_literature_cross_role_bonds(
        input_path=input_path,
        output_path=output_path,
        reaction_types={"AZ": "Buchwald-Hartwig amination"},
        retrieve_evidence=True,
    )

    bond = result["datasets"][0]["cross_role_bonds"][0]
    assert "buchwald_hartwig_acs_chemrev_2025" in bond["evidence_sources"]
    assert "buchwald_hartwig_libretexts" in bond["evidence_sources"]
