from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from role_graph_dft_aligner import (  # noqa: E402
    align_role_graphs_to_dft_atoms,
    parse_molecode_nodes,
)


def test_parse_molecode_nodes() -> None:
    component = {
        "component_id": "az:amine:m0000",
        "molecode_graph": """graph TB
    subgraph azaminem0000["az:amine:m0000"]
        azaminem0000_C_1[CH3]
        azaminem0000_N_1[NH2]
        azaminem0000_C_1 --- azaminem0000_N_1
    end""",
    }

    assert parse_molecode_nodes(component) == [
        {
            "node_id": "azaminem0000_C_1",
            "component_id": "az:amine:m0000",
            "element": "C",
            "node_index": 1,
            "display_label": "CH3",
        },
        {
            "node_id": "azaminem0000_N_1",
            "component_id": "az:amine:m0000",
            "element": "N",
            "node_index": 1,
            "display_label": "NH2",
        },
    ]


def test_aligns_dft_atoms_to_representative_molecode_nodes(tmp_path: Path) -> None:
    molecode_path = tmp_path / "role_molecode_summary.yaml"
    dft_path = tmp_path / "dft_atom_variables.yaml"
    ontology_path = tmp_path / "site_ontology.yaml"
    output_path = tmp_path / "role_graph_dft_alignment.yaml"

    molecode_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "molecode_components": [
                                    {
                                        "component_id": "az:amine:m0000",
                                        "source_smiles": "CN",
                                        "molecode_parser": {"status": "parsed"},
                                        "molecode_graph": """graph TB
    subgraph azaminem0000["az:amine:m0000"]
        azaminem0000_C_1[CH3]
        azaminem0000_N_1[NH2]
        azaminem0000_C_1 --- azaminem0000_N_1
    end""",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dft_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "atoms": [
                                    {
                                        "atom_label": "N1",
                                        "element": "N",
                                        "atom_index": 1,
                                        "descriptor_columns": ["amine_.N1_NMR_shift"],
                                        "descriptor_names": ["NMR_shift"],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ontology_path.write_text(
        """```yaml
site_types:
  AMINE_N:
    matching_strategy: SMARTS_OR_MOLECODE_PATTERN
    pattern_hint: "[N;H0,H1,H2,H3;!$(N=*)]"
```""",
        encoding="utf-8",
    )

    result = align_role_graphs_to_dft_atoms(
        molecode_path=molecode_path,
        dft_path=dft_path,
        ontology_path=ontology_path,
        output_path=output_path,
    )

    aligned_atom = result["datasets"][0]["roles"][0]["dft_atom_anchor_map"][0]
    assert aligned_atom["alignment_status"] == "exact_element_singleton"
    assert aligned_atom["confidence"] == "high"
    assert aligned_atom["candidate_molecode_nodes"][0]["node_id"] == "azaminem0000_N_1"
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result
