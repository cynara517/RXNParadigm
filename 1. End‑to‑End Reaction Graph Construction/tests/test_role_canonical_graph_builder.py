from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from role_canonical_graph_builder import (  # noqa: E402
    build_role_canonical_graphs,
    parse_molecode_edges,
)


def test_parse_molecode_edges() -> None:
    graph = """graph TB
    subgraph azaminem0000["az:amine:m0000"]
        azaminem0000_C_1[CH3]
        azaminem0000_N_1[NH2]
        azaminem0000_C_1 --- azaminem0000_N_1
    end"""

    assert parse_molecode_edges(graph) == [
        {
            "source_node_id": "azaminem0000_C_1",
            "target_node_id": "azaminem0000_N_1",
            "operator": "---",
            "bond_type": "SINGLE",
        }
    ]


def test_builds_canonical_role_graph_edges(tmp_path: Path) -> None:
    canonical_path = tmp_path / "role_atom_canonicalization.yaml"
    alignment_path = tmp_path / "role_graph_dft_alignment.yaml"
    output_path = tmp_path / "role_canonical_graphs.yaml"
    canonical_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "representative_component_id": "az:amine:m0000",
                                "representative_source_smiles": "CN",
                                "descriptor_coverage": {"canonical_atom_count": 2},
                                "mapping_status_summary": {
                                    "exact_element_singleton": 2
                                },
                                "canonical_role_atoms": [
                                    {
                                        "canonical_atom_id": "az:amine:C1",
                                        "dft_atom_label": "C1",
                                        "element": "C",
                                        "mapping_status": "exact_element_singleton",
                                        "confidence": "high",
                                        "candidate_molecode_nodes": [
                                            {
                                                "node_id": "azaminem0000_C_1",
                                                "component_id": "az:amine:m0000",
                                            }
                                        ],
                                    },
                                    {
                                        "canonical_atom_id": "az:amine:N1",
                                        "dft_atom_label": "N1",
                                        "element": "N",
                                        "mapping_status": "exact_element_singleton",
                                        "confidence": "high",
                                        "candidate_molecode_nodes": [
                                            {
                                                "node_id": "azaminem0000_N_1",
                                                "component_id": "az:amine:m0000",
                                            }
                                        ],
                                    },
                                ],
                                "canonical_role_graph": {
                                    "nodes": [
                                        {
                                            "canonical_atom_id": "az:amine:C1",
                                            "element": "C",
                                        },
                                        {
                                            "canonical_atom_id": "az:amine:N1",
                                            "element": "N",
                                        },
                                    ],
                                    "edges": [],
                                },
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    alignment_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "representative_molecode_graph": """graph TB
    subgraph azaminem0000["az:amine:m0000"]
        azaminem0000_C_1[CH3]
        azaminem0000_N_1[NH2]
        azaminem0000_C_1 --- azaminem0000_N_1
    end""",
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = build_role_canonical_graphs(
        canonical_path=canonical_path,
        alignment_path=alignment_path,
        output_path=output_path,
    )

    edge = result["datasets"][0]["roles"][0]["graph"]["edges"][0]
    assert edge["source_canonical_atom_id"] == "az:amine:C1"
    assert edge["target_canonical_atom_id"] == "az:amine:N1"
    assert edge["edge_status"] == "confirmed_representative_bond"
    assert edge["bond_types"] == ["SINGLE"]
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result
