from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dft_fixed_role_graph_builder import (  # noqa: E402
    build_dft_fixed_role_graphs,
    select_molecode_anchor,
)


def test_selects_exact_element_index_anchor() -> None:
    anchor = select_molecode_anchor(
        {
            "element": "N",
            "atom_index": 1,
            "candidate_molecode_nodes": [
                {
                    "node_id": "azaminem0000_N_1",
                    "element": "N",
                    "node_index": 1,
                }
            ],
        }
    )

    assert anchor["anchor_status"] == "fixed_to_molecode"
    assert anchor["anchor_method"] == "element_and_index_match"
    assert anchor["fixed_molecode_node"]["node_id"] == "azaminem0000_N_1"


def test_builds_dft_fixed_role_graph(tmp_path: Path) -> None:
    canonical_path = tmp_path / "role_atom_canonicalization.yaml"
    graph_path = tmp_path / "role_canonical_graphs.yaml"
    output_path = tmp_path / "dft_fixed_role_graphs.yaml"
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
                                "canonical_role_atoms": [
                                    {
                                        "canonical_atom_id": "az:amine:N1",
                                        "role": "amine",
                                        "dft_atom_label": "N1",
                                        "element": "N",
                                        "atom_index": 1,
                                        "descriptor_columns": [
                                            "amine_.N1_NMR_shift",
                                            "amine_.N1_electrostatic_charge",
                                        ],
                                        "descriptor_names": [
                                            "NMR_shift",
                                            "electrostatic_charge",
                                        ],
                                        "likely_site_types": ["AMINE_N"],
                                        "candidate_molecode_nodes": [
                                            {
                                                "node_id": "azaminem0000_N_1",
                                                "component_id": "az:amine:m0000",
                                                "element": "N",
                                                "node_index": 1,
                                            }
                                        ],
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
    graph_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "graph": {
                                    "edges": [
                                        {
                                            "edge_id": "edge:az_amine_C1--az_amine_N1",
                                            "source_canonical_atom_id": "az:amine:C1",
                                            "target_canonical_atom_id": "az:amine:N1",
                                            "evidence_molecode_node_pairs": [
                                                {
                                                    "source_node_id": "azaminem0000_C_1",
                                                    "target_node_id": "azaminem0000_N_1",
                                                    "source_component_id": "az:amine:m0000",
                                                    "target_component_id": "az:amine:m0000",
                                                    "operator": "---",
                                                    "bond_type": "SINGLE",
                                                }
                                            ],
                                        }
                                    ]
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

    result = build_dft_fixed_role_graphs(canonical_path, graph_path, output_path)

    role = result["datasets"][0]["roles"][0]
    assert role["graph_identity_policy"] == "dft_atom_label_is_primary_atom_id"
    assert role["fixed_atoms"][0]["fixed_atom_id"] == "az:amine:N1"
    assert role["descriptor_to_atom"] == {
        "amine_.N1_NMR_shift": "az:amine:N1",
        "amine_.N1_electrostatic_charge": "az:amine:N1",
    }
    assert role["fixed_edges"] == []
    assert role["coverage"]["bond_type_summary"] == {}
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result


def test_builds_fixed_bond_records_when_both_anchors_are_selected(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "role_atom_canonicalization.yaml"
    graph_path = tmp_path / "role_canonical_graphs.yaml"
    output_path = tmp_path / "dft_fixed_role_graphs.yaml"
    canonical_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "canonical_role_atoms": [
                                    {
                                        "canonical_atom_id": "az:amine:C1",
                                        "role": "amine",
                                        "dft_atom_label": "C1",
                                        "element": "C",
                                        "atom_index": 1,
                                        "descriptor_columns": ["amine_.C1_NMR_shift"],
                                        "descriptor_names": ["NMR_shift"],
                                        "likely_site_types": [],
                                        "candidate_molecode_nodes": [
                                            {
                                                "node_id": "azaminem0000_C_1",
                                                "component_id": "az:amine:m0000",
                                                "element": "C",
                                                "node_index": 1,
                                            }
                                        ],
                                    },
                                    {
                                        "canonical_atom_id": "az:amine:N1",
                                        "role": "amine",
                                        "dft_atom_label": "N1",
                                        "element": "N",
                                        "atom_index": 1,
                                        "descriptor_columns": ["amine_.N1_NMR_shift"],
                                        "descriptor_names": ["NMR_shift"],
                                        "likely_site_types": ["AMINE_N"],
                                        "candidate_molecode_nodes": [
                                            {
                                                "node_id": "azaminem0000_N_1",
                                                "component_id": "az:amine:m0000",
                                                "element": "N",
                                                "node_index": 1,
                                            }
                                        ],
                                    },
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
    graph_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "graph": {
                                    "edges": [
                                        {
                                            "source_canonical_atom_id": "az:amine:C1",
                                            "target_canonical_atom_id": "az:amine:N1",
                                            "evidence_molecode_node_pairs": [
                                                {
                                                    "source_node_id": "azaminem0000_C_1",
                                                    "target_node_id": "azaminem0000_N_1",
                                                    "source_component_id": "az:amine:m0000",
                                                    "target_component_id": "az:amine:m0000",
                                                    "operator": "---",
                                                    "bond_type": "SINGLE",
                                                }
                                            ],
                                        }
                                    ]
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

    result = build_dft_fixed_role_graphs(canonical_path, graph_path, output_path)

    edge = result["datasets"][0]["roles"][0]["fixed_edges"][0]
    assert edge["source_fixed_atom_id"] == "az:amine:C1"
    assert edge["target_fixed_atom_id"] == "az:amine:N1"
    assert edge["edge_category"] == "COVALENT_BOND"
    assert edge["bond_types"] == ["SINGLE"]
    assert edge["bond_orders"] == [1.0]
    assert edge["bond_records"][0]["molecode_operator"] == "---"
    assert edge["bond_records"][0]["bond_order"] == 1.0
    assert result["datasets"][0]["roles"][0]["coverage"]["bond_type_summary"] == {
        "SINGLE": 1
    }
