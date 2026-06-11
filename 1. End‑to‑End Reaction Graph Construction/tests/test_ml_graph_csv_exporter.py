from __future__ import annotations

import csv
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_graph_csv_exporter import export_ml_graph_csvs  # noqa: E402


def test_exports_clean_ml_csv_graphs(tmp_path: Path) -> None:
    fixed_path = tmp_path / "dft_fixed_role_graphs.yaml"
    cross_path = tmp_path / "literature_cross_role_bonds.yaml"
    output_dir = tmp_path / "csv"
    fixed_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "fixed_atoms": [
                                    atom("az:amine:C1", "amine", "C1", "C"),
                                    atom("az:amine:N1", "amine", "N1", "N"),
                                ],
                                "fixed_edges": [
                                    {
                                        "source_fixed_atom_id": "az:amine:C1",
                                        "target_fixed_atom_id": "az:amine:N1",
                                        "bond_types": ["SINGLE"],
                                    }
                                ],
                            },
                            {
                                "role": "aryl_halide",
                                "fixed_atoms": [
                                    atom("az:aryl_halide:C1", "aryl_halide", "C1", "C")
                                ],
                                "fixed_edges": [],
                            },
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cross_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "cross_role_bonds": [
                            {
                                "source_fixed_atom_id": "az:aryl_halide:C1",
                                "target_fixed_atom_id": "az:amine:N1",
                                "relation_type": "FORMING_BOND",
                                "evidence_sources": ["paper_1"],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = export_ml_graph_csvs(fixed_path, cross_path, output_dir)

    assert result["datasets"] == [
        {"dataset_key": "AZ", "node_count": 3, "positive_edge_count": 2}
    ]
    edges = read_csv(output_dir / "AZ_edges.csv")
    assert edges[0]["edge_feature_full_name"] == "INTRA_ROLE_COVALENT_SINGLE_BOND"
    assert edges[0]["weight"] == "0.33"
    assert edges[1]["edge_feature_full_name"] == "CROSS_ROLE_FORMING_BOND"
    assert edges[1]["weight"] == "0.5"

    adjacency = read_csv(output_dir / "AZ_adjacency.csv")
    row_by_node = {row["node_id"]: row for row in adjacency}
    assert row_by_node["az:amine:C1"]["az:amine:N1"] == "0.33"
    assert row_by_node["az:aryl_halide:C1"]["az:amine:N1"] == "0.5"
    assert row_by_node["az:amine:C1"]["az:aryl_halide:C1"] == "0.0"

    node_features = read_csv(output_dir / "AZ_node_edge_feature_names.csv")
    assert "CROSS_ROLE_FORMING_BOND" in node_features[1][
        "incident_edge_feature_full_names"
    ]
    assert "NO_EDGE" in node_features[1]["all_edge_feature_full_names"]


def atom(fixed_atom_id: str, role: str, label: str, element: str) -> dict:
    return {
        "fixed_atom_id": fixed_atom_id,
        "role": role,
        "dft_atom_label": label,
        "element": element,
        "descriptor_columns": [f"{role}_.{label}_NMR_shift"],
        "descriptor_names": ["NMR_shift"],
        "likely_site_types": [],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
