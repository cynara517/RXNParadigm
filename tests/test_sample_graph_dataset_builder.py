from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_graph_dataset_builder import build_sample_graph_datasets  # noqa: E402


def test_builds_per_row_sample_graph_dataset(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    output_dir = tmp_path / "samples"
    source_csv = tmp_path / "az_source.csv"
    graph_dir.mkdir()
    write_graph_csvs(graph_dir)
    write_csv(
        source_csv,
        [
            {
                "id": "rxn_1",
                "yield": "10.5",
                "halide_.C1_NMR_shift": "111.1",
                "halide_.C1_electrostatic_charge": "-0.2",
                "amine_.N1_electrostatic_charge": "-0.8",
            },
            {
                "id": "rxn_2",
                "yield": "20.0",
                "halide_.C1_NMR_shift": "",
                "halide_.C1_electrostatic_charge": "-0.1",
                "amine_.N1_electrostatic_charge": "-0.7",
            },
        ],
    )

    result = build_sample_graph_datasets(
        graph_csv_dir=graph_dir,
        output_dir=output_dir,
        dataset_sources={"AZ": source_csv},
        dataset_keys=["AZ"],
        split_seed=1,
    )

    assert result["datasets"][0]["sample_count"] == 2
    assert result["datasets"][0]["node_count"] == 2
    assert result["datasets"][0]["directed_edge_count"] == 2
    assert result["datasets"][0]["missing_source_column_count"] == 1

    payload = np.load(output_dir / "AZ_samples.npz", allow_pickle=True)
    assert payload["node_feature_names"].tolist() == [
        "NMR_shift",
        "electrostatic_charge",
    ]
    assert payload["node_features"].shape == (2, 2, 2)
    assert payload["node_feature_mask"].shape == (2, 2, 2)
    assert payload["node_features"][0, 0, 0] == np.float32(111.1)
    assert payload["node_features"][0, 0, 1] == np.float32(-0.2)
    assert payload["node_feature_mask"][1, 0, 0] == 0.0
    assert payload["node_features"][0, 1, 1] == np.float32(-0.8)
    assert payload["y"].tolist() == [10.5, 20.0]
    assert payload["edge_index"].tolist() == [[0, 1], [1, 0]]
    assert payload["edge_attr"].shape[0] == 2
    assert payload["adjacency"].tolist() == [[0.0, 0.5], [0.5, 0.0]]
    assert payload["missing_source_columns"].tolist() == ["amine_.N1_NMR_shift"]


def test_split_is_deterministic(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    source_csv = tmp_path / "az_source.csv"
    graph_dir.mkdir()
    write_graph_csvs(graph_dir)
    write_csv(
        source_csv,
        [
            {
                "id": f"rxn_{idx}",
                "yield": str(idx),
                "halide_.C1_NMR_shift": str(idx + 1),
                "halide_.C1_electrostatic_charge": str(idx + 2),
                "amine_.N1_electrostatic_charge": str(idx + 3),
                "amine_.N1_NMR_shift": str(idx + 4),
            }
            for idx in range(20)
        ],
    )

    build_sample_graph_datasets(
        graph_csv_dir=graph_dir,
        output_dir=first_output,
        dataset_sources={"AZ": source_csv},
        dataset_keys=["AZ"],
        split_seed=27407,
    )
    build_sample_graph_datasets(
        graph_csv_dir=graph_dir,
        output_dir=second_output,
        dataset_sources={"AZ": source_csv},
        dataset_keys=["AZ"],
        split_seed=27407,
    )

    first = np.load(first_output / "AZ_samples.npz", allow_pickle=True)
    second = np.load(second_output / "AZ_samples.npz", allow_pickle=True)
    assert first["splits"].tolist() == second["splits"].tolist()
    assert first["splits"].tolist().count("train") == 16
    assert first["splits"].tolist().count("val") == 2
    assert first["splits"].tolist().count("test") == 2


def write_graph_csvs(path: Path) -> None:
    write_csv(
        path / "graph_manifest.csv",
        [
            {
                "dataset_key": "AZ",
                "nodes_csv": "AZ_nodes.csv",
                "edges_csv": "AZ_edges.csv",
                "adjacency_csv": "AZ_adjacency.csv",
                "node_edge_features_csv": "AZ_node_edge_feature_names.csv",
                "node_count": "2",
                "positive_edge_count": "1",
            }
        ],
    )
    write_csv(
        path / "AZ_nodes.csv",
        [
            {
                "node_index": "0",
                "node_id": "az:aryl_halide:C1",
                "dataset_key": "AZ",
                "role": "aryl_halide",
                "dft_atom_label": "C1",
                "element": "C",
                "node_feature_full_names": "halide_.C1_NMR_shift;halide_.C1_electrostatic_charge",
                "descriptor_names": "NMR_shift;electrostatic_charge",
                "likely_site_types": "ARYL_C_IPSO",
            },
            {
                "node_index": "1",
                "node_id": "az:amine:N1",
                "dataset_key": "AZ",
                "role": "amine",
                "dft_atom_label": "N1",
                "element": "N",
                "node_feature_full_names": "amine_.N1_electrostatic_charge;amine_.N1_NMR_shift",
                "descriptor_names": "electrostatic_charge;NMR_shift",
                "likely_site_types": "AMINE_N",
            },
        ],
    )
    write_csv(
        path / "AZ_edges.csv",
        [
            {
                "edge_index": "0",
                "source_node_id": "az:aryl_halide:C1",
                "target_node_id": "az:amine:N1",
                "edge_scope": "cross_role",
                "role_or_relation": "FORMING_BOND",
                "edge_feature_name": "cross_role_forming_bond",
                "edge_feature_full_name": "CROSS_ROLE_FORMING_BOND",
                "relation_type": "FORMING_BOND",
                "bond_type": "",
                "weight": "0.5",
                "evidence_sources": "paper_1",
            }
        ],
    )
    write_csv(
        path / "edge_feature_schema.csv",
        [
            {
                "edge_feature_name": "no_edge",
                "edge_feature_full_name": "NO_EDGE",
                "weight": "0.0",
                "description": "No edge.",
            },
            {
                "edge_feature_name": "cross_role_forming_bond",
                "edge_feature_full_name": "CROSS_ROLE_FORMING_BOND",
                "weight": "0.5",
                "description": "Forming bond.",
            },
        ],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
