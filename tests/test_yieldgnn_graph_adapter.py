from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yieldgnn_graph_adapter import (  # noqa: E402
    YIELDGNN_EDGE_FEATURE_DIM,
    YIELDGNN_NODE_FEATURE_DIM,
    export_csv_graphs_to_yieldgnn_npz,
    load_csv_graph_as_yieldgnn_record,
)


def test_loads_csv_graph_as_yieldgnn_record(tmp_path: Path) -> None:
    write_minimal_csv_graph(tmp_path)

    record = load_csv_graph_as_yieldgnn_record(tmp_path, "AZ")

    assert record.dataset_key == "AZ"
    assert record.n_node == 2
    assert record.n_edge == 2
    assert record.node_attr.shape == (2, YIELDGNN_NODE_FEATURE_DIM)
    assert record.edge_attr.shape == (2, YIELDGNN_EDGE_FEATURE_DIM)
    assert record.src.tolist() == [0, 1]
    assert record.dst.tolist() == [1, 0]
    assert record.node_attr[0, 6] == 1.0
    assert record.node_attr[1, 7] == 1.0
    assert record.edge_attr[0, 4] == 1.0
    assert record.edge_attr[0, 5] == 1.0
    assert record.edge_attr[0, 7] == 0.5


def test_exports_yieldgnn_npz_payload(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    output_dir = tmp_path / "yieldgnn"
    csv_dir.mkdir()
    write_minimal_csv_graph(csv_dir)

    result = export_csv_graphs_to_yieldgnn_npz(csv_dir, output_dir)

    assert result["dataset_count"] == 1
    payload = np.load(output_dir / "test_0.npz", allow_pickle=True)["data"]
    rmol_dict, pmol_dict, reaction_dict = payload
    assert rmol_dict[0]["n_node"].tolist() == [2]
    assert rmol_dict[0]["n_edge"].tolist() == [2]
    assert rmol_dict[0]["node_attr"].shape == (2, YIELDGNN_NODE_FEATURE_DIM)
    assert rmol_dict[0]["edge_attr"].shape == (2, YIELDGNN_EDGE_FEATURE_DIM)
    assert pmol_dict[0]["n_node"].tolist() == [1]
    assert pmol_dict[0]["n_edge"].tolist() == [0]
    assert reaction_dict["yld"].tolist() == [0.0]
    assert reaction_dict["rsmi"].tolist() == ["AZ>>DUMMY"]


def write_minimal_csv_graph(path: Path) -> None:
    write_csv(
        path / "graph_manifest.csv",
        [
            {
                "dataset_key": "AZ",
                "nodes_csv": "AZ_nodes.csv",
                "edges_csv": "AZ_edges.csv",
                "adjacency_csv": "AZ_adjacency.csv",
                "node_edge_features_csv": "AZ_node_edge_feature_names.csv",
                "node_count": 2,
                "positive_edge_count": 1,
            }
        ],
    )
    write_csv(
        path / "AZ_nodes.csv",
        [
            {
                "node_index": 0,
                "node_id": "az:aryl_halide:C1",
                "dataset_key": "AZ",
                "role": "aryl_halide",
                "dft_atom_label": "C1",
                "element": "C",
                "node_feature_full_names": "halide_.C1_NMR_shift",
                "descriptor_names": "NMR_shift",
                "likely_site_types": "ARYL_C_IPSO",
            },
            {
                "node_index": 1,
                "node_id": "az:amine:N1",
                "dataset_key": "AZ",
                "role": "amine",
                "dft_atom_label": "N1",
                "element": "N",
                "node_feature_full_names": "amine_.N1_electrostatic_charge",
                "descriptor_names": "electrostatic_charge",
                "likely_site_types": "AMINE_N",
            },
        ],
    )
    write_csv(
        path / "AZ_edges.csv",
        [
            {
                "edge_index": 0,
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
