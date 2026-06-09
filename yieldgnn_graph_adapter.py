from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = ROOT / "generated" / "ml_graph_csv"
DEFAULT_OUTPUT_DIR = ROOT / "generated" / "yieldgnn_graph_input"

YIELDGNN_NODE_FEATURE_DIM = 155
YIELDGNN_EDGE_FEATURE_DIM = 8

ATOMIC_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "Ni": 28,
    "Pd": 46,
    "Cu": 29,
    "Zn": 30,
}

ROLE_FEATURE_SLOTS = {
    "aryl_halide": 129,
    "amine": 130,
    "ligand": 131,
    "base": 132,
    "organoboron": 133,
    "boronic_acid": 133,
    "catalyst": 134,
    "solvent": 135,
    "additive": 136,
    "product": 137,
}


@dataclass(frozen=True)
class YieldGNNGraphRecord:
    dataset_key: str
    node_ids: list[str]
    n_node: int
    n_edge: int
    node_attr: np.ndarray
    edge_attr: np.ndarray
    src: np.ndarray
    dst: np.ndarray


def export_csv_graphs_to_yieldgnn_npz(
    csv_dir: Path = DEFAULT_CSV_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_keys: Iterable[str] | None = None,
    output_name: str = "test_0.npz",
) -> dict[str, Any]:
    records = [
        load_csv_graph_as_yieldgnn_record(csv_dir, dataset_key)
        for dataset_key in resolve_dataset_keys(csv_dir, dataset_keys)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    write_yieldgnn_npz(records, output_path)
    manifest_rows = [
        {
            "dataset_key": record.dataset_key,
            "n_node": record.n_node,
            "n_directed_edge": record.n_edge,
            "node_feature_dim": record.node_attr.shape[1],
            "edge_feature_dim": record.edge_attr.shape[1],
            "role_graph_position": 0,
            "product_graph": "dummy",
        }
        for record in records
    ]
    write_csv(output_dir / "yieldgnn_adapter_manifest.csv", manifest_rows)
    return {
        "artifact_id": "yieldgnn_csv_adapter_v1",
        "output_npz": str(output_path),
        "dataset_count": len(records),
        "node_feature_dim": YIELDGNN_NODE_FEATURE_DIM,
        "edge_feature_dim": YIELDGNN_EDGE_FEATURE_DIM,
        "datasets": manifest_rows,
    }


def resolve_dataset_keys(
    csv_dir: Path,
    dataset_keys: Iterable[str] | None,
) -> list[str]:
    if dataset_keys is not None:
        return list(dataset_keys)
    manifest_path = csv_dir / "graph_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing graph manifest: {manifest_path}")
    return [row["dataset_key"] for row in read_csv(manifest_path)]


def load_csv_graph_as_yieldgnn_record(
    csv_dir: Path,
    dataset_key: str,
) -> YieldGNNGraphRecord:
    nodes_path = csv_dir / f"{dataset_key}_nodes.csv"
    edges_path = csv_dir / f"{dataset_key}_edges.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes CSV: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges CSV: {edges_path}")

    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    node_ids = [row["node_id"] for row in nodes]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    node_attr = np.vstack([node_to_attr(row) for row in nodes]).astype(np.float32)

    src: list[int] = []
    dst: list[int] = []
    edge_attr: list[np.ndarray] = []
    for edge in edges:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        if source not in node_index or target not in node_index:
            continue
        attr = edge_to_attr(edge)
        source_idx = node_index[source]
        target_idx = node_index[target]
        src.extend([source_idx, target_idx])
        dst.extend([target_idx, source_idx])
        edge_attr.extend([attr, attr.copy()])

    if edge_attr:
        edge_attr_array = np.vstack(edge_attr).astype(np.float32)
    else:
        edge_attr_array = np.empty((0, YIELDGNN_EDGE_FEATURE_DIM), dtype=np.float32)

    return YieldGNNGraphRecord(
        dataset_key=dataset_key,
        node_ids=node_ids,
        n_node=len(node_ids),
        n_edge=len(src),
        node_attr=node_attr,
        edge_attr=edge_attr_array,
        src=np.array(src, dtype=int),
        dst=np.array(dst, dtype=int),
    )


def node_to_attr(node: dict[str, str]) -> np.ndarray:
    attr = np.zeros(YIELDGNN_NODE_FEATURE_DIM, dtype=np.float32)
    atomic_number = ATOMIC_NUMBERS.get(node.get("element", ""))
    if atomic_number is not None:
        attr[min(atomic_number, 117)] = 1.0

    descriptor_names = split_semicolon(node.get("descriptor_names", ""))
    likely_site_types = split_semicolon(node.get("likely_site_types", ""))
    role = node.get("role", "")

    attr[118] = min(len(descriptor_names) / 10.0, 1.0)
    attr[119] = float("NMR_shift" in descriptor_names)
    attr[120] = float("electrostatic_charge" in descriptor_names)
    attr[121] = float(bool(likely_site_types))
    attr[122] = float(node.get("element") == "C")
    attr[123] = float(node.get("element") == "N")
    attr[124] = float(node.get("element") == "O")
    attr[125] = float(node.get("element") in {"F", "Cl", "Br", "I"})
    attr[126] = float(node.get("element") == "P")
    attr[127] = float(node.get("element") == "B")
    attr[128] = float(node.get("element") in {"Ni", "Pd", "Cu", "Zn"})
    if role in ROLE_FEATURE_SLOTS:
        attr[ROLE_FEATURE_SLOTS[role]] = 1.0
    return attr


def edge_to_attr(edge: dict[str, str]) -> np.ndarray:
    attr = np.zeros(YIELDGNN_EDGE_FEATURE_DIM, dtype=np.float32)
    weight = parse_float(edge.get("weight", ""), default=0.0)
    bond_type = edge.get("bond_type", "").upper()
    relation_type = edge.get("relation_type", "").upper()

    if bond_type == "SINGLE":
        attr[0] = weight
    elif bond_type in {"DOUBLE", "DOUBLE_E", "DOUBLE_Z"}:
        attr[1] = weight
    elif bond_type == "TRIPLE":
        attr[2] = weight
    elif bond_type in {"AROMATIC", "DATIVE"}:
        attr[3] = weight

    if edge.get("edge_scope") == "cross_role":
        attr[4] = 1.0
        attr[5] = float(relation_type in {"FORMING_BOND", "BREAKING_BOND"})
        attr[6] = float(
            relation_type == "METAL_COORDINATION" or relation_type.endswith("_EFFECT")
        )
        attr[7] = weight
    elif not np.any(attr):
        attr[7] = weight
    return attr


def write_yieldgnn_npz(
    records: list[YieldGNNGraphRecord],
    output_path: Path,
) -> None:
    if not records:
        raise ValueError("At least one graph record is required")

    rmol_dict = [records_to_mol_dict(records)]
    pmol_dict = [dummy_product_dict(len(records))]
    reaction_dict = {
        "yld": np.zeros(len(records), dtype=np.float32),
        "rsmi": np.array([f"{record.dataset_key}>>DUMMY" for record in records]),
        "dataset_key": np.array([record.dataset_key for record in records]),
        "node_ids": np.array([record.node_ids for record in records], dtype=object),
    }
    data = np.array([rmol_dict, pmol_dict, reaction_dict], dtype=object)
    np.savez_compressed(output_path, data=data)


def records_to_mol_dict(records: list[YieldGNNGraphRecord]) -> dict[str, np.ndarray]:
    edge_attrs = [record.edge_attr for record in records if record.n_edge > 0]
    src_arrays = [record.src for record in records if record.n_edge > 0]
    dst_arrays = [record.dst for record in records if record.n_edge > 0]
    return {
        "n_node": np.array([record.n_node for record in records], dtype=int),
        "n_edge": np.array([record.n_edge for record in records], dtype=int),
        "node_attr": np.vstack([record.node_attr for record in records]).astype(np.float32),
        "edge_attr": (
            np.vstack(edge_attrs).astype(np.float32)
            if edge_attrs
            else np.empty((0, YIELDGNN_EDGE_FEATURE_DIM), dtype=np.float32)
        ),
        "src": (
            np.hstack(src_arrays).astype(int)
            if src_arrays
            else np.empty(0, dtype=int)
        ),
        "dst": (
            np.hstack(dst_arrays).astype(int)
            if dst_arrays
            else np.empty(0, dtype=int)
        ),
    }


def dummy_product_dict(record_count: int) -> dict[str, np.ndarray]:
    return {
        "n_node": np.ones(record_count, dtype=int),
        "n_edge": np.zeros(record_count, dtype=int),
        "node_attr": np.zeros((record_count, YIELDGNN_NODE_FEATURE_DIM), dtype=np.float32),
        "edge_attr": np.empty((0, YIELDGNN_EDGE_FEATURE_DIM), dtype=np.float32),
        "src": np.empty(0, dtype=int),
        "dst": np.empty(0, dtype=int),
    }


def build_dgl_graphs_from_yieldgnn_npz(npz_path: Path) -> list[Any]:
    try:
        import torch
        from dgl.convert import graph
    except ImportError as exc:
        raise RuntimeError("DGL and torch are required for DGL graph loading") from exc

    rmol_dict, _pmol_dict, _reaction_dict = np.load(
        npz_path, allow_pickle=True
    )["data"]
    mol = rmol_dict[0]
    node_csum = np.concatenate([[0], np.cumsum(mol["n_node"])])
    edge_csum = np.concatenate([[0], np.cumsum(mol["n_edge"])])
    graphs = []
    for idx in range(len(mol["n_node"])):
        start_e = edge_csum[idx]
        end_e = edge_csum[idx + 1]
        start_n = node_csum[idx]
        end_n = node_csum[idx + 1]
        dgl_graph = graph(
            (mol["src"][start_e:end_e], mol["dst"][start_e:end_e]),
            num_nodes=int(mol["n_node"][idx]),
        )
        dgl_graph.ndata["attr"] = torch.from_numpy(
            mol["node_attr"][start_n:end_n]
        ).float()
        dgl_graph.edata["edge_attr"] = torch.from_numpy(
            mol["edge_attr"][start_e:end_e]
        ).float()
        graphs.append(dgl_graph)
    return graphs


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_semicolon(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(";") if item]


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ML graph CSVs to a YieldGNN-compatible npz payload."
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="test_0.npz")
    parser.add_argument("--dataset-key", action="append", dest="dataset_keys")
    args = parser.parse_args()

    result = export_csv_graphs_to_yieldgnn_npz(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        dataset_keys=args.dataset_keys,
        output_name=args.output_name,
    )
    print(f"Wrote YieldGNN-compatible graph payload: {result['output_npz']}")
    print(f"Datasets: {result['dataset_count']}")


if __name__ == "__main__":
    main()
