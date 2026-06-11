from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_GRAPH_CSV_DIR = ROOT / "generated" / "ml_graph_csv"
DEFAULT_OUTPUT_DIR = ROOT / "generated" / "sample_graph_dataset"

DEFAULT_DATASET_SOURCES = {
    "AZ": ROOT / "datasets" / "az_no_rdkit.csv",
    "SU_NO": ROOT / "datasets" / "su_no_rdkit.csv",
    "DY": ROOT / "datasets" / "dy_dft.csv",
}

DEFAULT_SPLIT_FRACTIONS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def build_sample_graph_datasets(
    graph_csv_dir: Path = DEFAULT_GRAPH_CSV_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_sources: dict[str, Path] | None = None,
    dataset_keys: Iterable[str] | None = None,
    split_seed: int = 27407,
) -> dict[str, Any]:
    dataset_sources = dataset_sources or DEFAULT_DATASET_SOURCES
    keys = list(dataset_keys) if dataset_keys is not None else resolve_dataset_keys(graph_csv_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    split_rows = []
    node_schema_rows = []
    edge_schema_rows = []

    for dataset_key in keys:
        if dataset_key not in dataset_sources:
            raise KeyError(f"No source CSV configured for dataset {dataset_key}")
        source_csv = dataset_sources[dataset_key]
        graph = load_graph_definition(graph_csv_dir, dataset_key)
        source_df = pd.read_csv(source_csv)
        sample_graph = build_dataset_sample_graph(
            dataset_key=dataset_key,
            source_csv=source_csv,
            source_df=source_df,
            graph=graph,
            split_seed=split_seed,
        )
        output_npz = output_dir / f"{dataset_key}_samples.npz"
        write_sample_graph_npz(sample_graph, output_npz)

        dataset_rows.append(
            {
                "dataset_key": dataset_key,
                "source_csv": str(source_csv),
                "output_npz": str(output_npz),
                "sample_count": sample_graph["sample_count"],
                "node_count": sample_graph["node_count"],
                "undirected_edge_count": sample_graph["undirected_edge_count"],
                "directed_edge_count": sample_graph["directed_edge_count"],
                "node_feature_dim": len(sample_graph["node_feature_names"]),
                "edge_feature_dim": len(sample_graph["edge_feature_names"]),
                "missing_source_column_count": len(sample_graph["missing_source_columns"]),
                "yield_mean": sample_graph["yield_mean"],
                "yield_std": sample_graph["yield_std"],
            }
        )
        split_rows.extend(sample_graph["split_rows"])
        node_schema_rows.extend(sample_graph["node_schema_rows"])
        edge_schema_rows.extend(sample_graph["edge_schema_rows"])

    write_csv(output_dir / "dataset_manifest.csv", dataset_rows)
    write_csv(output_dir / "split_manifest.csv", split_rows)
    write_csv(output_dir / "node_feature_schema.csv", node_schema_rows)
    write_csv(output_dir / "edge_feature_schema.csv", unique_rows(edge_schema_rows))

    return {
        "artifact_id": "sample_graph_dataset_v1",
        "output_dir": str(output_dir),
        "datasets": dataset_rows,
    }


def resolve_dataset_keys(graph_csv_dir: Path) -> list[str]:
    manifest_path = graph_csv_dir / "graph_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing graph manifest: {manifest_path}")
    return [row["dataset_key"] for row in read_csv(manifest_path)]


def load_graph_definition(graph_csv_dir: Path, dataset_key: str) -> dict[str, Any]:
    nodes_path = graph_csv_dir / f"{dataset_key}_nodes.csv"
    edges_path = graph_csv_dir / f"{dataset_key}_edges.csv"
    edge_schema_path = graph_csv_dir / "edge_feature_schema.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes CSV: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges CSV: {edges_path}")
    if not edge_schema_path.exists():
        raise FileNotFoundError(f"Missing edge feature schema CSV: {edge_schema_path}")
    return {
        "nodes": read_csv(nodes_path),
        "edges": read_csv(edges_path),
        "edge_schema": read_csv(edge_schema_path),
    }


def build_dataset_sample_graph(
    dataset_key: str,
    source_csv: Path,
    source_df: pd.DataFrame,
    graph: dict[str, Any],
    split_seed: int,
) -> dict[str, Any]:
    if "yield" not in source_df.columns:
        raise ValueError(f"Missing required yield column in {source_csv}")

    nodes = graph["nodes"]
    edges = graph["edges"]
    node_ids = [node["node_id"] for node in nodes]
    node_feature_names = collect_node_feature_names(nodes)
    node_features, node_feature_mask, missing_source_columns = build_node_feature_tensor(
        source_df, nodes, node_feature_names
    )
    edge_index, edge_attr, edge_feature_names = build_directed_edge_tensors(
        nodes, edges, graph["edge_schema"]
    )
    y = pd.to_numeric(source_df["yield"], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(y).any():
        raise ValueError(f"Yield column contains missing or non-numeric values in {source_csv}")

    source_row_index = source_df.index.to_numpy(dtype=int)
    source_ids = build_source_ids(source_df)
    splits = make_splits(len(source_df), dataset_key, split_seed)
    split_rows = build_split_rows(dataset_key, source_row_index, source_ids, y, splits)
    adjacency = build_weighted_adjacency(nodes, edges)
    node_schema_rows = build_node_schema_rows(dataset_key, nodes, node_feature_names)
    edge_schema_rows = build_edge_schema_rows(dataset_key, edge_feature_names)

    return {
        "dataset_key": dataset_key,
        "sample_count": len(source_df),
        "node_count": len(nodes),
        "undirected_edge_count": len(edges),
        "directed_edge_count": edge_index.shape[1],
        "node_ids": node_ids,
        "node_roles": [node["role"] for node in nodes],
        "node_dft_atom_labels": [node["dft_atom_label"] for node in nodes],
        "node_elements": [node["element"] for node in nodes],
        "node_feature_names": node_feature_names,
        "node_features": node_features,
        "node_feature_mask": node_feature_mask,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_feature_names": edge_feature_names,
        "adjacency": adjacency,
        "y": y,
        "source_row_index": source_row_index,
        "source_ids": source_ids,
        "splits": splits,
        "split_rows": split_rows,
        "node_schema_rows": node_schema_rows,
        "edge_schema_rows": edge_schema_rows,
        "missing_source_columns": sorted(missing_source_columns),
        "yield_mean": float(np.mean(y)),
        "yield_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
    }


def collect_node_feature_names(nodes: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for node in nodes:
        for descriptor_name in split_semicolon(node.get("descriptor_names", "")):
            if descriptor_name not in names:
                names.append(descriptor_name)
    if not names:
        raise ValueError("At least one node descriptor name is required")
    return names


def build_node_feature_tensor(
    source_df: pd.DataFrame,
    nodes: list[dict[str, str]],
    node_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    feature_index = {name: idx for idx, name in enumerate(node_feature_names)}
    node_features = np.zeros(
        (len(source_df), len(nodes), len(node_feature_names)), dtype=np.float32
    )
    node_feature_mask = np.zeros_like(node_features, dtype=np.float32)
    missing_source_columns: set[str] = set()

    for node_idx, node in enumerate(nodes):
        source_columns = split_semicolon(node.get("node_feature_full_names", ""))
        descriptor_names = split_semicolon(node.get("descriptor_names", ""))
        for source_column, descriptor_name in zip(source_columns, descriptor_names):
            if descriptor_name not in feature_index:
                continue
            if source_column not in source_df.columns:
                missing_source_columns.add(source_column)
                continue
            values = pd.to_numeric(source_df[source_column], errors="coerce")
            valid = values.notna().to_numpy()
            feature_idx = feature_index[descriptor_name]
            node_features[valid, node_idx, feature_idx] = values[valid].to_numpy(
                dtype=np.float32
            )
            node_feature_mask[valid, node_idx, feature_idx] = 1.0

    return node_features, node_feature_mask, missing_source_columns


def build_directed_edge_tensors(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
    edge_schema: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    node_index = {node["node_id"]: idx for idx, node in enumerate(nodes)}
    categorical_features = [
        row["edge_feature_full_name"]
        for row in edge_schema
        if row["edge_feature_full_name"] != "NO_EDGE"
    ]
    edge_feature_names = ["edge_weight", *categorical_features, "is_intra_role", "is_cross_role"]
    edge_feature_index = {name: idx for idx, name in enumerate(edge_feature_names)}

    directed_edges: list[tuple[int, int]] = []
    edge_attrs: list[np.ndarray] = []
    for edge in edges:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        if source not in node_index or target not in node_index:
            continue
        attr = np.zeros(len(edge_feature_names), dtype=np.float32)
        weight = parse_float(edge.get("weight", ""), default=0.0)
        attr[edge_feature_index["edge_weight"]] = weight
        feature_name = edge.get("edge_feature_full_name", "")
        if feature_name in edge_feature_index:
            attr[edge_feature_index[feature_name]] = 1.0
        if edge.get("edge_scope") == "intra_role":
            attr[edge_feature_index["is_intra_role"]] = 1.0
        if edge.get("edge_scope") == "cross_role":
            attr[edge_feature_index["is_cross_role"]] = 1.0

        source_idx = node_index[source]
        target_idx = node_index[target]
        directed_edges.extend([(source_idx, target_idx), (target_idx, source_idx)])
        edge_attrs.extend([attr, attr.copy()])

    if directed_edges:
        edge_index = np.array(directed_edges, dtype=int).T
        edge_attr = np.vstack(edge_attrs).astype(np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=int)
        edge_attr = np.empty((0, len(edge_feature_names)), dtype=np.float32)
    return edge_index, edge_attr, edge_feature_names


def build_weighted_adjacency(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> np.ndarray:
    node_index = {node["node_id"]: idx for idx, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for edge in edges:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        if source not in node_index or target not in node_index:
            continue
        weight = parse_float(edge.get("weight", ""), default=0.0)
        source_idx = node_index[source]
        target_idx = node_index[target]
        adjacency[source_idx, target_idx] = max(adjacency[source_idx, target_idx], weight)
        adjacency[target_idx, source_idx] = max(adjacency[target_idx, source_idx], weight)
    return adjacency


def build_source_ids(source_df: pd.DataFrame) -> np.ndarray:
    if "id" in source_df.columns:
        return source_df["id"].astype(str).to_numpy(dtype=object)
    return source_df.index.astype(str).to_numpy(dtype=object)


def make_splits(sample_count: int, dataset_key: str, split_seed: int) -> np.ndarray:
    seed = split_seed + zlib.crc32(dataset_key.encode("utf-8"))
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(sample_count)
    train_count = int(sample_count * DEFAULT_SPLIT_FRACTIONS["train"])
    val_count = int(sample_count * DEFAULT_SPLIT_FRACTIONS["val"])
    splits = np.empty(sample_count, dtype=object)
    splits[permuted[:train_count]] = "train"
    splits[permuted[train_count : train_count + val_count]] = "val"
    splits[permuted[train_count + val_count :]] = "test"
    return splits


def build_split_rows(
    dataset_key: str,
    source_row_index: np.ndarray,
    source_ids: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_key": dataset_key,
            "sample_index": sample_idx,
            "source_row_index": int(source_row_index[sample_idx]),
            "source_id": source_ids[sample_idx],
            "split": splits[sample_idx],
            "yield": float(y[sample_idx]),
        }
        for sample_idx in range(len(y))
    ]


def build_node_schema_rows(
    dataset_key: str,
    nodes: list[dict[str, str]],
    node_feature_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for feature_idx, feature_name in enumerate(node_feature_names):
        rows.append(
            {
                "dataset_key": dataset_key,
                "feature_index": feature_idx,
                "feature_name": feature_name,
                "feature_kind": "node_dft_descriptor_value",
            }
        )
    for node in nodes:
        rows.append(
            {
                "dataset_key": dataset_key,
                "feature_index": "",
                "feature_name": f"{node['node_id']} source columns",
                "feature_kind": node.get("node_feature_full_names", ""),
            }
        )
    return rows


def build_edge_schema_rows(
    dataset_key: str,
    edge_feature_names: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_key": dataset_key,
            "feature_index": feature_idx,
            "feature_name": feature_name,
            "feature_kind": "edge_weight_or_category",
        }
        for feature_idx, feature_name in enumerate(edge_feature_names)
    ]


def write_sample_graph_npz(sample_graph: dict[str, Any], output_npz: Path) -> None:
    np.savez_compressed(
        output_npz,
        dataset_key=np.array(sample_graph["dataset_key"], dtype=object),
        node_ids=np.array(sample_graph["node_ids"], dtype=object),
        node_roles=np.array(sample_graph["node_roles"], dtype=object),
        node_dft_atom_labels=np.array(sample_graph["node_dft_atom_labels"], dtype=object),
        node_elements=np.array(sample_graph["node_elements"], dtype=object),
        node_feature_names=np.array(sample_graph["node_feature_names"], dtype=object),
        node_features=sample_graph["node_features"],
        node_feature_mask=sample_graph["node_feature_mask"],
        edge_index=sample_graph["edge_index"],
        edge_attr=sample_graph["edge_attr"],
        edge_feature_names=np.array(sample_graph["edge_feature_names"], dtype=object),
        adjacency=sample_graph["adjacency"],
        y=sample_graph["y"],
        source_row_index=sample_graph["source_row_index"],
        source_ids=sample_graph["source_ids"],
        splits=sample_graph["splits"],
        missing_source_columns=np.array(sample_graph["missing_source_columns"], dtype=object),
    )


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


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in rows:
        key = tuple(row.items())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def split_semicolon(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(";") if item]


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-experiment graph samples from fixed role graphs and DFT CSVs."
    )
    parser.add_argument("--graph-csv-dir", type=Path, default=DEFAULT_GRAPH_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-key", action="append", dest="dataset_keys")
    parser.add_argument("--split-seed", type=int, default=27407)
    args = parser.parse_args()

    result = build_sample_graph_datasets(
        graph_csv_dir=args.graph_csv_dir,
        output_dir=args.output_dir,
        dataset_keys=args.dataset_keys,
        split_seed=args.split_seed,
    )
    print(f"Wrote sample graph datasets under {result['output_dir']}")
    for dataset in result["datasets"]:
        print(
            "{dataset_key}: samples={sample_count}, nodes={node_count}, "
            "directed_edges={directed_edge_count}, node_dim={node_feature_dim}, "
            "edge_dim={edge_feature_dim}, missing_columns={missing_source_column_count}".format(
                **dataset
            )
        )


if __name__ == "__main__":
    main()
