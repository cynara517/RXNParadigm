from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_FIXED_GRAPHS_PATH = ROOT / "generated" / "dft_fixed_role_graphs.yaml"
DEFAULT_CROSS_ROLE_PATH = ROOT / "generated" / "literature_cross_role_bonds.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "generated" / "ml_graph_csv"

NO_EDGE_WEIGHT = 0.0
CROSS_ROLE_WEIGHT = 0.5
BOND_TYPE_WEIGHTS = {
    "SINGLE": 0.33,
    "DOUBLE": 0.67,
    "DOUBLE_E": 0.67,
    "DOUBLE_Z": 0.67,
    "AROMATIC": 0.5,
    "DATIVE": 0.5,
    "TRIPLE": 1.0,
}

EDGE_FEATURE_SCHEMA = [
    {
        "edge_feature_name": "no_edge",
        "edge_feature_full_name": "NO_EDGE",
        "weight": NO_EDGE_WEIGHT,
        "description": "No relation or bond between the two fixed atoms.",
    },
    {
        "edge_feature_name": "intra_role_covalent_single_bond",
        "edge_feature_full_name": "INTRA_ROLE_COVALENT_SINGLE_BOND",
        "weight": BOND_TYPE_WEIGHTS["SINGLE"],
        "description": "Single covalent bond recorded by MoleCode within one role graph.",
    },
    {
        "edge_feature_name": "intra_role_covalent_double_bond",
        "edge_feature_full_name": "INTRA_ROLE_COVALENT_DOUBLE_BOND",
        "weight": BOND_TYPE_WEIGHTS["DOUBLE"],
        "description": "Double covalent bond recorded by MoleCode within one role graph.",
    },
    {
        "edge_feature_name": "intra_role_covalent_triple_bond",
        "edge_feature_full_name": "INTRA_ROLE_COVALENT_TRIPLE_BOND",
        "weight": BOND_TYPE_WEIGHTS["TRIPLE"],
        "description": "Triple covalent bond recorded by MoleCode within one role graph.",
    },
    {
        "edge_feature_name": "intra_role_covalent_aromatic_bond",
        "edge_feature_full_name": "INTRA_ROLE_COVALENT_AROMATIC_BOND",
        "weight": BOND_TYPE_WEIGHTS["AROMATIC"],
        "description": "Aromatic covalent bond recorded by MoleCode within one role graph.",
    },
    {
        "edge_feature_name": "intra_role_dative_bond",
        "edge_feature_full_name": "INTRA_ROLE_DATIVE_BOND",
        "weight": BOND_TYPE_WEIGHTS["DATIVE"],
        "description": "Dative/coordination bond recorded by MoleCode within one role graph.",
    },
    {
        "edge_feature_name": "cross_role_forming_bond",
        "edge_feature_full_name": "CROSS_ROLE_FORMING_BOND",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role forming bond proposal.",
    },
    {
        "edge_feature_name": "cross_role_breaking_bond",
        "edge_feature_full_name": "CROSS_ROLE_BREAKING_BOND",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role breaking bond proposal.",
    },
    {
        "edge_feature_name": "cross_role_metal_coordination",
        "edge_feature_full_name": "CROSS_ROLE_METAL_COORDINATION",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role metal coordination proposal.",
    },
    {
        "edge_feature_name": "cross_role_acid_base_interaction",
        "edge_feature_full_name": "CROSS_ROLE_ACID_BASE_INTERACTION",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role acid/base interaction proposal.",
    },
    {
        "edge_feature_name": "cross_role_electronic_effect",
        "edge_feature_full_name": "CROSS_ROLE_ELECTRONIC_EFFECT",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role electronic effect proposal.",
    },
    {
        "edge_feature_name": "cross_role_steric_effect",
        "edge_feature_full_name": "CROSS_ROLE_STERIC_EFFECT",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role steric effect proposal.",
    },
    {
        "edge_feature_name": "cross_role_solvent_effect",
        "edge_feature_full_name": "CROSS_ROLE_SOLVENT_EFFECT",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role solvent effect proposal.",
    },
    {
        "edge_feature_name": "cross_role_additive_effect",
        "edge_feature_full_name": "CROSS_ROLE_ADDITIVE_EFFECT",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role additive effect proposal.",
    },
    {
        "edge_feature_name": "cross_role_user_defined_relation",
        "edge_feature_full_name": "CROSS_ROLE_USER_DEFINED_RELATION",
        "weight": CROSS_ROLE_WEIGHT,
        "description": "Literature-grounded cross-role user-defined relation proposal.",
    },
]


def export_ml_graph_csvs(
    fixed_graphs_path: Path = DEFAULT_FIXED_GRAPHS_PATH,
    cross_role_path: Path = DEFAULT_CROSS_ROLE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    fixed_graphs = yaml.safe_load(fixed_graphs_path.read_text(encoding="utf-8"))
    cross_role = yaml.safe_load(cross_role_path.read_text(encoding="utf-8"))
    cross_role_index = index_cross_role_bonds(cross_role)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "edge_feature_schema.csv", EDGE_FEATURE_SCHEMA)

    manifest_rows = []
    datasets = []
    for dataset in fixed_graphs.get("datasets", []):
        dataset_key = dataset["dataset_key"]
        graph = build_dataset_graph(dataset, cross_role_index.get(dataset_key, []))
        write_dataset_csvs(output_dir, dataset_key, graph)
        manifest_rows.append(
            {
                "dataset_key": dataset_key,
                "nodes_csv": f"{dataset_key}_nodes.csv",
                "edges_csv": f"{dataset_key}_edges.csv",
                "adjacency_csv": f"{dataset_key}_adjacency.csv",
                "node_edge_features_csv": f"{dataset_key}_node_edge_feature_names.csv",
                "node_count": len(graph["nodes"]),
                "positive_edge_count": len(graph["edges"]),
            }
        )
        datasets.append(
            {
                "dataset_key": dataset_key,
                "node_count": len(graph["nodes"]),
                "positive_edge_count": len(graph["edges"]),
            }
        )

    write_csv(output_dir / "graph_manifest.csv", manifest_rows)
    return {
        "artifact_id": "ml_graph_csv_export_v1",
        "output_dir": str(output_dir),
        "datasets": datasets,
    }


def build_dataset_graph(
    dataset: dict[str, Any],
    cross_role_bonds: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes = build_nodes(dataset)
    node_ids = {node["node_id"] for node in nodes}
    edges = build_intra_role_edges(dataset, node_ids)
    edges.extend(build_cross_role_edges(cross_role_bonds, node_ids, len(edges)))
    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency": build_adjacency(nodes, edges),
        "node_edge_features": build_node_edge_features(nodes, edges),
    }


def build_nodes(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for role in dataset.get("roles", []):
        for atom in role.get("fixed_atoms", []):
            nodes.append(
                {
                    "node_index": len(nodes),
                    "node_id": atom["fixed_atom_id"],
                    "dataset_key": dataset["dataset_key"],
                    "role": role["role"],
                    "dft_atom_label": atom["dft_atom_label"],
                    "element": atom["element"],
                    "node_feature_full_names": join_list(atom.get("descriptor_columns", [])),
                    "descriptor_names": join_list(atom.get("descriptor_names", [])),
                    "likely_site_types": join_list(atom.get("likely_site_types", [])),
                }
            )
    return nodes


def build_intra_role_edges(
    dataset: dict[str, Any],
    node_ids: set[str],
) -> list[dict[str, Any]]:
    edges = []
    for role in dataset.get("roles", []):
        for fixed_edge in role.get("fixed_edges", []):
            source = fixed_edge["source_fixed_atom_id"]
            target = fixed_edge["target_fixed_atom_id"]
            if source not in node_ids or target not in node_ids:
                continue
            bond_type = choose_primary_bond_type(fixed_edge.get("bond_types", []))
            feature_name = intra_role_feature_name(bond_type)
            edges.append(
                {
                    "edge_index": len(edges),
                    "source_node_id": source,
                    "target_node_id": target,
                    "edge_scope": "intra_role",
                    "role_or_relation": role["role"],
                    "edge_feature_name": feature_name,
                    "edge_feature_full_name": feature_name.upper(),
                    "relation_type": "COVALENT_BOND",
                    "bond_type": bond_type,
                    "weight": BOND_TYPE_WEIGHTS.get(bond_type, NO_EDGE_WEIGHT),
                    "evidence_sources": "",
                }
            )
    return edges


def build_cross_role_edges(
    cross_role_bonds: list[dict[str, Any]],
    node_ids: set[str],
    start_index: int,
) -> list[dict[str, Any]]:
    edges = []
    for bond in cross_role_bonds:
        source = bond["source_fixed_atom_id"]
        target = bond["target_fixed_atom_id"]
        if source not in node_ids or target not in node_ids:
            continue
        feature_name = cross_role_feature_name(bond["relation_type"])
        edges.append(
            {
                "edge_index": start_index + len(edges),
                "source_node_id": source,
                "target_node_id": target,
                "edge_scope": "cross_role",
                "role_or_relation": bond["relation_type"],
                "edge_feature_name": feature_name,
                "edge_feature_full_name": feature_name.upper(),
                "relation_type": bond["relation_type"],
                "bond_type": "",
                "weight": CROSS_ROLE_WEIGHT,
                "evidence_sources": join_list(bond.get("evidence_sources", [])),
            }
        )
    return edges


def choose_primary_bond_type(bond_types: list[str]) -> str:
    priority = ["TRIPLE", "DOUBLE_E", "DOUBLE_Z", "DOUBLE", "AROMATIC", "DATIVE", "SINGLE"]
    for bond_type in priority:
        if bond_type in bond_types:
            return bond_type
    return bond_types[0] if bond_types else "SINGLE"


def intra_role_feature_name(bond_type: str) -> str:
    normalized = {
        "SINGLE": "single",
        "DOUBLE": "double",
        "DOUBLE_E": "double",
        "DOUBLE_Z": "double",
        "TRIPLE": "triple",
        "AROMATIC": "aromatic",
        "DATIVE": "dative",
    }.get(bond_type, bond_type.lower())
    if normalized == "dative":
        return "intra_role_dative_bond"
    return f"intra_role_covalent_{normalized}_bond"


def cross_role_feature_name(relation_type: str) -> str:
    return f"cross_role_{relation_type.lower()}"


def build_adjacency(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    node_ids = [node["node_id"] for node in nodes]
    weights = {(node_id, node_id): NO_EDGE_WEIGHT for node_id in node_ids}
    for edge in edges:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        weight = float(edge["weight"])
        weights[(source, target)] = max(weights.get((source, target), 0.0), weight)
        weights[(target, source)] = max(weights.get((target, source), 0.0), weight)

    rows = []
    for source in node_ids:
        row = {"node_id": source}
        for target in node_ids:
            row[target] = weights.get((source, target), NO_EDGE_WEIGHT)
        rows.append(row)
    return rows


def build_node_edge_features(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, str]]:
    incident: dict[str, list[str]] = {node["node_id"]: [] for node in nodes}
    for edge in edges:
        feature = edge["edge_feature_full_name"]
        incident[edge["source_node_id"]].append(feature)
        incident[edge["target_node_id"]].append(feature)
    return [
        {
            "node_id": node["node_id"],
            "incident_edge_feature_full_names": join_list(sorted(set(incident[node["node_id"]]))),
            "all_edge_feature_full_names": join_list(
                row["edge_feature_full_name"] for row in EDGE_FEATURE_SCHEMA
            ),
        }
        for node in nodes
    ]


def write_dataset_csvs(output_dir: Path, dataset_key: str, graph: dict[str, Any]) -> None:
    write_csv(output_dir / f"{dataset_key}_nodes.csv", graph["nodes"])
    write_csv(output_dir / f"{dataset_key}_edges.csv", graph["edges"])
    write_csv(output_dir / f"{dataset_key}_adjacency.csv", graph["adjacency"])
    write_csv(
        output_dir / f"{dataset_key}_node_edge_feature_names.csv",
        graph["node_edge_features"],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_cross_role_bonds(cross_role: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        dataset["dataset_key"]: dataset.get("cross_role_bonds", [])
        for dataset in cross_role.get("datasets", [])
    }


def join_list(values: Any) -> str:
    if not values:
        return ""
    return ";".join(str(value) for value in values)


def main() -> None:
    export_ml_graph_csvs()
    print(f"Wrote CSV graph files under {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
