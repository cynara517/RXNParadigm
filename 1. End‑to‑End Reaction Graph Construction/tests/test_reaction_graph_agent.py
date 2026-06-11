from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reaction_graph_agent import ReactionGraphAgent  # noqa: E402
from reaction_graph_agent.eval import construct_graphs, graph_construction  # noqa: E402
from reaction_graph_agent.reporting import match_evidence  # noqa: E402


def test_graph_construction_quality_report_api(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    report_dir = tmp_path / "reports"
    source_csv = tmp_path / "source.csv"
    graph_dir.mkdir()
    write_agent_graph_csvs(graph_dir)
    write_csv(
        source_csv,
        [
            {
                "yield": "10.0",
                "ligand_molecular_weight": "100.0",
                "ligand_.P1_NMR_shift": "1.0",
                "ligand_.P1_electrostatic_charge": "-0.1",
                "halide_.C1_NMR_shift": "110.0",
                "halide_.C1_electrostatic_charge": "-0.2",
            }
        ],
    )

    result = graph_construction(
        task_type="quality_report",
        dataset_keys=["AZ"],
        graph_csv_dir=graph_dir,
        report_dir=report_dir,
        dataset_sources={"AZ": source_csv},
    )

    assert result["artifact_id"] == "reaction_graph_agent_quality_report_v1"
    item_types = {item["item_type"] for item in result["review_queue"]}
    assert "cross_role_edge" in item_types
    assert "minimal_reaction_center_graph" in item_types
    assert "sparse_role_topology" in item_types
    assert (report_dir / "review_queue.yaml").exists()
    assert (report_dir / "dataset_profile.csv").exists()


def test_reaction_graph_agent_builds_sample_dataset(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    output_dir = tmp_path / "samples"
    source_csv = tmp_path / "source.csv"
    graph_dir.mkdir()
    write_agent_graph_csvs(graph_dir)
    write_csv(
        source_csv,
        [
            {
                "id": "rxn_1",
                "yield": "10.0",
                "ligand_.P1_NMR_shift": "1.0",
                "ligand_.P1_electrostatic_charge": "-0.1",
                "halide_.C1_NMR_shift": "110.0",
                "halide_.C1_electrostatic_charge": "-0.2",
            },
            {
                "id": "rxn_2",
                "yield": "20.0",
                "ligand_.P1_NMR_shift": "2.0",
                "ligand_.P1_electrostatic_charge": "-0.2",
                "halide_.C1_NMR_shift": "120.0",
                "halide_.C1_electrostatic_charge": "-0.3",
            },
        ],
    )
    agent = ReactionGraphAgent(
        graph_csv_dir=graph_dir,
        output_dir=output_dir,
        dataset_sources={"AZ": source_csv},
    )

    result = agent.run("sample_graph_dataset", dataset_keys=["AZ"], split_seed=1)

    assert result["datasets"][0]["sample_count"] == 2
    assert result["datasets"][0]["node_count"] == 2
    assert (output_dir / "AZ_samples.npz").exists()


def test_construct_graphs_returns_structured_report(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    report_dir = tmp_path / "reports"
    source_csv = tmp_path / "source.csv"
    graph_dir.mkdir()
    write_agent_graph_csvs(graph_dir)
    write_csv(
        source_csv,
        [
            {
                "yield": "10.0",
                "ligand_molecular_weight": "100.0",
                "ligand_.P1_NMR_shift": "1.0",
                "ligand_.P1_electrostatic_charge": "-0.1",
                "halide_.C1_NMR_shift": "110.0",
                "halide_.C1_electrostatic_charge": "-0.2",
            }
        ],
    )

    result = construct_graphs(
        datasets={"AZ": source_csv},
        llm={
            "provider": "openai_compatible",
            "model": "",
            "api_key": "",
        },
        task_type="structured_report",
        dataset_keys=["AZ"],
        graph_csv_dir=graph_dir,
        report_dir=report_dir,
    )

    assert result["artifact_id"] == "reaction_graph_agent_structured_report_v1"
    assert result["datasets"][0]["gnn_data"]["format"] == "npz"
    relation = result["datasets"][0]["cross_role_relations"][0]
    assert relation["llm_search_and_parse_reason"]["llm_status"] == "not_called"
    assert "human_control_points" in result
    assert (report_dir / "structured_graph_report.yaml").exists()
    assert (report_dir / "cross_role_relation_report.csv").exists()


def test_cross_role_report_uses_curated_evidence_fallback() -> None:
    evidence = match_evidence(
        {
            "source_node_id": "su_no:aryl_halide:C1",
            "target_node_id": "su_no:organoboron:C1",
            "relation_type": "FORMING_BOND",
            "evidence_sources": "suzuki_miyaura_intramolecular_review_pmc_2024",
        },
        [],
    )

    assert evidence
    assert evidence[0]["source_id"] == "suzuki_miyaura_intramolecular_review_pmc_2024"
    assert "Suzuki Miyaura" in evidence[0]["retrieval_query"]


def write_agent_graph_csvs(path: Path) -> None:
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
                "node_id": "az:ligand:P1",
                "dataset_key": "AZ",
                "role": "ligand",
                "dft_atom_label": "P1",
                "element": "P",
                "node_feature_full_names": "ligand_.P1_NMR_shift;ligand_.P1_electrostatic_charge",
                "descriptor_names": "NMR_shift;electrostatic_charge",
                "likely_site_types": "LIGAND_DONOR_ATOM",
            },
            {
                "node_index": "1",
                "node_id": "az:aryl_halide:C1",
                "dataset_key": "AZ",
                "role": "aryl_halide",
                "dft_atom_label": "C1",
                "element": "C",
                "node_feature_full_names": "halide_.C1_NMR_shift;halide_.C1_electrostatic_charge",
                "descriptor_names": "NMR_shift;electrostatic_charge",
                "likely_site_types": "ARYL_C_IPSO",
            },
        ],
    )
    write_csv(
        path / "AZ_edges.csv",
        [
            {
                "edge_index": "0",
                "source_node_id": "az:ligand:P1",
                "target_node_id": "az:aryl_halide:C1",
                "edge_scope": "cross_role",
                "role_or_relation": "METAL_COORDINATION",
                "edge_feature_name": "cross_role_metal_coordination",
                "edge_feature_full_name": "CROSS_ROLE_METAL_COORDINATION",
                "relation_type": "METAL_COORDINATION",
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
                "edge_feature_name": "cross_role_metal_coordination",
                "edge_feature_full_name": "CROSS_ROLE_METAL_COORDINATION",
                "weight": "0.5",
                "description": "Coordination.",
            },
        ],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
