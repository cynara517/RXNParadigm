from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ml_graph_csv_exporter import DEFAULT_OUTPUT_DIR as DEFAULT_GRAPH_CSV_DIR
from ml_graph_csv_exporter import export_ml_graph_csvs
from sample_graph_dataset_builder import (
    DEFAULT_DATASET_SOURCES,
    DEFAULT_OUTPUT_DIR as DEFAULT_SAMPLE_OUTPUT_DIR,
    build_sample_graph_datasets,
)

from reaction_graph_agent.llm import LLMConfig
from reaction_graph_agent.profiler import build_agent_quality_report
from reaction_graph_agent.reporting import build_structured_graph_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "generated" / "agent_reports"


class ReactionGraphAgent:
    """Controlled reaction graph construction agent.

    The agent is intentionally a deterministic orchestrator. It can prepare
    review queues for human approval, but it does not let an LLM generate atom
    IDs, covalent bonds, cross-role atom edges, or adjacency matrices.
    """

    def __init__(
        self,
        graph_csv_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        report_dir: str | Path | None = None,
        dataset_sources: dict[str, str | Path] | None = None,
        llm: LLMConfig | dict[str, Any] | None = None,
        allow_llm: bool = False,
    ) -> None:
        self.graph_csv_dir = Path(graph_csv_dir) if graph_csv_dir else DEFAULT_GRAPH_CSV_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_SAMPLE_OUTPUT_DIR
        self.report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
        self.dataset_sources = normalize_dataset_sources(dataset_sources)
        self.llm = llm
        self.allow_llm = allow_llm

    def run(
        self,
        task_type: str,
        dataset_keys: Iterable[str] | None = None,
        split_seed: int = 27407,
    ) -> dict[str, Any]:
        task_type = normalize_task_type(task_type)
        keys = list(dataset_keys) if dataset_keys is not None else None

        if task_type == "profile":
            return self.quality_report(dataset_keys=keys)
        if task_type == "quality_report":
            return self.quality_report(dataset_keys=keys)
        if task_type == "structured_report":
            return self.structured_report(dataset_keys=keys)
        if task_type == "ml_graph_csv":
            return self.export_ml_graph_csvs()
        if task_type == "sample_graph_dataset":
            return self.build_sample_graph_dataset(
                dataset_keys=keys,
                split_seed=split_seed,
            )
        if task_type == "full_pipeline":
            csv_result = self.export_ml_graph_csvs()
            sample_result = self.build_sample_graph_dataset(
                dataset_keys=keys,
                split_seed=split_seed,
            )
            structured_report = self.structured_report(dataset_keys=keys)
            return {
                "artifact_id": "reaction_graph_agent_full_pipeline_v1",
                "description": (
                    "Graph construction finished without blocking for human approval. "
                    "Human-control issues are reported as structured control points."
                ),
                "tasks": {
                    "ml_graph_csv": csv_result,
                    "sample_graph_dataset": sample_result,
                    "structured_report": {
                        "report_dir": str(self.report_dir),
                        "structured_report_yaml": structured_report["report_outputs"][
                            "structured_report_yaml"
                        ],
                        "structured_report_json": structured_report["report_outputs"][
                            "structured_report_json"
                        ],
                        "human_control_point_count": len(
                            structured_report["human_control_points"]
                        ),
                    },
                },
            }
        raise ValueError(f"Unsupported task_type: {task_type}")

    def construct(
        self,
        dataset_keys: Iterable[str] | None = None,
        split_seed: int = 27407,
    ) -> dict[str, Any]:
        return self.run(
            task_type="full_pipeline",
            dataset_keys=dataset_keys,
            split_seed=split_seed,
        )

    def export_ml_graph_csvs(self) -> dict[str, Any]:
        return export_ml_graph_csvs(output_dir=self.graph_csv_dir)

    def build_sample_graph_dataset(
        self,
        dataset_keys: Iterable[str] | None = None,
        split_seed: int = 27407,
    ) -> dict[str, Any]:
        return build_sample_graph_datasets(
            graph_csv_dir=self.graph_csv_dir,
            output_dir=self.output_dir,
            dataset_sources=self.dataset_sources,
            dataset_keys=dataset_keys,
            split_seed=split_seed,
        )

    def quality_report(
        self,
        dataset_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        keys = list(dataset_keys) if dataset_keys is not None else default_dataset_keys()
        missing_sources = [key for key in keys if key not in self.dataset_sources]
        if missing_sources:
            raise KeyError(f"Missing dataset source paths for: {missing_sources}")
        return build_agent_quality_report(
            dataset_keys=keys,
            graph_csv_dir=self.graph_csv_dir,
            dataset_sources=self.dataset_sources,
            report_dir=self.report_dir,
        )

    def structured_report(
        self,
        dataset_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        keys = list(dataset_keys) if dataset_keys is not None else default_dataset_keys()
        missing_sources = [key for key in keys if key not in self.dataset_sources]
        if missing_sources:
            raise KeyError(f"Missing dataset source paths for: {missing_sources}")
        return build_structured_graph_report(
            dataset_keys=keys,
            graph_csv_dir=self.graph_csv_dir,
            sample_output_dir=self.output_dir,
            dataset_sources=self.dataset_sources,
            report_dir=self.report_dir,
            llm_config=self.llm,
        )


def normalize_task_type(task_type: str) -> str:
    aliases = {
        "construct_graph": "ml_graph_csv",
        "graph_csv": "ml_graph_csv",
        "ml_graph": "ml_graph_csv",
        "samples": "sample_graph_dataset",
        "sample_dataset": "sample_graph_dataset",
        "dataset": "sample_graph_dataset",
        "report": "quality_report",
        "review": "quality_report",
        "graph_report": "structured_report",
        "structured": "structured_report",
        "construct": "full_pipeline",
        "graph_construction": "full_pipeline",
        "run_all": "full_pipeline",
    }
    return aliases.get(task_type, task_type)


def normalize_dataset_sources(
    dataset_sources: dict[str, str | Path] | None,
) -> dict[str, Path]:
    raw_sources = dataset_sources or DEFAULT_DATASET_SOURCES
    return {key: Path(path) for key, path in raw_sources.items()}


def default_dataset_keys() -> list[str]:
    return list(DEFAULT_DATASET_SOURCES)
