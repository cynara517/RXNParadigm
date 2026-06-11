from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from reaction_graph_agent.agent import ReactionGraphAgent
from reaction_graph_agent.llm import LLMConfig


def construct_graphs(
    datasets: dict[str, str | Path] | None = None,
    llm: LLMConfig | dict[str, Any] | None = None,
    task_type: str = "full_pipeline",
    dataset_keys: Iterable[str] | None = None,
    graph_csv_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    split_seed: int = 27407,
) -> dict[str, Any]:
    """Construct reaction graphs and emit GNN-readable data plus reports.

    This intentionally mirrors small-entrypoint APIs such as
    ``reaction_prediction(...)`` or ``DrfpEncoder.encode(...)`` while keeping
    atom IDs, covalent bonds, and adjacency construction deterministic.
    """
    agent = ReactionGraphAgent(
        graph_csv_dir=graph_csv_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        dataset_sources=datasets,
        llm=llm,
    )
    return agent.run(
        task_type=task_type,
        dataset_keys=dataset_keys,
        split_seed=split_seed,
    )


def graph_construction(
    task_type: str,
    dataset_keys: Iterable[str] | None = None,
    graph_csv_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    dataset_sources: dict[str, str | Path] | None = None,
    datasets: dict[str, str | Path] | None = None,
    llm: LLMConfig | dict[str, Any] | None = None,
    split_seed: int = 27407,
    allow_llm: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper around ``construct_graphs``."""
    _ = allow_llm
    return construct_graphs(
        datasets=datasets or dataset_sources,
        llm=llm,
        task_type=task_type,
        dataset_keys=dataset_keys,
        graph_csv_dir=graph_csv_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        split_seed=split_seed,
    )
