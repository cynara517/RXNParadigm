from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from literature_evidence_retriever import (  # noqa: E402
    retrieve_evidence_by_dataset,
    retrieve_literature_evidence,
)


def test_retrieves_buchwald_evidence() -> None:
    evidence = retrieve_literature_evidence("Buchwald-Hartwig amination")

    source_ids = {record["source_id"] for record in evidence}
    assert "buchwald_hartwig_acs_chemrev_2025" in source_ids
    assert "buchwald_hartwig_libretexts" in source_ids


def test_retrieves_suzuki_evidence() -> None:
    evidence = retrieve_literature_evidence("Suzuki-Miyaura cross-coupling")

    source_ids = {record["source_id"] for record in evidence}
    assert "suzuki_miyaura_intramolecular_review_pmc_2024" in source_ids
    assert "suzuki_transmetalation_pubmed_2013" in source_ids


def test_writes_dataset_evidence_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "literature_retrieval_evidence.yaml"

    result = retrieve_evidence_by_dataset(
        {"AZ": "Buchwald-Hartwig amination"},
        output_path=output_path,
    )

    assert result["datasets"][0]["dataset_key"] == "AZ"
    assert result["datasets"][0]["evidence"]
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result
