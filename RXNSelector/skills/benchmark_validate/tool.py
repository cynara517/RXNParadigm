from __future__ import annotations

import json
from pathlib import Path

import yaml

from dft_preprocess_agent.screening.analysis import ensure_dir


def run(context: dict) -> dict:
    state = context["state"]
    params = context.get("params", {})
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_07_benchmark_validate")

    expected_path = params.get("expected_features_path")
    if not expected_path:
        raise ValueError("expected_features_path is required for benchmark validation.")
    expected_data = yaml.safe_load(Path(expected_path).read_text(encoding="utf-8"))
    expected = expected_data.get("features", expected_data)
    selected = state.get("selected_features", [])
    missing = sorted(set(expected) - set(selected))
    extra = sorted(set(selected) - set(expected))
    matched = not missing and not extra
    result = {
        "matched": matched,
        "expected_count": len(expected),
        "selected_count": len(selected),
        "missing": missing,
        "extra": extra,
    }
    output_path = output_dir / "benchmark_validation.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "success" if matched else "failed",
        "artifacts": {"benchmark_validation": str(output_path)},
        "message": "Benchmark matched." if matched else "Benchmark mismatch.",
    }
