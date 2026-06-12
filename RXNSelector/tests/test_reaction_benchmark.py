from __future__ import annotations

import json
from pathlib import Path

from dft_preprocess_agent.core.engine import DEFAULT_SKILL_SEQUENCE, WorkflowEngine
from skills.llm_chain_decomposition import tool as chain_tool


ROOT = Path(__file__).resolve().parents[1]


def test_reaction_workflow_produces_generic_screening_artifacts() -> None:
    engine = WorkflowEngine(
        skills_dir=ROOT / "skills",
        runs_dir=ROOT / "runs",
    )
    state = engine.create_run(
        dataset_path=ROOT / "examples" / "00_01_data.csv",
        config_path=ROOT / "configs" / "reaction_dft_screening.yaml",
        run_id="pytest_reaction_benchmark",
    )
    engine.run_workflow(state, skill_sequence=DEFAULT_SKILL_SEQUENCE)
    profile = json.loads(Path(state.artifacts["dataset_profile"]).read_text(encoding="utf-8"))

    assert profile["target_column"] == "yield"
    assert profile["feature_count"] == 120
    assert set(profile["groups"]) == {"additive", "aryl_halide", "base", "ligand"}
    assert state.artifacts["mi_scores_by_group"].endswith("mi_scores_by_group.csv")
    assert "ridge_dependency_report" in state.artifacts
    assert "llm_chain_decomposition" in state.artifacts
    assert "feature_decision_table" in state.artifacts
    assert "final_features" in state.artifacts
    assert len(state.selected_features) > 0


def test_llm_chain_decomposition_adds_chain_review_without_replacing_decisions(monkeypatch) -> None:
    baseline = {
        "mode": "rule_based_llm_protocol_adapter",
        "groups": {
            "additive": {
                "chains": [
                    {
                        "chain_id": "additive_chain_1",
                        "source_component_id": "additive_component_1",
                        "importance": "high",
                        "features": ["additive_a", "additive_b"],
                        "ridge_edges": [],
                        "rendered_relation": "a ; b",
                        "retained_features": [
                            {
                                "feature": "additive_a",
                                "mi_score": 0.1,
                                "reason": "baseline keep",
                            }
                        ],
                        "removed_features": [
                            {
                                "feature": "additive_b",
                                "mi_score": 0.2,
                                "reason": "baseline remove",
                            }
                        ],
                    }
                ]
            }
        },
    }

    def fake_call(messages, settings):
        assert settings["model"] == "test-model"
        assert messages[0]["role"] == "system"
        return {
            "groups": {
                "additive": {
                    "group_summary": "Additive descriptors mainly separate steric and electronic effects.",
                    "suggested_operations": [
                        {
                            "operation": "split",
                            "chain_ids": ["additive_chain_1"],
                            "reason": "The chain mixes size and charge descriptors.",
                        }
                    ],
                    "chain_reviews": [
                        {
                            "chain_id": "additive_chain_1",
                            "recommended_action": "split",
                            "merge_with_chain_ids": [],
                            "split_suggestions": [
                                {
                                    "name": "steric/electronic descriptors",
                                    "features": ["additive_a", "additive_b"],
                                    "reason": "Both may describe additive environment effects.",
                                }
                            ],
                            "meaning_summary": "This chain captures coupled additive descriptors.",
                            "reaction_role_analysis": "It may affect yield through additive-controlled reactivity.",
                            "literature_or_search_basis": "General additive effects are commonly interpreted through steric/electronic descriptors.",
                            "confidence": "medium",
                        }
                    ]
                }
            },
            "global_summary": "LLM reviewed deterministic chains after MI/ridge screening.",
            "_llm_usage": {"total_tokens": 123},
        }

    monkeypatch.setattr(chain_tool, "_call_openai_compatible", fake_call)
    output = chain_tool._apply_llm_if_enabled(
        {
            "config": {
                "llm_chain_decomposition": {
                    "enabled": True,
                    "provider": "openai_compatible",
                    "model": "test-model",
                    "strict_validation": True,
                }
            },
            "params": {},
        },
        baseline,
    )

    chain = output["groups"]["additive"]["chains"][0]
    assert output["mode"] == "rule_based_with_llm_chain_review"
    assert output["llm_usage"] == {"total_tokens": 123}
    assert output["llm_global_summary"] == "LLM reviewed deterministic chains after MI/ridge screening."
    assert chain["retained_features"][0]["feature"] == "additive_a"
    assert chain["removed_features"][0]["feature"] == "additive_b"
    assert chain["llm_review"]["recommended_action"] == "split"
    assert chain["llm_review"]["meaning_summary"] == "This chain captures coupled additive descriptors."
    assert output["groups"]["additive"]["llm_suggested_operations"][0]["operation"] == "split"


def test_llm_chain_review_can_request_responses_web_search(monkeypatch) -> None:
    baseline = {
        "mode": "rule_based_llm_protocol_adapter",
        "groups": {
            "base": {
                "chains": [
                    {
                        "chain_id": "base_chain_1",
                        "source_component_id": "base_component_1",
                        "importance": "medium",
                        "features": ["base_a"],
                        "ridge_edges": [],
                        "rendered_relation": "a",
                        "retained_features": [
                            {"feature": "base_a", "mi_score": 0.4, "reason": "baseline keep"}
                        ],
                        "removed_features": [],
                    }
                ]
            }
        },
    }

    def fake_responses_call(messages, settings):
        user_payload = json.loads(messages[1]["content"])
        assert settings["provider"] == "openai_responses"
        assert settings["literature_search"] is True
        assert settings["web_search_tool_type"] == "web_search_preview"
        assert user_payload["literature_search_requested"] is True
        assert user_payload["user_literature_context"] == "Suzuki coupling base effects."
        return {
            "groups": {
                "base": {
                    "chain_reviews": [
                        {
                            "chain_id": "base_chain_1",
                            "recommended_action": "keep",
                            "meaning_summary": "Base descriptor reviewed with literature context.",
                            "reaction_role_analysis": "Base identity may influence transmetalation or neutralization.",
                            "literature_or_search_basis": "Search context requested.",
                            "confidence": "medium",
                        }
                    ]
                }
            },
            "_llm_usage": {"total_tokens": 50},
        }

    monkeypatch.setattr(chain_tool, "_call_openai_responses", fake_responses_call)
    output = chain_tool._apply_llm_if_enabled(
        {
            "config": {
                "llm_chain_decomposition": {
                    "enabled": True,
                    "provider": "openai_responses",
                    "model": "test-model",
                    "literature_context": "Suzuki coupling base effects.",
                }
            },
            "params": {},
        },
        baseline,
    )

    chain = output["groups"]["base"]["chains"][0]
    assert output["mode"] == "rule_based_with_llm_chain_review"
    assert chain["retained_features"][0]["feature"] == "base_a"
    assert chain["llm_review"]["literature_or_search_basis"] == "Search context requested."
