from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

import pandas as pd

from dft_preprocess_agent.screening.analysis import ensure_dir


def _importance(max_value: float) -> str:
    if max_value >= 0.97:
        return "high"
    if max_value >= 0.9:
        return "medium"
    return "low"


def _mi_map(path: str) -> dict[str, float]:
    mi_df = pd.read_csv(path)
    return {row.feature: float(row.mi_score) for row in mi_df.itertuples()}


def _short_name(feature: str, group_name: str) -> str:
    prefix = f"{group_name}_"
    return feature[len(prefix) :] if feature.startswith(prefix) else feature


def _chain_relation_text(chain: dict, group_name: str, include_values: bool = True) -> str:
    features = chain.get("features", [])
    edges = chain.get("ridge_edges", [])
    if not features:
        return ""
    if len(features) == 1:
        return _short_name(features[0], group_name)
    feature_set = set(features)
    component_edges = [
        edge
        for edge in edges
        if edge["feature_1"] in feature_set and edge["feature_2"] in feature_set
    ]
    if not component_edges:
        return " ; ".join(_short_name(feature, group_name) for feature in features)
    relation_parts = []
    for edge in sorted(component_edges, key=lambda item: -item["abs_ridge_value"]):
        left = _short_name(edge["feature_1"], group_name)
        right = _short_name(edge["feature_2"], group_name)
        value = float(edge["ridge_value"])
        arrow = f"↔<sup>{value:.3f}</sup>" if include_values else "↔"
        relation_parts.append(f"{left} {arrow} {right}")
    return " ; ".join(relation_parts)


def _build_rule_based_output(ridge_report: dict, mi_scores: dict[str, float]) -> dict:
    output = {
        "mode": "rule_based_llm_protocol_adapter",
        "note": (
            "No external LLM client is configured in this repository. This step "
            "uses the same structured inputs and produces the LLM contract output "
            "with deterministic MI/ridge rules."
        ),
        "groups": {},
    }

    for group_name, group_data in ridge_report["groups"].items():
        chains = []
        handled_features: set[str] = set()
        edges = group_data["ridge_edges"]
        for raw_component in group_data["raw_components"]:
            features = raw_component["features"]
            component_edges = [
                edge for edge in edges if edge["feature_1"] in features and edge["feature_2"] in features
            ]
            max_edge = max((edge["abs_ridge_value"] for edge in component_edges), default=0.0)
            retained_feature = max(features, key=lambda feature: mi_scores.get(feature, 0.0))
            removed = [feature for feature in features if feature != retained_feature]
            chain = {
                "chain_id": raw_component["component_id"].replace("component", "chain"),
                "source_component_id": raw_component["component_id"],
                "importance": _importance(float(max_edge)),
                "features": features,
                "ridge_edges": component_edges,
                "rendered_relation": "",
                "retained_features": [
                    {
                        "feature": retained_feature,
                        "mi_score": mi_scores.get(retained_feature, 0.0),
                        "reason": "Retained because it has the highest MI score within this ridge chain.",
                    }
                ],
                "removed_features": [
                    {
                        "feature": feature,
                        "mi_score": mi_scores.get(feature, 0.0),
                        "reason": (
                            "Removed because it belongs to the same ridge chain and has lower MI "
                            f"than {retained_feature}."
                        ),
                    }
                    for feature in removed
                ],
            }
            chain["rendered_relation"] = _chain_relation_text(chain, group_name, include_values=True)
            chains.append(chain)
            handled_features.update(features)

        multicollinear_features = [
            feature
            for feature, variable_type in group_data["feature_types"].items()
            if variable_type == "multicollinear_variable" and feature not in handled_features
        ]
        for feature in multicollinear_features:
            formula = group_data["formulas"].get(feature, {})
            chain = {
                "chain_id": f"{group_name}_multicollinear_{len(chains) + 1}",
                "source_component_id": None,
                "importance": "medium",
                "features": [feature],
                "ridge_edges": [],
                "rendered_relation": _short_name(feature, group_name),
                "retained_features": [
                    {
                        "feature": feature,
                        "mi_score": mi_scores.get(feature, 0.0),
                        "reason": (
                            "Retained for review because it is multicollinear but has no "
                            "ridge-chain alternative in this component."
                        ),
                    }
                ],
                "removed_features": [],
                "ridge_formula": formula,
            }
            chains.append(chain)

        output["groups"][group_name] = {"chains": chains}

    return output


def _chain_payload(rule_output: dict) -> dict:
    groups = {}
    for group_name, group_data in rule_output["groups"].items():
        groups[group_name] = {
            "chains": [
                {
                    "chain_id": chain["chain_id"],
                    "source_component_id": chain.get("source_component_id"),
                    "importance": chain["importance"],
                    "features": chain["features"],
                    "ridge_edges": chain.get("ridge_edges", []),
                    "rendered_relation": chain.get("rendered_relation", ""),
                    "baseline_retained_features": chain.get("retained_features", []),
                    "baseline_removed_features": chain.get("removed_features", []),
                    "ridge_formula": chain.get("ridge_formula"),
                }
                for chain in group_data["chains"]
            ]
        }
    return {"groups": groups}


def _llm_config(context: dict) -> dict:
    config = context.get("config", {}).get("llm_chain_decomposition", {}) or {}
    raw_params = context.get("params", {}) or {}
    params = raw_params.get("llm_chain_decomposition", raw_params) or {}
    merged = {**config, **params}
    merged.setdefault("enabled", False)
    merged.setdefault("provider", "openai_responses")
    merged.setdefault("base_url", "https://api.openai.com/v1")
    merged.setdefault("api_key_env", "OPENAI_API_KEY")
    merged.setdefault("timeout_seconds", 60)
    merged.setdefault("temperature", 0.0)
    merged.setdefault("max_tokens", 4000)
    merged.setdefault("fallback_on_error", True)
    merged.setdefault("strict_validation", True)
    merged.setdefault("literature_search", True)
    merged.setdefault("web_search_tool_type", "web_search_preview")
    merged.setdefault("literature_context", "")
    return merged


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _call_openai_compatible(messages: list[dict], settings: dict) -> dict:
    api_key = os.getenv(settings["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"Environment variable {settings['api_key_env']} is not set.")
    model = settings.get("model")
    if not model:
        raise RuntimeError("llm_chain_decomposition.model is required when LLM mode is enabled.")

    base_url = str(settings["base_url"]).rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(settings["temperature"]),
        "max_tokens": int(settings["max_tokens"]),
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(settings["timeout_seconds"])) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    completion = json.loads(response_body)
    content = completion["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)
    parsed.setdefault("_llm_usage", completion.get("usage", {}))
    return parsed


def _response_output_text(response_body: dict) -> str:
    if response_body.get("output_text"):
        return response_body["output_text"]
    texts = []
    for item in response_body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if "text" in content:
                texts.append(content["text"])
            elif content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(text for text in texts if text).strip()


def _call_openai_responses(messages: list[dict], settings: dict) -> dict:
    api_key = os.getenv(settings["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"Environment variable {settings['api_key_env']} is not set.")
    model = settings.get("model")
    if not model:
        raise RuntimeError("llm_chain_decomposition.model is required when LLM mode is enabled.")

    system_message = next((item["content"] for item in messages if item["role"] == "system"), "")
    user_message = next((item["content"] for item in messages if item["role"] == "user"), "")
    payload = {
        "model": model,
        "instructions": system_message,
        "input": user_message,
        "temperature": float(settings["temperature"]),
        "max_output_tokens": int(settings["max_tokens"]),
    }
    if settings.get("literature_search", True):
        payload["tools"] = [{"type": settings.get("web_search_tool_type", "web_search_preview")}]

    base_url = str(settings["base_url"]).rstrip("/")
    req = request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(settings["timeout_seconds"])) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    completion = json.loads(response_text)
    parsed = _extract_json_object(_response_output_text(completion))
    parsed.setdefault("_llm_usage", completion.get("usage", {}))
    if completion.get("sources"):
        parsed.setdefault("_llm_sources", completion["sources"])
    return parsed


def _normalize_llm_review(llm_output: dict, baseline: dict, strict: bool) -> dict:
    if "groups" not in llm_output or not isinstance(llm_output["groups"], dict):
        raise ValueError("LLM output must contain a groups object.")

    normalized = {
        "mode": "rule_based_with_llm_chain_review",
        "provider": llm_output.get("provider", "openai_compatible"),
        "model": llm_output.get("model"),
        "note": llm_output.get(
            "note",
            "Default MI/ridge chain decisions were generated first, then reviewed by an external LLM.",
        ),
        "llm_usage": llm_output.get("_llm_usage", llm_output.get("llm_usage", {})),
        "llm_global_summary": llm_output.get("global_summary", ""),
        "groups": {},
    }

    for group_name, baseline_group in baseline["groups"].items():
        llm_group = llm_output["groups"].get(group_name, {})
        llm_chains = {
            chain.get("chain_id"): chain
            for chain in llm_group.get("chain_reviews", llm_group.get("chains", []))
            if isinstance(chain, dict) and chain.get("chain_id")
        }
        normalized_chains = []
        for baseline_chain in baseline_group["chains"]:
            chain_id = baseline_chain["chain_id"]
            llm_chain = llm_chains.get(chain_id)
            chain = dict(baseline_chain)
            if not llm_chain:
                if strict:
                    raise ValueError(f"LLM output missing chain {chain_id}.")
                normalized_chains.append(chain)
                continue

            feature_set = set(baseline_chain["features"])
            for related_id in llm_chain.get("merge_with_chain_ids", []):
                if related_id not in {
                    item["chain_id"] for item in baseline_group["chains"]
                } and strict:
                    raise ValueError(f"LLM suggested unknown merge target {related_id!r} for {chain_id}.")
            split_groups = llm_chain.get("split_suggestions", [])
            if split_groups:
                split_features = {
                    feature
                    for split_group in split_groups
                    for feature in split_group.get("features", [])
                }
                unknown = split_features - feature_set
                if unknown:
                    raise ValueError(f"LLM returned unknown split features for {chain_id}: {sorted(unknown)}.")

            if llm_chain.get("rendered_relation"):
                chain["rendered_relation"] = llm_chain["rendered_relation"]
            chain["llm_review"] = {
                "recommended_action": llm_chain.get("recommended_action", "keep"),
                "merge_with_chain_ids": llm_chain.get("merge_with_chain_ids", []),
                "split_suggestions": split_groups,
                "meaning_summary": llm_chain.get("meaning_summary", ""),
                "reaction_role_analysis": llm_chain.get("reaction_role_analysis", ""),
                "literature_or_search_basis": llm_chain.get("literature_or_search_basis", ""),
                "confidence": llm_chain.get("confidence", "medium"),
            }
            if llm_chain.get("llm_rationale"):
                chain["llm_review"]["rationale"] = llm_chain["llm_rationale"]
            normalized_chains.append(chain)

        normalized["groups"][group_name] = {
            "chains": normalized_chains,
            "llm_group_summary": llm_group.get("group_summary", ""),
            "llm_suggested_operations": llm_group.get("suggested_operations", []),
        }

    return normalized


def _normalize_decision_items(
    raw_items: list,
    baseline_chain: dict,
    feature_set: set[str],
    default_reason: str,
) -> list[dict]:
    mi_by_feature = {
        item["feature"]: item.get("mi_score", 0.0)
        for item in baseline_chain["retained_features"] + baseline_chain["removed_features"]
    }
    normalized = []
    seen = set()
    for item in raw_items:
        if isinstance(item, str):
            feature = item
            reason = default_reason
        elif isinstance(item, dict):
            feature = item.get("feature")
            reason = item.get("reason") or default_reason
        else:
            continue
        if feature not in feature_set:
            raise ValueError(f"LLM returned unknown feature {feature!r} in {baseline_chain['chain_id']}.")
        if feature in seen:
            continue
        seen.add(feature)
        normalized.append(
            {
                "feature": feature,
                "mi_score": float(mi_by_feature.get(feature, 0.0)),
                "reason": reason,
            }
        )
    return normalized


def _build_llm_messages(payload: dict, settings: dict) -> list[dict]:
    schema_hint = {
        "groups": {
            "<group_name>": {
                "group_summary": "<short summary for this reaction-variable group>",
                "suggested_operations": [
                    {
                        "operation": "merge|split|keep",
                        "chain_ids": ["<input chain_id>"],
                        "reason": "<why this structural adjustment is chemically/statistically justified>",
                    }
                ],
                "chain_reviews": [
                    {
                        "chain_id": "<same chain_id as input>",
                        "recommended_action": "keep|merge|split",
                        "merge_with_chain_ids": ["<optional input chain_id>"],
                        "split_suggestions": [
                            {
                                "name": "<subchain meaning>",
                                "features": ["<feature from this chain>"],
                                "reason": "<why these features form a subchain>",
                            }
                        ],
                        "meaning_summary": "<plain-language meaning of this chain>",
                        "reaction_role_analysis": "<how this descriptor chain may relate to yield/reactivity/selectivity>",
                        "literature_or_search_basis": "<brief literature/search basis or 'not found'>",
                        "confidence": "high|medium|low",
                    }
                ]
            }
        },
        "global_summary": "<short overall summary>",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a reaction descriptor chain reviewer. The deterministic "
                "MI/ridge rules have already produced retained_features and "
                "removed_features; do not replace those feature-selection decisions. "
                "Your job is to validate and interpret each reaction chain after the "
                "default rules: decide whether chains can be merged, whether a chain "
                "should be split into chemically meaningful subchains, and write a "
                "brief meaning analysis for each chain. If your model or endpoint has "
                "web/literature search capability, use it to ground the analysis in "
                "known reaction chemistry and descriptor interpretation. Use only the "
                "provided chain_ids and feature names. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "required_output_schema": schema_hint,
                    "input_chains": payload,
                    "review_requirements": [
                        "Keep baseline retained_features and removed_features unchanged.",
                        "For every input chain, provide one chain_reviews item.",
                        "Use recommended_action='merge' only when two or more chains represent the same chemical effect.",
                        "Use recommended_action='split' when a single chain mixes different interpretable effects.",
                        "Keep literature_or_search_basis concise and avoid fabricating exact citations if none are available.",
                    ],
                    "literature_search_requested": bool(settings.get("literature_search", True)),
                    "user_literature_context": settings.get("literature_context", ""),
                },
                ensure_ascii=False,
            ),
        },
    ]


def _apply_llm_if_enabled(context: dict, baseline: dict) -> dict:
    settings = _llm_config(context)
    if not settings.get("enabled"):
        return baseline
    try:
        if settings.get("provider") != "openai_compatible":
            if settings.get("provider") != "openai_responses":
                raise RuntimeError(
                    "Only provider='openai_responses' and provider='openai_compatible' are currently supported."
                )
        messages = _build_llm_messages(_chain_payload(baseline), settings)
        if settings.get("provider") == "openai_responses":
            llm_output = _call_openai_responses(messages, settings)
        else:
            llm_output = _call_openai_compatible(messages, settings)
        llm_output["provider"] = settings["provider"]
        llm_output["model"] = settings.get("model")
        return _normalize_llm_review(
            llm_output,
            baseline,
            strict=bool(settings.get("strict_validation", True)),
        )
    except Exception as exc:
        if not settings.get("fallback_on_error", True):
            raise
        fallback = dict(baseline)
        fallback["mode"] = "rule_based_llm_protocol_adapter_fallback"
        fallback["llm_error"] = str(exc)
        fallback["note"] = (
            "External LLM chain review failed or was unavailable. "
            "The workflow kept deterministic MI/ridge chain decisions."
        )
        return fallback


def _build_markdown(output: dict) -> list[str]:
    markdown = ["# Chain Decomposition", ""]
    if output.get("mode"):
        markdown.extend([f"Mode: `{output['mode']}`", ""])
    if output.get("llm_error"):
        markdown.extend([f"LLM fallback reason: `{output['llm_error']}`", ""])
    if output.get("llm_global_summary"):
        markdown.extend([f"LLM global summary: {output['llm_global_summary']}", ""])

    for group_name, group_data in output["groups"].items():
        markdown.extend(
            [
                f"## {group_name}",
                "",
                "Chain relationship summary: from each chain, variables are screened using ridge structure and MI with the target.",
                "",
            ]
        )
        if group_data.get("llm_group_summary"):
            markdown.extend([f"LLM group summary: {group_data['llm_group_summary']}", ""])
        for index, chain in enumerate(group_data["chains"], start=1):
            markdown.append(f"{index}. {chain['rendered_relation']}")
            kept = ", ".join(item["feature"] for item in chain["retained_features"])
            removed = ", ".join(item["feature"] for item in chain["removed_features"]) or "none"
            markdown.append("")
            markdown.append(f"   - Importance: `{chain['importance']}`")
            markdown.append(f"   - Keep: {kept}")
            markdown.append(f"   - Remove: {removed}")
            if chain.get("llm_review"):
                review = chain["llm_review"]
                markdown.append(f"   - LLM action: `{review['recommended_action']}`")
                if review.get("merge_with_chain_ids"):
                    markdown.append(f"   - Merge with: {', '.join(review['merge_with_chain_ids'])}")
                if review.get("meaning_summary"):
                    markdown.append(f"   - Meaning: {review['meaning_summary']}")
                if review.get("reaction_role_analysis"):
                    markdown.append(f"   - Reaction role: {review['reaction_role_analysis']}")
                if review.get("literature_or_search_basis"):
                    markdown.append(f"   - Literature/search basis: {review['literature_or_search_basis']}")
            markdown.append("")
        markdown.append("")
    return markdown


def run(context: dict) -> dict:
    state = context["state"]
    run_dir = Path(context["run_dir"])
    output_dir = ensure_dir(run_dir / "step_05_llm_chain_decomposition")

    ridge_path = state["artifacts"].get("ridge_dependency_report")
    mi_path = state["artifacts"].get("mi_scores_by_group")
    if not ridge_path:
        raise ValueError("Ridge dependency report missing. Run ridge_dependency_graph first.")
    if not mi_path:
        raise ValueError("MI scores missing. Run mutual_information first.")

    ridge_report = json.loads(Path(ridge_path).read_text(encoding="utf-8"))
    mi_scores = _mi_map(mi_path)
    baseline = _build_rule_based_output(ridge_report, mi_scores)
    output = _apply_llm_if_enabled(context, baseline)
    markdown = _build_markdown(output)

    json_path = output_dir / "llm_chain_decomposition.json"
    md_path = output_dir / "llm_chain_decomposition.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(markdown), encoding="utf-8")

    return {
        "status": "success",
        "artifacts": {
            "llm_chain_decomposition": str(json_path),
            "llm_chain_decomposition_md": str(md_path),
        },
        "message": "Decomposed ridge structures into chain decisions.",
    }
