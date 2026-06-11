from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai_compatible"
    model: str = ""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    timeout_seconds: int = 60

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "LLMConfig | None":
        if payload is None:
            return None
        return cls(
            provider=str(payload.get("provider", "openai_compatible")),
            model=str(payload.get("model", "")),
            api_key=str(payload.get("api_key", "")),
            base_url=str(payload.get("base_url", "https://api.openai.com/v1")),
            temperature=float(payload.get("temperature", 0.0)),
            timeout_seconds=int(payload.get("timeout_seconds", 60)),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "api_key_provided": bool(self.api_key),
        }


class LLMRationaleClient:
    def __init__(self, config: LLMConfig | dict[str, Any] | None) -> None:
        if isinstance(config, LLMConfig):
            self.config = config
        else:
            self.config = LLMConfig.from_mapping(config)

    @property
    def enabled(self) -> bool:
        return bool(self.config and self.config.api_key and self.config.model)

    def explain_cross_role_relation(
        self,
        relation: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = deterministic_relation_rationale(relation, evidence)
        if not self.enabled:
            fallback["llm_status"] = "not_called"
            fallback["llm_reason"] = "No user LLM API config was provided."
            return fallback
        if self.config and self.config.provider != "openai_compatible":
            fallback["llm_status"] = "not_called"
            fallback["llm_reason"] = f"Unsupported provider: {self.config.provider}"
            return fallback

        prompt = build_relation_prompt(relation, evidence)
        try:
            content = self._chat_completion(prompt)
            parsed = parse_json_object(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response was not a JSON object")
            return {
                "llm_status": "called",
                "llm_provider": self.config.provider if self.config else "",
                "llm_model": self.config.model if self.config else "",
                "search_queries": parsed.get("search_queries", fallback["search_queries"]),
                "evidence_summary": parsed.get("evidence_summary", fallback["evidence_summary"]),
                "mechanistic_rationale": parsed.get(
                    "mechanistic_rationale",
                    fallback["mechanistic_rationale"],
                ),
                "confidence": parsed.get("confidence", fallback["confidence"]),
                "limitations": parsed.get("limitations", fallback["limitations"]),
            }
        except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
            fallback["llm_status"] = "failed"
            fallback["llm_reason"] = str(exc)
            return fallback

    def _chat_completion(self, prompt: str) -> str:
        if not self.config:
            raise ValueError("Missing LLM config")
        base_url = self.config.base_url.rstrip("/")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You explain evidence for reaction graph cross-role "
                                "relations. You never invent atom IDs, bonds, or "
                                "adjacency matrices. Return only JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]


def build_relation_prompt(
    relation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    return (
        "Explain why the following already-constructed cross-role relation is "
        "chemically plausible. Do not create new edges. Do not change atom IDs. "
        "Use only the provided evidence records. Return JSON with fields: "
        "search_queries, evidence_summary, mechanistic_rationale, confidence, "
        "limitations.\n\n"
        f"Relation:\n{json.dumps(relation, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def deterministic_relation_rationale(
    relation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    relation_type = relation.get("relation_type", "")
    source = relation.get("source_node_id", "")
    target = relation.get("target_node_id", "")
    summaries = [item.get("evidence_summary", "") for item in evidence if item.get("evidence_summary")]
    queries = sorted(
        {
            item.get("retrieval_query", "")
            for item in evidence
            if item.get("retrieval_query", "")
        }
    )
    if relation_type == "FORMING_BOND":
        mechanistic = (
            f"{source} and {target} are represented as the two atom-level sites for "
            "a product-forming cross-coupling relation. The edge is kept as a "
            "candidate relation rather than an RDKit covalent bond."
        )
    elif relation_type == "METAL_COORDINATION":
        mechanistic = (
            f"{source} and {target} are represented as ligand/metal-associated "
            "coordination or catalyst-cycle interaction sites. The edge encodes "
            "mechanistic proximity, not a fixed covalent bond."
        )
    else:
        mechanistic = (
            f"{source} and {target} are connected by a site-level {relation_type} "
            "rule compiled into a candidate graph edge."
        )
    return {
        "llm_status": "deterministic_fallback",
        "search_queries": queries,
        "evidence_summary": " ".join(summaries) if summaries else "No retrieved evidence summary found.",
        "mechanistic_rationale": mechanistic,
        "confidence": "medium" if evidence else "low",
        "limitations": [
            "This rationale explains an already-constructed candidate relation.",
            "It does not modify atom IDs, RDKit covalent bonds, or adjacency matrices.",
        ],
    }


def parse_json_object(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)
