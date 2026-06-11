from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from literature_evidence_retriever import retrieve_evidence_by_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "generated" / "dft_fixed_role_graphs.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "literature_cross_role_bonds.yaml"
DEFAULT_EVIDENCE_OUTPUT_PATH = ROOT / "generated" / "literature_retrieval_evidence.yaml"

DEFAULT_REACTION_TYPES = {
    "AZ": "Buchwald-Hartwig amination",
    "DY": "Buchwald-Hartwig amination",
    "SU_NO": "Suzuki-Miyaura cross-coupling",
}

ALLOWED_RELATION_TYPES = {
    "FORMING_BOND",
    "BREAKING_BOND",
    "METAL_COORDINATION",
    "ACID_BASE_INTERACTION",
    "ELECTRONIC_EFFECT",
    "STERIC_EFFECT",
    "SOLVENT_EFFECT",
    "ADDITIVE_EFFECT",
    "USER_DEFINED_RELATION",
}

DEFAULT_INITIAL_WEIGHTS = {
    "FORMING_BOND": 0.8,
    "BREAKING_BOND": 0.7,
    "METAL_COORDINATION": 0.5,
    "ACID_BASE_INTERACTION": 0.5,
    "ELECTRONIC_EFFECT": 0.3,
    "STERIC_EFFECT": 0.3,
    "SOLVENT_EFFECT": 0.2,
    "ADDITIVE_EFFECT": 0.2,
    "USER_DEFINED_RELATION": 0.5,
}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def build_literature_cross_role_bonds(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    reaction_types: dict[str, str] | None = None,
    evidence_texts: list[dict[str, str]] | None = None,
    llm_response_text: str | None = None,
    retrieve_evidence: bool = True,
) -> dict[str, Any]:
    fixed_graphs = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    reaction_types = reaction_types or DEFAULT_REACTION_TYPES
    retrieved_evidence = (
        retrieve_evidence_by_dataset(
            reaction_types,
            output_path=DEFAULT_EVIDENCE_OUTPUT_PATH,
        )
        if retrieve_evidence
        else {"datasets": []}
    )
    evidence_index = index_retrieved_evidence(retrieved_evidence)
    evidence_texts = evidence_texts or []
    prompt_evidence_texts = flatten_evidence_index(evidence_index) + evidence_texts
    llm_proposals = (
        parse_llm_bond_response(llm_response_text)
        if llm_response_text
        else None
    )
    result = {
        "artifact_id": "literature_cross_role_bonds_v1",
        "description": (
            "Grounded cross-role relation proposals parsed from literature/LLM "
            "output or deterministic mechanism defaults. Atom identities are "
            "fixed by dft_fixed_role_graphs.yaml. This artifact is not an "
            "adjacency matrix."
        ),
        "llm_prompt": build_literature_bond_prompt(
            fixed_graphs=fixed_graphs,
            reaction_types=reaction_types,
            evidence_texts=prompt_evidence_texts,
        ),
        "datasets": [
            build_dataset_bonds(
                dataset=dataset,
                reaction_type=reaction_types.get(dataset["dataset_key"], "unknown"),
                evidence_texts=dataset_evidence_texts(
                    dataset["dataset_key"],
                    evidence_index,
                    evidence_texts,
                ),
                llm_proposals=llm_proposals,
            )
            for dataset in fixed_graphs.get("datasets", [])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(result, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def build_literature_bond_prompt(
    fixed_graphs: dict[str, Any],
    reaction_types: dict[str, str],
    evidence_texts: list[dict[str, str]],
) -> str:
    allowed_atoms = []
    for dataset in fixed_graphs.get("datasets", []):
        dataset_key = dataset["dataset_key"]
        for role in dataset.get("roles", []):
            for atom in role.get("fixed_atoms", []):
                allowed_atoms.append(
                    {
                        "dataset_key": dataset_key,
                        "reaction_type": reaction_types.get(dataset_key, "unknown"),
                        "role": role["role"],
                        "fixed_atom_id": atom["fixed_atom_id"],
                        "dft_atom_label": atom["dft_atom_label"],
                        "element": atom["element"],
                        "likely_site_types": atom.get("likely_site_types", []),
                    }
                )

    evidence_block = "\n\n".join(
        f"Source: {source.get('source_id', 'unknown')}\n{source.get('text', '')}"
        for source in evidence_texts
    )
    return (
        "You are extracting grounded cross-role reaction relations from "
        "literature. Return only JSON with a top-level 'bonds' array. Each item "
        "must use only allowed dataset_key, role, and DFT atom labels from the "
        "provided atom list. Do not invent atoms. Do not create adjacency "
        "matrices.\n\n"
        "Required JSON item fields: dataset_key, source_role, source_atom_label, "
        "target_role, target_atom_label, relation_type, mechanism_step, "
        "evidence_summary, evidence_sources.\n\n"
        f"Allowed relation_type values: {sorted(ALLOWED_RELATION_TYPES)}\n\n"
        f"Allowed fixed atoms:\n{json.dumps(allowed_atoms, ensure_ascii=False, indent=2)}\n\n"
        f"Literature evidence:\n{evidence_block}"
    )


def index_retrieved_evidence(
    retrieved_evidence: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    return {
        dataset["dataset_key"]: [
            {
                "source_id": evidence["source_id"],
                "text": evidence["evidence_summary"],
                "url": evidence["url"],
                "title": evidence["title"],
            }
            for evidence in dataset.get("evidence", [])
        ]
        for dataset in retrieved_evidence.get("datasets", [])
    }


def flatten_evidence_index(
    evidence_index: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    return [
        evidence
        for dataset_evidence in evidence_index.values()
        for evidence in dataset_evidence
    ]


def dataset_evidence_texts(
    dataset_key: str,
    evidence_index: dict[str, list[dict[str, str]]],
    manual_evidence_texts: list[dict[str, str]],
) -> list[dict[str, str]]:
    return evidence_index.get(dataset_key, []) + manual_evidence_texts


def parse_llm_bond_response(response_text: str) -> list[dict[str, Any]]:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict) or not isinstance(parsed.get("bonds"), list):
        raise ValueError("LLM response must be a JSON object with a 'bonds' array")
    return parsed["bonds"]


def build_dataset_bonds(
    dataset: dict[str, Any],
    reaction_type: str,
    evidence_texts: list[dict[str, str]],
    llm_proposals: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    atom_index = index_fixed_atoms(dataset)
    raw_proposals = (
        [
            proposal
            for proposal in llm_proposals
            if proposal.get("dataset_key") == dataset["dataset_key"]
        ]
        if llm_proposals is not None
        else default_reaction_bond_proposals(dataset, reaction_type)
    )

    grounded = []
    validation_errors = []
    for proposal in raw_proposals:
        grounded_proposal, errors = ground_proposal(
            dataset_key=dataset["dataset_key"],
            proposal=proposal,
            atom_index=atom_index,
            evidence_texts=evidence_texts,
            created_by="llm" if llm_proposals is not None else "reaction_type_default",
        )
        if errors:
            validation_errors.extend(errors)
            continue
        grounded.append(grounded_proposal)

    return {
        "dataset_key": dataset["dataset_key"],
        "reaction_type": reaction_type,
        "cross_role_bonds": grounded,
        "validation_errors": validation_errors,
    }


def default_reaction_bond_proposals(
    dataset: dict[str, Any],
    reaction_type: str,
) -> list[dict[str, Any]]:
    roles = {role["role"] for role in dataset.get("roles", [])}
    proposals = []
    reaction_type_lower = reaction_type.lower()

    if "buchwald" in reaction_type_lower:
        if {"aryl_halide", "amine"} <= roles:
            proposals.append(
                proposal(
                    dataset,
                    "aryl_halide",
                    "C1",
                    "amine",
                    "N1",
                    "FORMING_BOND",
                    "reductive_elimination",
                    "Buchwald-Hartwig amination forms a C-N bond between the aryl electrophile and amine nitrogen.",
                )
            )
        if {"ligand", "aryl_halide"} <= roles:
            proposals.append(
                proposal(
                    dataset,
                    "ligand",
                    "P1",
                    "aryl_halide",
                    "C1",
                    "METAL_COORDINATION",
                    "ligand_controlled_oxidative_addition",
                    "Phosphine ligand environment modulates metal-mediated oxidative addition at the aryl halide center.",
                )
            )
        if {"base", "amine"} <= roles:
            proposals.append(
                proposal(
                    dataset,
                    "base",
                    "N1",
                    "amine",
                    "N1",
                    "ACID_BASE_INTERACTION",
                    "amine_activation",
                    "Base can assist amine deprotonation or neutralize acid generated during coupling.",
                )
            )

    if "suzuki" in reaction_type_lower:
        if {"aryl_halide", "organoboron"} <= roles:
            proposals.append(
                proposal(
                    dataset,
                    "aryl_halide",
                    "C1",
                    "organoboron",
                    "C1",
                    "FORMING_BOND",
                    "reductive_elimination",
                    "Suzuki-Miyaura coupling forms a C-C bond between the aryl electrophile and organoboron carbon.",
                )
            )
            proposals.append(
                proposal(
                    dataset,
                    "organoboron",
                    "B1",
                    "aryl_halide",
                    "C1",
                    "METAL_COORDINATION",
                    "transmetallation",
                    "Organoboron species transfers an organic group through the metal center before C-C bond formation.",
                )
            )
        if {"ligand", "aryl_halide"} <= roles:
            proposals.append(
                proposal(
                    dataset,
                    "ligand",
                    "P1",
                    "aryl_halide",
                    "C1",
                    "METAL_COORDINATION",
                    "oxidative_addition",
                    "Ligand donor atoms coordinate the catalytic metal and affect oxidative addition at the aryl halide.",
                )
            )

    return proposals


def proposal(
    dataset: dict[str, Any],
    source_role: str,
    source_atom_label: str,
    target_role: str,
    target_atom_label: str,
    relation_type: str,
    mechanism_step: str,
    evidence_summary: str,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset["dataset_key"],
        "source_role": source_role,
        "source_atom_label": source_atom_label,
        "target_role": target_role,
        "target_atom_label": target_atom_label,
        "relation_type": relation_type,
        "mechanism_step": mechanism_step,
        "evidence_summary": evidence_summary,
        "evidence_sources": [],
    }


def ground_proposal(
    dataset_key: str,
    proposal: dict[str, Any],
    atom_index: dict[tuple[str, str], dict[str, Any]],
    evidence_texts: list[dict[str, str]],
    created_by: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    errors = validate_proposal(dataset_key, proposal, atom_index)
    if errors:
        return None, errors

    source_atom = atom_index[(proposal["source_role"], proposal["source_atom_label"])]
    target_atom = atom_index[(proposal["target_role"], proposal["target_atom_label"])]
    relation_type = proposal["relation_type"]
    edge = {
        "edge_id": make_cross_role_edge_id(
            source_atom["fixed_atom_id"],
            target_atom["fixed_atom_id"],
            relation_type,
        ),
        "source_fixed_atom_id": source_atom["fixed_atom_id"],
        "target_fixed_atom_id": target_atom["fixed_atom_id"],
        "source_role": proposal["source_role"],
        "target_role": proposal["target_role"],
        "source_dft_atom_label": proposal["source_atom_label"],
        "target_dft_atom_label": proposal["target_atom_label"],
        "relation_type": relation_type,
        "edge_category": relation_type,
        "initial_weight": float(
            proposal.get("initial_weight", DEFAULT_INITIAL_WEIGHTS[relation_type])
        ),
        "mechanism_step": proposal.get("mechanism_step"),
        "evidence_summary": proposal.get("evidence_summary", ""),
        "evidence_sources": proposal.get("evidence_sources") or [
            source["source_id"] for source in evidence_texts if source.get("source_id")
        ],
        "grounding": {
            "source_descriptor_columns": source_atom.get("descriptor_columns", []),
            "target_descriptor_columns": target_atom.get("descriptor_columns", []),
            "source_molecode_anchor": source_atom.get("molecode_anchor"),
            "target_molecode_anchor": target_atom.get("molecode_anchor"),
        },
        "validation_status": "grounded",
        "created_by": created_by,
    }
    return edge, []


def validate_proposal(
    dataset_key: str,
    proposal: dict[str, Any],
    atom_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    required = [
        "source_role",
        "source_atom_label",
        "target_role",
        "target_atom_label",
        "relation_type",
    ]
    for field in required:
        if not proposal.get(field):
            errors.append(error(dataset_key, proposal, "missing_field", field))

    if errors:
        return errors

    if proposal["source_role"] == proposal["target_role"]:
        errors.append(error(dataset_key, proposal, "not_cross_role", "source_role"))

    if proposal["relation_type"] not in ALLOWED_RELATION_TYPES:
        errors.append(error(dataset_key, proposal, "invalid_relation_type", "relation_type"))

    source_key = (proposal["source_role"], proposal["source_atom_label"])
    target_key = (proposal["target_role"], proposal["target_atom_label"])
    if source_key not in atom_index:
        errors.append(error(dataset_key, proposal, "unknown_source_atom", "source_atom_label"))
    if target_key not in atom_index:
        errors.append(error(dataset_key, proposal, "unknown_target_atom", "target_atom_label"))
    return errors


def error(
    dataset_key: str,
    proposal: dict[str, Any],
    error_type: str,
    field: str,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "error_type": error_type,
        "field": field,
        "proposal": proposal,
    }


def index_fixed_atoms(dataset: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (role["role"], atom["dft_atom_label"]): atom
        for role in dataset.get("roles", [])
        for atom in role.get("fixed_atoms", [])
    }


def make_cross_role_edge_id(
    source_atom_id: str,
    target_atom_id: str,
    relation_type: str,
) -> str:
    safe_source = source_atom_id.replace(":", "_")
    safe_target = target_atom_id.replace(":", "_")
    return f"literature_edge:{relation_type}:{safe_source}--{safe_target}"


def main() -> None:
    build_literature_cross_role_bonds()
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
