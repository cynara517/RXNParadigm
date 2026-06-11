from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT / "generated" / "literature_retrieval_evidence.yaml"


CURATED_LITERATURE = [
    {
        "source_id": "buchwald_hartwig_acs_chemrev_2025",
        "reaction_type_keywords": ["buchwald", "hartwig", "amination"],
        "title": "Metal-N-Heterocyclic Carbene Complexes in Buchwald-Hartwig Amination Reactions",
        "url": "https://pubs.acs.org/doi/10.1021/acs.chemrev.5c00088",
        "retrieval_query": "Buchwald Hartwig amination mechanism aryl halide amine oxidative addition deprotonation reductive elimination",
        "evidence_summary": (
            "Review evidence describes active Pd/Ni complexes promoting oxidative "
            "addition of aryl halides, followed by amine coordination, base-assisted "
            "deprotonation to a metal-amido species, and product-forming reductive elimination."
        ),
        "supports": [
            "aryl_halide:C1 -> amine:N1 FORMING_BOND",
            "ligand:P1 -> aryl_halide:C1 METAL_COORDINATION",
            "base:* -> amine:N1 ACID_BASE_INTERACTION",
        ],
    },
    {
        "source_id": "buchwald_hartwig_libretexts",
        "reaction_type_keywords": ["buchwald", "hartwig", "amination"],
        "title": "Buchwald-Hartwig Amination",
        "url": "https://chem.libretexts.org/Bookshelves/Inorganic_Chemistry/Supplemental_Modules_and_Websites_%28Inorganic_Chemistry%29/Catalysis/Catalyst_Examples/Buchwald-Hartwig_Amination",
        "retrieval_query": "Buchwald Hartwig amination aryl halides amines C N bond palladium",
        "evidence_summary": (
            "Teaching reference identifies Buchwald-Hartwig amination as palladium-catalyzed "
            "cross-coupling of amines and aryl halides that forms C-N bonds."
        ),
        "supports": [
            "aryl_halide:C1 -> amine:N1 FORMING_BOND",
        ],
    },
    {
        "source_id": "buchwald_hartwig_nature_index_topic",
        "reaction_type_keywords": ["buchwald", "hartwig", "amination"],
        "title": "Palladium-Catalyzed C-N Cross-Coupling Reactions",
        "url": "https://www.nature.com/nature-index/topics/l4/palladium-catalyzed-c-n-cross-coupling-reactions",
        "retrieval_query": "palladium catalyzed C N cross coupling oxidative addition amine coordination deprotonation reductive elimination",
        "evidence_summary": (
            "Topic summary describes oxidative addition into aryl-halide bonds, "
            "ligand-assisted amine coordination, deprotonation, and reductive elimination "
            "as central steps in palladium-catalyzed C-N cross-coupling."
        ),
        "supports": [
            "aryl_halide:C1 -> amine:N1 FORMING_BOND",
            "ligand:P1 -> aryl_halide:C1 METAL_COORDINATION",
            "base:* -> amine:N1 ACID_BASE_INTERACTION",
        ],
    },
    {
        "source_id": "suzuki_miyaura_intramolecular_review_pmc_2024",
        "reaction_type_keywords": ["suzuki", "miyaura", "cross-coupling"],
        "title": "Cyclization by Intramolecular Suzuki-Miyaura Cross-Coupling-A Review",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11711311/",
        "retrieval_query": "Suzuki Miyaura mechanism oxidative addition transmetalation reductive elimination organoboron aryl halide base",
        "evidence_summary": (
            "Review evidence describes Suzuki-Miyaura coupling between an organic "
            "halide or triflate and an organoboron derivative through oxidative addition, "
            "base-mediated transmetalation, and reductive elimination to form C-C bonds."
        ),
        "supports": [
            "aryl_halide:C1 -> organoboron:C1 FORMING_BOND",
            "organoboron:B1 -> aryl_halide:C1 METAL_COORDINATION",
        ],
    },
    {
        "source_id": "suzuki_miyaura_recent_developments_pmc_2014",
        "reaction_type_keywords": ["suzuki", "miyaura", "cross-coupling"],
        "title": "Recent Developments in the Suzuki-Miyaura Reaction: 2010-2014",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6272665/",
        "retrieval_query": "Suzuki Miyaura organoboron organic halide base ligand oxidative addition reductive elimination",
        "evidence_summary": (
            "Review evidence frames Suzuki-Miyaura reaction as coupling an organoboron "
            "reagent with an organic halide or pseudohalide in the presence of metal catalyst "
            "and base, following oxidative addition, transmetalation, and reductive elimination."
        ),
        "supports": [
            "aryl_halide:C1 -> organoboron:C1 FORMING_BOND",
            "ligand:P1 -> aryl_halide:C1 METAL_COORDINATION",
        ],
    },
    {
        "source_id": "suzuki_transmetalation_pubmed_2013",
        "reaction_type_keywords": ["suzuki", "miyaura", "transmetalation"],
        "title": "Transmetalation in the Suzuki-Miyaura coupling: the fork in the trail",
        "url": "https://pubmed.ncbi.nlm.nih.gov/23780626/",
        "retrieval_query": "Transmetalation in Suzuki Miyaura coupling oxidative addition transmetalation reductive elimination",
        "evidence_summary": (
            "Mechanistic review abstract emphasizes the generic oxidative-addition, "
            "transmetalation, reductive-elimination sequence and the distinct role of "
            "transmetalation in Suzuki-Miyaura chemistry."
        ),
        "supports": [
            "organoboron:B1 -> aryl_halide:C1 METAL_COORDINATION",
        ],
    },
]


def retrieve_literature_evidence(
    reaction_type: str,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    reaction_type_lower = reaction_type.lower()
    records = [
        record
        for record in CURATED_LITERATURE
        if any(keyword in reaction_type_lower for keyword in record["reaction_type_keywords"])
    ]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(
                {
                    "artifact_id": "literature_retrieval_evidence_v1",
                    "reaction_type": reaction_type,
                    "evidence": records,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return records


def retrieve_evidence_by_dataset(
    reaction_types: dict[str, str],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    result = {
        "artifact_id": "literature_retrieval_evidence_v1",
        "datasets": [
            {
                "dataset_key": dataset_key,
                "reaction_type": reaction_type,
                "evidence": retrieve_literature_evidence(reaction_type),
            }
            for dataset_key, reaction_type in sorted(reaction_types.items())
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    from literature_cross_role_bond_tool import DEFAULT_REACTION_TYPES

    retrieve_evidence_by_dataset(DEFAULT_REACTION_TYPES)
    print(f"Wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
