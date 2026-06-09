from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from role_atom_canonicalizer import (  # noqa: E402
    canonicalize_role_atoms,
    infer_likely_site_types,
)


def test_infers_likely_site_types() -> None:
    assert infer_likely_site_types("amine", "N") == ["AMINE_N", "AMINE_N_H"]
    assert infer_likely_site_types("ligand", "P") == ["LIGAND_DONOR_ATOM"]
    assert infer_likely_site_types("unknown", "C") == []


def test_canonicalizes_aligned_atoms(tmp_path: Path) -> None:
    input_path = tmp_path / "role_graph_dft_alignment.yaml"
    output_path = tmp_path / "role_atom_canonicalization.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "dataset_key": "AZ",
                        "roles": [
                            {
                                "role": "amine",
                                "representative_component_id": "az:amine:m0000",
                                "representative_source_smiles": "CN",
                                "dft_atom_anchor_map": [
                                    {
                                        "atom_label": "N1",
                                        "element": "N",
                                        "atom_index": 1,
                                        "descriptor_columns": [
                                            "amine_.N1_NMR_shift",
                                            "amine_.N1_electrostatic_charge",
                                        ],
                                        "descriptor_names": [
                                            "NMR_shift",
                                            "electrostatic_charge",
                                        ],
                                        "alignment_status": "exact_element_singleton",
                                        "confidence": "high",
                                        "reason": "Exactly one node.",
                                        "candidate_molecode_nodes": [
                                            {
                                                "node_id": "azaminem0000_N_1",
                                                "component_id": "az:amine:m0000",
                                                "element": "N",
                                                "node_index": 1,
                                                "display_label": "NH2",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = canonicalize_role_atoms(input_path, output_path)

    role = result["datasets"][0]["roles"][0]
    atom = role["canonical_role_atoms"][0]
    assert atom["canonical_atom_id"] == "az:amine:N1"
    assert atom["likely_site_types"] == ["AMINE_N", "AMINE_N_H"]
    assert atom["mapping_status"] == "exact_element_singleton"
    assert role["descriptor_coverage"]["descriptor_column_count"] == 2
    assert role["mapping_status_summary"] == {"exact_element_singleton": 1}
    assert role["canonical_role_graph"]["edges"] == []
    assert role["canonical_role_graph"]["edge_status"] == (
        "pending_markush_role_graph_unification"
    )
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == result
