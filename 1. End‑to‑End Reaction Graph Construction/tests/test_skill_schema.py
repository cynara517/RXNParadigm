from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skill_loader import SkillLoaderError, load_skill
from skill_schema import Skill


def minimal_skill() -> dict:
    return {
        "skill_id": "test_skill_v1",
        "version": "1.0",
        "status": "draft",
        "reaction_family": "Test reaction",
        "component_classes": [
            {
                "class_name": "aryl_halide",
                "description": "Electrophile",
                "generic_template": "Ar-X",
                "site_types": ["ARYL_C_IPSO", "LEAVING_GROUP_X"],
            },
            {
                "class_name": "amine",
                "description": "Nucleophile",
                "generic_template": "R-NH2",
                "site_types": ["AMINE_N"],
            },
        ],
        "site_definitions": [
            {
                "site_type": "ARYL_C_IPSO",
                "component_class": "aryl_halide",
                "description": "Ipso carbon",
                "matching_strategy": "SMARTS_OR_MOLECODE_PATTERN",
                "pattern_hint": "aryl_carbon_bonded_to_halide",
                "editable": True,
            },
            {
                "site_type": "LEAVING_GROUP_X",
                "component_class": "aryl_halide",
                "description": "Leaving group",
                "matching_strategy": "SMARTS_OR_MOLECODE_PATTERN",
                "pattern_hint": "halide",
                "editable": True,
            },
            {
                "site_type": "AMINE_N",
                "component_class": "amine",
                "description": "Amine nitrogen",
                "matching_strategy": "SMARTS_OR_MOLECODE_PATTERN",
                "pattern_hint": "amine_nitrogen",
                "editable": True,
            },
        ],
        "relation_rules": [
            {
                "rule_id": "rule_001",
                "relation_type": "FORMING_BOND",
                "source_site_type": "ARYL_C_IPSO",
                "target_site_type": "AMINE_N",
                "description": "C-N bond formation",
                "initial_weight": 0.5,
                "enabled": True,
                "editable": True,
            }
        ],
    }


@pytest.mark.parametrize(
    "filename",
    ["azSKILL.yaml", "suSKILL.yaml", "dySKILL.yaml"],
)
def test_loads_existing_skill_files(filename: str) -> None:
    skill = load_skill(ROOT / "skills" / filename)

    assert skill.skill_id
    assert skill.component_classes
    assert skill.site_definitions
    assert skill.relation_rules


def test_loader_rejects_missing_file() -> None:
    with pytest.raises(SkillLoaderError, match="does not exist"):
        load_skill(ROOT / "missingSKILL.yaml")


def test_duplicate_component_classes_are_rejected() -> None:
    data = minimal_skill()
    data["component_classes"][1]["class_name"] = "aryl_halide"

    with pytest.raises(ValueError, match="unique class_name"):
        Skill.parse_obj(data)


def test_site_definition_unknown_component_class_is_rejected() -> None:
    data = minimal_skill()
    data["site_definitions"][0]["component_class"] = "unknown_component"

    with pytest.raises(ValueError, match="unknown component_class"):
        Skill.parse_obj(data)


def test_relation_rule_unknown_source_site_type_is_rejected() -> None:
    data = minimal_skill()
    data["relation_rules"][0]["source_site_type"] = "UNKNOWN_SITE"

    with pytest.raises(ValueError, match="unknown source_site_type"):
        Skill.parse_obj(data)


def test_relation_rule_unknown_target_site_type_is_rejected() -> None:
    data = minimal_skill()
    data["relation_rules"][0]["target_site_type"] = "UNKNOWN_SITE"

    with pytest.raises(ValueError, match="unknown target_site_type"):
        Skill.parse_obj(data)


@pytest.mark.parametrize("initial_weight", [-0.1, 1.1])
def test_invalid_initial_weight_is_rejected(initial_weight: float) -> None:
    data = minimal_skill()
    data["relation_rules"][0]["initial_weight"] = initial_weight

    with pytest.raises(ValueError, match="initial_weight"):
        Skill.parse_obj(data)


def test_missing_initial_weight_defaults_to_half() -> None:
    data = minimal_skill()
    del data["relation_rules"][0]["initial_weight"]

    skill = Skill.parse_obj(data)

    assert skill.relation_rules[0].initial_weight == 0.5


@pytest.mark.parametrize("field_name", ["enabled", "editable"])
def test_relation_rule_flags_are_required(field_name: str) -> None:
    data = minimal_skill()
    del data["relation_rules"][0][field_name]

    with pytest.raises(ValueError, match=field_name):
        Skill.parse_obj(data)


@pytest.mark.parametrize("field_name", ["enabled", "editable"])
def test_relation_rule_flags_must_be_booleans(field_name: str) -> None:
    data = minimal_skill()
    data["relation_rules"][0][field_name] = "true"

    with pytest.raises(ValueError, match=field_name):
        Skill.parse_obj(data)


def test_site_definition_editable_is_required() -> None:
    data = minimal_skill()
    del data["site_definitions"][0]["editable"]

    with pytest.raises(ValueError, match="editable"):
        Skill.parse_obj(data)


def test_site_definition_editable_must_be_boolean() -> None:
    data = minimal_skill()
    data["site_definitions"][0]["editable"] = "true"

    with pytest.raises(ValueError, match="editable"):
        Skill.parse_obj(data)


def test_valid_inherits_from_is_accepted() -> None:
    data = minimal_skill()
    data["inherits_from"] = ["parent_skill_v1"]

    skill = Skill.parse_obj(data)

    assert skill.inherits_from == ["parent_skill_v1"]


@pytest.mark.parametrize("inherits_from", [[], [""], ["   "]])
def test_invalid_inherits_from_is_rejected(inherits_from: list[str]) -> None:
    data = minimal_skill()
    data["inherits_from"] = inherits_from

    with pytest.raises(ValueError, match="inherits_from"):
        Skill.parse_obj(data)


def test_extra_metadata_fields_are_preserved() -> None:
    data = minimal_skill()
    data["dataset_context"] = {"dataset_key": "TEST"}
    data["relation_rules"][0]["evidence_policy"] = "mechanism_default"

    skill = Skill.parse_obj(data)

    assert skill.dataset_context == {"dataset_key": "TEST"}
    assert skill.relation_rules[0].evidence_policy == "mechanism_default"
