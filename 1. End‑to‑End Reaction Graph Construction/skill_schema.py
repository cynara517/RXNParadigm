from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Extra, Field, StrictBool, root_validator, validator


class EdgeCategory(str, Enum):
    COVALENT_BOND = "COVALENT_BOND"
    BREAKING_BOND = "BREAKING_BOND"
    FORMING_BOND = "FORMING_BOND"
    METAL_COORDINATION = "METAL_COORDINATION"
    ACID_BASE_INTERACTION = "ACID_BASE_INTERACTION"
    ELECTRONIC_EFFECT = "ELECTRONIC_EFFECT"
    STERIC_EFFECT = "STERIC_EFFECT"
    SOLVENT_EFFECT = "SOLVENT_EFFECT"
    ADDITIVE_EFFECT = "ADDITIVE_EFFECT"
    USER_DEFINED_RELATION = "USER_DEFINED_RELATION"


class SkillBaseModel(BaseModel):
    class Config:
        extra = Extra.allow
        anystr_strip_whitespace = True


def _validate_non_empty_strings(values: List[str], field_name: str) -> List[str]:
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain only non-empty strings")
    return values


class ComponentClass(SkillBaseModel):
    class_name: str
    description: Optional[str] = None
    generic_template: Optional[str] = None
    site_types: List[str] = Field(..., min_items=1)

    @validator("class_name")
    def class_name_must_be_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("class_name must be a non-empty string")
        return value

    @validator("site_types")
    def site_types_must_be_non_empty_strings(cls, values: List[str]) -> List[str]:
        return _validate_non_empty_strings(values, "site_types")


class SiteDefinition(SkillBaseModel):
    site_type: str
    component_class: str
    description: Optional[str] = None
    matching_strategy: Optional[str] = None
    pattern: Optional[str] = None
    pattern_hint: Optional[str] = None
    editable: StrictBool

    @validator("site_type", "component_class")
    def required_strings_must_be_non_empty(cls, value: str, field: Any) -> str:
        if not value:
            raise ValueError(f"{field.name} must be a non-empty string")
        return value


class RelationRule(SkillBaseModel):
    rule_id: str
    relation_type: EdgeCategory
    source_site_type: str
    target_site_type: str
    mechanism_step: Optional[str] = None
    description: Optional[str] = None
    initial_weight: float = 0.5
    enabled: StrictBool
    editable: StrictBool

    @validator("rule_id", "source_site_type", "target_site_type")
    def required_strings_must_be_non_empty(cls, value: str, field: Any) -> str:
        if not value:
            raise ValueError(f"{field.name} must be a non-empty string")
        return value

    @validator("initial_weight")
    def initial_weight_must_be_between_zero_and_one(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("initial_weight must be between 0.0 and 1.0")
        return value


class Skill(SkillBaseModel):
    skill_id: str
    version: str
    status: str
    reaction_family: str
    inherits_from: Optional[List[str]] = None
    component_classes: List[ComponentClass] = Field(..., min_items=1)
    site_definitions: List[SiteDefinition] = Field(..., min_items=1)
    relation_rules: List[RelationRule] = Field(..., min_items=1)

    @validator("version", pre=True)
    def version_may_be_numeric_in_yaml(cls, value: Any) -> str:
        return str(value)

    @validator("skill_id", "version", "status", "reaction_family")
    def top_level_strings_must_be_non_empty(cls, value: str, field: Any) -> str:
        if not value:
            raise ValueError(f"{field.name} must be a non-empty string")
        return value

    @validator("inherits_from")
    def inherits_from_must_be_non_empty_strings(
        cls, values: Optional[List[str]]
    ) -> Optional[List[str]]:
        if values is None:
            return values
        if not values:
            raise ValueError("inherits_from must be a non-empty list when present")
        return _validate_non_empty_strings(values, "inherits_from")

    @root_validator
    def validate_cross_references(cls, values: dict[str, Any]) -> dict[str, Any]:
        component_classes = values.get("component_classes") or []
        site_definitions = values.get("site_definitions") or []
        relation_rules = values.get("relation_rules") or []

        class_names = [component.class_name for component in component_classes]
        duplicate_class_names = _duplicates(class_names)
        if duplicate_class_names:
            raise ValueError(
                "component_classes must have unique class_name values: "
                + ", ".join(duplicate_class_names)
            )

        known_classes = set(class_names)
        for site in site_definitions:
            if site.component_class not in known_classes:
                raise ValueError(
                    f"site_definition {site.site_type!r} references unknown "
                    f"component_class {site.component_class!r}"
                )

        site_types = [site.site_type for site in site_definitions]
        duplicate_site_types = _duplicates(site_types)
        if duplicate_site_types:
            raise ValueError(
                "site_definitions must have unique site_type values: "
                + ", ".join(duplicate_site_types)
            )

        rule_ids = [rule.rule_id for rule in relation_rules]
        duplicate_rule_ids = _duplicates(rule_ids)
        if duplicate_rule_ids:
            raise ValueError(
                "relation_rules must have unique rule_id values: "
                + ", ".join(duplicate_rule_ids)
            )

        known_site_types = set(site_types)
        allows_inherited_site_types = bool(values.get("inherits_from"))
        for rule in relation_rules:
            if (
                rule.source_site_type not in known_site_types
                and not allows_inherited_site_types
            ):
                raise ValueError(
                    f"relation_rule {rule.rule_id!r} references unknown "
                    f"source_site_type {rule.source_site_type!r}"
                )
            if (
                rule.target_site_type not in known_site_types
                and not allows_inherited_site_types
            ):
                raise ValueError(
                    f"relation_rule {rule.rule_id!r} references unknown "
                    f"target_site_type {rule.target_site_type!r}"
                )

        return values


def _duplicates(values: List[str]) -> List[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
