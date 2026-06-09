# AGENTS.md

# Project Name

Reaction Graph Constructor

---

# Project Goal

Build a MoleCode-grounded, knowledge-augmented reaction graph construction system.

The system converts reaction equation SMILES and user-defined reaction-family knowledge into structured graph representations.

The generated graph output will be consumed by external downstream machine-learning code.

This repository only performs reaction graph construction.

This repository does NOT perform model training.

---

# Core Concept

This project has two major workflows:

1. Skill Construction Workflow
2. Graph Construction Workflow

A Skill is a user-editable reaction-family knowledge object.

A Skill describes:

* what molecular component classes exist in a reaction family
* what generic or Makush-like SMILES templates represent these classes
* what chemically meaningful atom sites exist
* what site-site relations may exist
* what literature or mechanistic evidence supports those relations
* what initial edge weights should be assigned

The Skill is later used to construct graph representations from concrete reaction SMILES.

---

# Design Philosophy

This system is a graph compiler.

It is NOT a chemical question-answering agent.

It is NOT a yield prediction model.

It is NOT a free-form autonomous agent.

The system should be deterministic wherever possible.

LLMs are used only for controlled evidence retrieval, mechanism summarization, and Skill drafting.

LLMs must never directly generate atom-level adjacency matrices.

---

# Out of Scope

Do NOT implement:

* yield prediction models
* GNN architectures
* training loops
* loss functions
* optimizer logic
* hyperparameter tuning
* benchmark evaluation
* checkpointing
* model inference services

The downstream training code exists outside this repository.

---

# Workflow A: Skill Construction Workflow

## Purpose

Create a user-editable Skill from:

* dataset description
* reaction-family text description
* reaction equation SMILES
* Makush-like or R-group SMILES templates
* optional literature references
* optional user notes

This workflow is used when a user defines a new reaction family or edits an existing one.

---

## Skill Construction Input

Example:

```json
{
  "skill_id": "buchwald_hartwig_user_v1",
  "reaction_family_hint": "Buchwald-Hartwig amination",
  "dataset_name": "AstraZeneca ELN Buchwald-Hartwig",
  "dataset_description": "It covers a much wider chemical space, with 340 aryl halides, 260 amines, 24 ligands, 15 bases and 15 solvents.",
  "reaction_smiles_templates": [
    "..."
  ],
  "user_notes": [
    "Use the aryl halide ipso carbon and leaving group as oxidative addition sites.",
    "Use amine nitrogen as the nucleophilic site."
  ]
}
```

The input may include:

* full reaction SMILES
* reaction equation SMILES
* generic reaction SMILES
* Makush-like SMILES
* R-group SMILES
* text descriptions of component classes

---

## Skill Construction Nodes

Use LangGraph to implement this workflow.

Required nodes:

1. SkillInputValidationNode
2. ReactionFamilyUnderstandingNode
3. ComponentClassExtractionNode
4. MoleCodeTemplateParserNode
5. LiteratureRetrievalNode
6. SiteDefinitionDraftNode
7. RelationRuleDraftNode
8. SkillValidationNode
9. SkillFormatterNode

---

## Skill Construction Output

The output is a Skill Draft.

Example structure:

```json
{
  "skill_id": "buchwald_hartwig_user_v1",
  "reaction_family": "Buchwald-Hartwig amination",
  "status": "draft",
  "component_classes": [],
  "generic_templates": [],
  "site_definitions": [],
  "relation_rules": [],
  "evidence_sources": [],
  "editable_fields": []
}
```

The output Skill must be JSON-serializable.

The Skill must be user-editable.

---

# Skill Object

A Skill is not hard-coded code.

A Skill is a structured knowledge object.

It may be saved as JSON or YAML.

A Skill must contain:

* skill_id
* reaction_family
* version
* status
* component_classes
* generic_templates
* site_definitions
* relation_rules
* evidence_sources
* editable_fields
* metadata

---

## Component Classes

Component classes represent categories of molecular species in a reaction family.

Examples:

* aryl_halide
* amine
* ligand
* base
* solvent
* boronic_acid
* catalyst
* additive
* product

Each component class may contain:

```json
{
  "class_name": "aryl_halide",
  "description": "Aryl or heteroaryl halide electrophile",
  "generic_smiles_template": "...",
  "count_in_dataset": 340,
  "site_types": ["ARYL_C_IPSO", "LEAVING_GROUP_X"]
}
```

---

## Site Definitions

Site definitions describe chemically meaningful atom-level sites.

Each site definition must include:

```json
{
  "site_type": "ARYL_C_IPSO",
  "component_class": "aryl_halide",
  "description": "Aryl carbon bonded to the leaving group.",
  "matching_strategy": "SMARTS_OR_MOLECODE_PATTERN",
  "pattern": "...",
  "editable": true
}
```

Site definitions must not depend on LLM output during graph construction.

They must be usable by deterministic matching logic.

---

## Relation Rules

Relation rules describe possible site-site relations.

Each relation rule must include:

```json
{
  "relation_type": "METAL_COORDINATION",
  "source_site_type": "METAL_CENTER",
  "target_site_type": "ARYL_C_IPSO",
  "description": "Metal center interacts with the aryl halide reaction center during oxidative addition.",
  "initial_weight": 0.5,
  "evidence_summary": "...",
  "evidence_sources": [],
  "enabled": true,
  "editable": true
}
```

Relation rules are not atom-level edges.

They are site-level rules.

Atom-level edges are constructed later by the RelationCompilerNode.

---

# Workflow B: Graph Construction Workflow

## Purpose

Construct a reaction graph from a concrete reaction equation SMILES using a selected Skill.

This workflow should be deterministic whenever possible.

LLM calls should be minimized in this workflow.

---

## Graph Construction Input

Example:

```json
{
  "reaction_id": "rxn_001",
  "reaction_smiles": "...",
  "skill_id": "buchwald_hartwig_user_v1"
}
```

The user provides:

* reaction_id
* reaction_smiles
* skill_id

The user does NOT provide:

* graph1
* graph2
* graph3
* graph4
* atom IDs
* adjacency matrix

These must be generated by the system.

---

## Graph Construction Nodes

Use LangGraph to implement this workflow.

Required nodes:

1. GraphInputValidationNode
2. SkillLoadingNode
3. MoleCodeParserNode
4. RDKitBondEncoderNode
5. ComponentRoleAssignmentNode
6. SiteMatchingNode
7. RelationCompilerNode
8. AdjacencyMatrixNode
9. GraphOutputFormatterNode

---

# MoleCode Requirements

The first parsing operation in Graph Construction Workflow must call MoleCode.

MoleCode is responsible for:

* parsing reaction SMILES
* decomposing reaction SMILES into molecular component graphs
* creating atom-level graph representations
* generating persistent atom IDs
* preserving R-group or Makush-like information where possible

Atom IDs must be persistent and stable.

Recommended atom ID format:

```text
graph1:a0
graph1:a1
graph2:a0
graph2:a1
```

MoleCode-derived component graphs are internal objects.

They are not user inputs.

---

# RDKit Requirements

RDKit is responsible for:

* covalent bond extraction
* bond order extraction
* atomic descriptors
* SMARTS-based site matching
* fallback molecule parsing when possible

RDKit-derived covalent bonds are deterministic.

They must never depend on LLM output.

---

# Component Role Assignment

After MoleCode parsing, the system should assign component roles.

Roles may be assigned using:

* Skill component class definitions
* reaction side information
* SMARTS or MoleCode patterns
* user-provided hints
* deterministic rules
* optional LLM assistance during Skill Construction

Graph Construction should not rely on free-form LLM reasoning for role assignment unless explicitly configured.

---

# Site Matching

SiteMatchingNode maps Skill site definitions onto MoleCode atom IDs.

Input:

* MoleCode component graphs
* RDKit atom/bond information
* Skill site definitions

Output:

* site records
* site_id
* site_type
* component_id
* atom_ids
* matching evidence

Example:

```json
{
  "site_id": "site_001",
  "site_type": "AMINE_N",
  "component_id": "graph2",
  "atom_ids": ["graph2:a5"],
  "matched_by": "SMARTS"
}
```

---

# Relation Compiler

Only RelationCompilerNode may create atom-level candidate edges.

RelationCompilerNode uses:

* validated Skill relation rules
* matched site records
* MoleCode atom IDs
* RDKit covalent bonds

LLMs must never directly create atom-level edges.

RelationCompilerNode converts:

```text
source_site_type + target_site_type
```

into:

```text
source_atom_id + target_atom_id
```

---

# Edge Types

Allowed edge categories:

* COVALENT_BOND
* BREAKING_BOND
* FORMING_BOND
* METAL_COORDINATION
* ACID_BASE_INTERACTION
* ELECTRONIC_EFFECT
* STERIC_EFFECT
* SOLVENT_EFFECT
* ADDITIVE_EFFECT
* USER_DEFINED_RELATION

New edge categories may be introduced only through an explicit Skill definition.

---

# Edge Weights

RDKit covalent bonds:

```text
weight = bond_order
```

Skill-derived candidate relations:

```text
default weight = relation_rule.initial_weight
```

If no value is provided:

```text
default weight = 0.5
```

Weights are initialization values only.

They may be modified later by downstream learning systems.

---

# Adjacency Matrix

The system must output:

* edge_index
* edge_attr
* adjacency_matrix

Adjacency matrices may be constructed only from:

* RDKit covalent bonds
* RelationCompilerNode output

LLMs must never directly output adjacency matrices.

---

# Final Graph Output

The final graph output must contain:

* reaction_id
* skill_id
* atom list
* component graph list
* covalent bond list
* matched site list
* relation rule applications
* edge list
* adjacency matrix
* evidence summaries
* errors
* metadata

All output must be JSON-serializable.

---

# Error Handling

The system must return structured errors.

Do not silently fail.

Common errors:

* MoleCode parsing failure
* RDKit parsing failure
* unsupported R-group expression
* missing Skill
* invalid Skill schema
* site matching failure
* relation rule cannot be grounded
* adjacency construction failure

---

# Testing Requirements

Every workflow stage must include unit tests.

Use:

```bash
pytest
```

No phase is complete unless all tests pass.

---

# Development Phases

## Phase 1

Implement core schemas:

* SkillConstructionInput
* SkillDraft
* GraphConstructionInput
* MoleCodeGraph
* AtomRecord
* BondRecord
* SiteRecord
* RelationRule
* EdgeRecord
* GraphOutput

## Phase 2

Implement Graph Construction deterministic backbone:

* MoleCodeParserNode stub or adapter
* RDKitBondEncoderNode
* atom list generation
* covalent bond adjacency

## Phase 3

Implement Skill object loading and validation.

## Phase 4

Implement SiteMatchingNode.

## Phase 5

Implement RelationCompilerNode.

## Phase 6

Implement Skill Construction Workflow skeleton.

## Phase 7

Implement LiteratureRetrievalNode and RelationRuleDraftNode.

## Phase 8

Implement full structured graph output.

---

# Non-Negotiable Rules

1. Do not implement yield prediction training.
2. Do not implement GNN models.
3. Do not let LLMs generate adjacency matrices.
4. Do not let LLMs generate atom IDs.
5. Do not let LLMs overwrite RDKit covalent bonds.
6. Do not hard-code one reaction family as the only supported family.
7. Skills must be user-editable.
8. Graph Construction must use a selected Skill.
9. MoleCode parsing happens before graph construction.
10. Relation rules are site-level, not atom-level.
11. Atom-level edges are created only by RelationCompilerNode.
12. All outputs must be JSON-serializable.
