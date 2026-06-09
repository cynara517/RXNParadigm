# Reaction Graph Constructor

Reaction Graph Constructor builds structured reaction graph artifacts from
reaction-family knowledge, role-level chemistry definitions, DFT/RDKit-derived
atom descriptors, and curated relation evidence.

This repository is a graph construction tool. It does not implement yield
prediction models, GNN architectures, model training loops, checkpointing, or
inference services.

## What Is Included

- Core Python modules for reaction graph construction and export.
- `reaction_graph_agent/`, a deterministic orchestration layer for graph CSVs,
  sample graph datasets, and quality reports.
- User-editable Skill YAML files in `skills/`.
- Site ontology definitions in `ontology/`.
- Source datasets in `datasets/`.
- Precomputed example outputs in `generated/`.
- A bundled copy of MoleCode in `MoleCode-main/`.
- Unit tests in `tests/`.

## Repository Layout

```text
.
├── reaction_graph_agent/        # Controlled graph-construction agent API
├── skills/                      # User-editable reaction-family Skill files
├── ontology/                    # Site ontology definitions
├── datasets/                    # Source datasets
├── generated/                   # Precomputed example outputs
├── MoleCode-main/               # Bundled MoleCode source
├── tests/                       # Pytest unit tests
├── *_builder.py                 # Graph/data construction stages
├── *_exporter.py                # Export utilities
├── skill_schema.py              # Skill schema definitions
└── skill_loader.py              # Skill loading and validation
```

## Installation

Create a Python environment, then install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e MoleCode-main
```

If RDKit installation fails through `pip`, install it with conda first:

```bash
conda create -n reaction-graph python=3.11 rdkit -c conda-forge
conda activate reaction-graph
pip install -r requirements.txt
pip install -e MoleCode-main
```

## Quick Check

Run the test suite:

```bash
pytest
```

Run the deterministic graph-construction pipeline from Python:

```python
from reaction_graph_agent import ReactionGraphAgent

agent = ReactionGraphAgent()
result = agent.construct()
print(result["artifact_id"])
```

## Important Design Rules

- LLMs must not generate atom IDs.
- LLMs must not generate adjacency matrices.
- RDKit-derived covalent bonds must not be overwritten by LLM output.
- Relation rules are site-level rules; atom-level candidate edges are compiled
  only by deterministic graph-construction logic.
- Skills are structured, user-editable knowledge objects.
- This project produces graph data for downstream ML code; it does not train
  the downstream model.

## Data And Third-Party Code

The `datasets/` directory is included for reproducibility. Confirm that each
dataset can be redistributed under your intended public release terms.

The `MoleCode-main/` directory is included as a bundled dependency. Before
publishing publicly, add your preferred third-party attribution and license
statement in `THIRD_PARTY_NOTICES.md` or in this README.

## GitHub Release Notes

Before pushing to GitHub, review:

- `LICENSE`: add a top-level license for this repository if you want others to
  reuse it clearly.
- `THIRD_PARTY_NOTICES.md`: add MoleCode attribution and dataset/source notices.
- `datasets/`: remove or anonymize anything that should not be public.
- `generated/`: keep it if you want example outputs; remove it if you want a
  source-only repository.
