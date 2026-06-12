# RXNParadigm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/Research-Chemical%20Graph%20Learning-red)]()

**RXNParadigm** is a novel framework for **complete graph structure learning of chemical reactions**. Unlike existing methods that rely on molecular fingerprints or SMILES sequences, ParadigmRXN builds full reaction graphs directly from DFT descriptor datasets and applies a **paradigm‑shifting feature selection** method to assign essential attributes to nodes.

> This is the first work that tackles reaction graph learning from the perspective of **complete graph construction**, bridging the gap between raw reaction data and meaningful node features.

---

## Table of Contents
- [Key Innovations](#key-innovations)
- [Comparison with Existing Methods](#comparison-with-existing-methods)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

---

## Key Innovations

### 1. End‑to‑End Reaction Graph Construction
- No pre‑defined templates or molecular graphs required.
- Directly generates full graphs (reactants, products, and their relationships) from DFT descriptors and SMILES.
- Supports popular datasets: Suzuki reactions, Doyle B–H (2018), etc.
- **First tool** that accounts for both **inter‑molecular interactions** among reactants and **atomic‑level connections** within single molecules.
- Packaged as an **agent tool** – easy to call from your own code.

### 2. Paradigm‑Shifting Feature Selection
- Replaces the traditional *use all hundreds of DFT descriptors* OR *use none of DFT descriptors but only SMILES* approach.
- Selects the most informative node features via:
  - Correlation analysis
  - Information entropy strategies
  - Decoupling of chain relationships among reaction nodes
- Also provided as an **interactive GUI** for exploration.

## RXNSelector Streamlit App

The release-ready RXNSelector app is packaged under:

```text
RXNSelector/
```

For Streamlit Community Cloud deployment, use:

```text
Main file path: RXNSelector/ui/streamlit_app.py
```

The repository root `requirements.txt` mirrors the RXNSelector app dependencies
so Streamlit Cloud installs packages such as `PyYAML` before importing the app.

The app-specific installation guide, CLI examples, and Streamlit usage notes are
available in:

```text
RXNSelector/USER_GUIDE.md
```

---

## Comparison with Existing Methods

| Method | Graph Source | Feature Assignment | Complete Graph Construction |
|--------|--------------|--------------------|-----------------------------|
| RXNPredictor | Molecular graph concatenation | Fixed atom/bond features | ❌ |
| RXNTransformer | Sequence (SMILES) | Attention weights | ❌ |
| YieldGNN | Molecular graphs + chemical descriptors | Structural (AGNN) + chemical features | ❌ |
| BH‑RGNN | Custom B–H reaction graph (each reactant as a node) | Learned via GNN on tailored graph | ❌ |
| **RXNParadigm** | **Directly from reaction DFT descriptor datasets** | **Paradigm‑shifting feature selector (correlation + entropy + chain decoupling)** | ✅ **First** |

---
