# RXNSelector

RXNSelector is an auditable reaction feature-screening workflow. It profiles a
reaction descriptor table, diagnoses grouped correlations, calculates mutual
information with the prediction target, builds ridge-regression dependency
structures, decomposes chain relationships, and exports a screened feature set.

## Data Format

RXNSelector uses a strict dataset format:

1. The first row contains column names.
2. Columns `1..N-1` are feature columns.
3. The final column is the prediction target.
4. Feature names are used for group parsing and chain reasoning, so check them
   carefully before running the workflow.

Do not add `target_column` or `first_n_feature_columns` to the config. The target
is always the final dataset column.

## Install

From the project directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install exposes two commands:

```bash
rxnselector --help
rxnselector-ui
```

If you do not want to install the package yet, you can still run the source
scripts directly:

```bash
python scripts/start_streamlit.py
python scripts/run_reaction_screening.py
```

## Streamlit UI

Launch the UI:

```bash
rxnselector-ui
```

or:

```bash
python scripts/start_streamlit.py
```

Then open:

```text
http://127.0.0.1:8501
```

The UI lets users edit the YAML config, upload a dataset, run each workflow step,
view step-specific results, and download artifacts.

## CLI

Run the bundled reaction-yield example:

```bash
rxnselector run \
  --dataset examples/00_01_data.csv \
  --config configs/reaction_dft_screening.yaml \
  --run-id demo_run
```

Equivalent source-tree command:

```bash
python -m dft_preprocess_agent.cli run \
  --dataset examples/00_01_data.csv \
  --config configs/reaction_dft_screening.yaml \
  --run-id demo_run
```

Each run writes artifacts under:

```text
runs/<run_id>/
```

Key outputs include:

- `step_01_dataset_profile/dataset_profile.json`
- `step_02_grouped_correlation/high_correlation_pairs.csv`
- `step_02_grouped_correlation/corr_<group>.svg`
- `step_03_mutual_information/mi_scores_by_group.csv`
- `step_04_ridge_dependency_graph/ridge_dependency_report.md`
- `step_05_llm_chain_decomposition/llm_chain_decomposition.md`
- `step_06_feature_selection_from_chains/feature_decision_table.csv`
- `step_07_residual_collinearity_check/final_features.json`
- `step_08_final_feature_export/selected_dataset.csv`

## Workflow Steps

1. `dataset_profile`: validates the strict dataset format and records feature
   names, the target column, and group membership.
2. `grouped_correlation`: computes group-wise Pearson matrices, high-correlation
   pairs, summaries, and SVG heatmaps with coordinate rulers.
3. `mutual_information`: computes feature-vs-target MI scores by group.
4. `ridge_dependency_graph`: builds ridge-regression dependency structures,
   equations, R² values, and rendered chain relationships.
5. `llm_chain_decomposition`: decomposes ridge structures into chain decisions.
   By default this uses deterministic MI/ridge rules. Set
   `llm_chain_decomposition.enabled: true` to add an LLM review layer after the
   default rules. With `provider: openai_responses`, the request can include a
   web-search tool so the LLM can use literature/network context to judge whether
   chains should be merged or split and summarize the reaction meaning of each
   chain; baseline retain/remove decisions are kept stable.
6. `feature_selection_from_chains`: combines independent variables and chain
   decisions into an initial selected feature set.
7. `residual_collinearity_check`: iteratively checks selected features for
   residual pairwise correlation and removes as few variables as possible.
8. `final_feature_export`: exports the screened dataset and final report.

## Config

Start from:

```text
configs/reaction_dft_screening.yaml
```

An annotated example is available at:

```text
docs/config_example_annotated.md
```

The most important sections are:

- `groups`: how feature names are assigned to reaction groups.
- `correlation`: Pearson correlation thresholds.
- `ridge_dependency`: ridge edge and multicollinearity settings.
- `llm_chain_decomposition`: optional OpenAI-compatible LLM settings for
  post-rule chain review, merge/split suggestions, and reaction-meaning summaries.
- `residual_check`: final correlation compliance settings.

## Test

```bash
python -m pytest -q
python scripts/run_reaction_screening.py
python -m compileall dft_preprocess_agent skills scripts ui tests
```

## Release Status

RXNSelector is currently prepared for source-tree and editable-install use. The
first public release should be tested on a clean environment before publishing
to GitHub or PyPI.
