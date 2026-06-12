# Skill Workflow Design

## Goal

This workflow is a general feature-screening system for reaction datasets. It is
not designed to force a fixed number of output features.

The required dataset format is strict:

1. The first row contains column names.
2. Columns `1..N-1` are feature columns.
3. The final column is the prediction target.
4. Feature names are used to parse reaction groups and to support LLM reasoning,
   so users must check names carefully before running the workflow.

The bundled reaction-yield example dataset in `examples/00_01_data.csv` is only a
benchmark fixture. Its expected 30-feature result must not define the generic
workflow.

## Runtime Flow

1. User uploads a normalized dataset or provides a dataset path through the CLI.
2. The platform creates a `run_id` and copies the dataset into
   `runs/<run_id>/input/`.
3. `workflow_state.json` records the current dataset path, completed Skills,
   selected features, and generated artifacts.
4. Every step is executed as one Skill through `WorkflowEngine.run_skill(...)`.
5. Later Skills read artifacts produced by earlier Skills. Group names and the
   target column are read from `dataset_profile.json`.
6. If a Skill creates a new dataset, `current_dataset_path` is updated.

## Skill Sequence

The generic workflow order is:

1. `dataset_profile`
2. `grouped_correlation`
3. `mutual_information`
4. `ridge_dependency_graph`
5. `llm_chain_decomposition`
6. `feature_selection_from_chains`
7. `residual_collinearity_check`
8. `final_feature_export`

`benchmark_validate` is allowed only for benchmark runs.

## Skill Contracts

### dataset_profile

Purpose: validate the strict input format and parse the dataset once.

Target rule: the final column is the target. Feature columns are every column
before the final column.

Output artifacts:

- `dataset_profile.json`

Required content:

- dataset-format warning text
- target column name
- original feature count
- original feature name list
- group-to-feature mapping

No filtering is performed in this step.

### grouped_correlation

Purpose: diagnose Pearson correlation inside each parsed group.

Inputs:

- `dataset_profile.json`

Output artifacts:

- `matrix_<group>.csv` for every group
- `corr_<group>.svg` for every group
- `high_correlation_pairs.csv`
- `correlation_summary.csv`

Group names must come from `dataset_profile.json`.

### mutual_information

Purpose: calculate mutual information between every feature and the target.

Inputs:

- `dataset_profile.json`

Output artifacts:

- `mi_scores_by_group.csv`

The MI target is always the final dataset column recorded in
`dataset_profile.json`.

### ridge_dependency_graph

Purpose: construct the numerical dependency structure used for chain reasoning.

For each group, the Skill fits ridge regressions of the form:

```text
feature_i ~ other features in the same group
```

It uses these regressions to classify features into three base types:

- `completely_independent_variable`
- `chain_correlated_variable`
- `multicollinear_variable`

Output artifacts:

- `ridge_dependency_report.json`
- `ridge_dependency_report.md`

The report must include ridge edges, ridge values, fitted formulas, raw connected
structures, and the base variable type for every feature.

### llm_chain_decomposition

Purpose: use the LLM only after numerical ridge structures exist.

The LLM may:

- split ridge structures into interpretable subchains
- rank chains by ridge values and importance
- combine chain structure with MI scores
- explain which chain or multicollinear variables should be retained
- explain which variables should be removed

The LLM must not:

- invent chain relationships that do not exist in `ridge_dependency_report.json`
- invent MI scores or correlation values
- make a keep/remove decision without citing numerical evidence

Output artifacts:

- `llm_chain_decomposition.json`
- `llm_chain_decomposition.md`

### feature_selection_from_chains

Purpose: combine deterministic variable types with LLM chain decisions.

Rules:

- Completely independent variables are retained.
- Chain-correlated variables are retained or removed according to LLM chain
  decomposition plus MI.
- Multicollinear variables are retained or removed according to LLM reasoning
  over ridge formulas, chain structure, and MI.

Output artifacts:

- `initial_selected_features.json`
- `feature_decision_table.csv`

### residual_collinearity_check

Purpose: compliance loop for the selected feature set.

Rule: the final selected features should have pairwise absolute Pearson
correlations below `0.8`, unless no further automatic deletion is safe.

Loop:

1. Compute correlations among currently selected features.
2. If no pair has `abs(r) >= 0.8`, stop.
3. If pairs exceed the threshold, identify conflict-center variables.
4. Delete as few variables as possible.
5. If one variable causes multiple over-threshold pairs, delete that variable.
6. Otherwise use MI, previous chain decisions, and LLM explanations to choose
   the lower-value deletion.
7. Recompute and continue.

Output artifacts:

- `residual_collinearity_iterations.csv`
- `final_features.json`
- `final_correlation_matrix.csv`
- `final_correlation_graph.svg`

### final_feature_export

Purpose: export final results only.

Output artifacts:

- `selected_dataset.csv`
- `final_report.md`

## Interfaces

The Streamlit UI, CLI, and Python API must call the same `WorkflowEngine` and the
same Skills. UI code should not call screening algorithms directly.

The UI/help text must warn users:

```text
Please upload a normalized dataset: the first N-1 columns are features and the
last column is the prediction target. The header row will be used as feature
names for group parsing and LLM reaction reasoning. Check names carefully before
running the workflow.
```
