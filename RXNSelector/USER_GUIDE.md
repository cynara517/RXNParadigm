# RXNSelector User Guide

RXNSelector is an auditable reaction descriptor screening workflow. It supports
both command-line execution and an interactive Streamlit interface.

## 1. Installation

From the `RXNSelector` directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install exposes:

```bash
rxnselector --help
rxnselector-ui
```

## 2. Input Data

The dataset format is fixed:

1. The first row contains column names.
2. The first `N-1` columns are reaction descriptor features.
3. The final column is the prediction target.
4. Feature names are used for group parsing and chain interpretation.

The bundled example dataset is:

```text
examples/00_01_data.csv
```

The bundled screening config is:

```text
configs/reaction_dft_screening.yaml
```

## 3. CLI Usage

Run the bundled example:

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

Each run writes outputs to:

```text
runs/<run_id>/
```

Important outputs include:

- `step_01_dataset_profile/dataset_profile.json`
- `step_02_grouped_correlation/high_correlation_pairs.csv`
- `step_03_mutual_information/mi_scores_by_group.csv`
- `step_04_ridge_dependency_graph/ridge_dependency_report.md`
- `step_05_llm_chain_decomposition/llm_chain_decomposition.md`
- `step_06_feature_selection_from_chains/feature_decision_table.csv`
- `step_07_residual_collinearity_check/final_features.json`
- `step_08_final_feature_export/selected_dataset.csv`

## 4. Streamlit Experience

Launch the UI:

```bash
rxnselector-ui
```

or:

```bash
python scripts/start_streamlit.py
```

Open:

```text
http://127.0.0.1:8501
```

The Streamlit interface supports:

- editing the YAML screening config;
- uploading a reaction descriptor dataset;
- creating a run;
- executing workflow steps;
- viewing chain decomposition, selected features, reports, and downloadable artifacts.

## 5. Optional LLM Chain Review

By default, RXNSelector uses deterministic MI/ridge rules. To add an LLM review
layer after the default chain decomposition, enable:

```yaml
llm_chain_decomposition:
  enabled: true
  provider: openai_responses
  base_url: https://api.openai.com/v1
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
  literature_search: true
  web_search_tool_type: web_search_preview
  literature_context: "Optional reaction class, catalyst family, or literature notes."
```

Set the API key before running:

```bash
export OPENAI_API_KEY="your_api_key"
```

The LLM review does not replace the deterministic retain/remove decisions. It
adds merge/split suggestions and concise reaction-meaning summaries for each
chain.

## 6. Testing

Run:

```bash
python -m pytest -q
python -m compileall dft_preprocess_agent skills scripts ui tests
```

## 7. Deployment Notes

The Streamlit app can be deployed after a local smoke test. Recommended order:

1. Run the test suite.
2. Start Streamlit locally and confirm the app loads.
3. Deploy to a target platform such as Streamlit Community Cloud, Hugging Face
   Spaces, Docker, or an internal server.
4. Configure secrets such as `OPENAI_API_KEY` in the deployment platform rather
   than committing them to the repository.
