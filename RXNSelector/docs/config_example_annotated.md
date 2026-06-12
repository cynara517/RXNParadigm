# Annotated Screening Config Example

This config controls how the feature-screening workflow parses groups, computes
correlations, builds ridge-regression dependency chains, and performs the final
correlation compliance check.

Important dataset rule: the workflow always treats the first `N-1` columns as
features and the final column as the prediction target. Do not put
`target_column` or `first_n_feature_columns` in this config.

## Copyable YAML

```yaml
# A human-readable name for this screening configuration.
name: reaction_dft_variable_screening

# A short description shown in saved run metadata and reports.
description: RXNSelector workflow for reaction-yield datasets.

# Random seed used by algorithms that need reproducible random behavior.
random_state: 42

# Feature groups are parsed from column names.
# Each group can define prefixes and/or regular-expression patterns.
# A feature is assigned to the first matching group.
groups:
  # All feature names starting with "additive_" are assigned to this group.
  additive:
    prefixes:
      - additive_

  # All feature names starting with "aryl_halide_" are assigned to this group.
  aryl_halide:
    prefixes:
      - aryl_halide_

  # All feature names starting with "base_" are assigned to this group.
  base:
    prefixes:
      - base_

  # All feature names starting with "ligand_" are assigned to this group.
  ligand:
    prefixes:
      - ligand_

# Pearson-correlation diagnosis settings.
correlation:
  # Main threshold used to define high-correlation pairs.
  # Example: 0.8 means pairs with abs(Pearson r) >= 0.8 are reported as high.
  primary_threshold: 0.8

  # Extra thresholds included in correlation_summary.csv.
  report_thresholds:
    - 0.8
    - 0.9

# Ridge-regression dependency settings.
# These values control how chain-correlated and multicollinear variables are detected.
ridge_dependency:
  # Ridge regularization strength. Larger values shrink coefficients more strongly.
  alpha: 1.0

  # Minimum absolute standardized ridge coefficient for creating a dependency edge.
  # Higher values create fewer, stronger chain links.
  edge_threshold: 0.9

  # Minimum ridge R^2 for labeling a variable as explainable by same-group variables.
  multicollinearity_r2_threshold: 0.9

  # Minimum absolute standardized coefficient for a predictor to count as a contributor.
  contributor_threshold: 0.05

  # Minimum number of contributors required before a high-R^2 variable is treated
  # as a multicollinear variable rather than a simple pairwise chain relation.
  min_multicollinearity_contributors: 2

# Optional external LLM chain-review settings.
# Keep disabled for fully local deterministic screening. When enabled, the LLM
# reviews the default MI/ridge chains after they are generated.
llm_chain_decomposition:
  # When false, the workflow uses the deterministic MI/ridge protocol adapter.
  enabled: false

  # Use openai_responses when you want the LLM to use a web-search-capable
  # Responses endpoint. Use openai_compatible for generic chat-completions APIs.
  provider: openai_responses
  base_url: https://api.openai.com/v1
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY

  # Low temperature is recommended for auditable chain-review summaries.
  temperature: 0.0
  max_tokens: 4000
  timeout_seconds: 60

  # Validate that the LLM preserves chain IDs and uses only known features in
  # merge/split suggestions.
  strict_validation: true

  # Prompt search-capable endpoints to use literature/network context when
  # reviewing whether chains can be merged or should be split.
  literature_search: true

  # Tool type sent to the Responses API when literature_search is enabled.
  web_search_tool_type: web_search_preview

  # Optional user-supplied text context, such as known reaction class,
  # substrates, catalyst family, or manually collected citation notes.
  literature_context: ""

  # If the API key, request, or validation fails, continue with deterministic
  # MI/ridge chains and record the fallback reason in the output JSON.
  fallback_on_error: true

# Final compliance-check settings.
residual_check:
  # Final retained variables should have pairwise abs(Pearson r) below this value.
  threshold: 0.8

  # Safety cap for the residual loop.
  max_iterations: 100
```

## Field Reference

`name`

Use a short identifier for the config. It does not affect calculations.

`description`

Use one sentence to describe the intended dataset or screening scenario. It does
not affect calculations.

`random_state`

Use any integer. Keep it fixed when you want reproducible mutual-information
scores.

`groups`

Define how feature columns are split into reaction groups. The group names in
this section become output names such as `matrix_additive.csv` and
`corr_ligand.svg`.

`groups.<group>.prefixes`

List column-name prefixes for the group. For example, `additive_` matches
`additive_E_HOMO` and `additive_surface_area`.

`groups.<group>.patterns`

Optional list of regular expressions. Use this only when prefixes are not enough.
For example, `^cat_[0-9]+_` could match catalyst descriptors with numbered
prefixes.

`correlation.primary_threshold`

Controls which Pearson pairs are written to `high_correlation_pairs.csv`.
Recommended starting value: `0.8`.

`correlation.report_thresholds`

Controls summary counts only. These values do not change downstream screening
unless another step explicitly reads them.

`ridge_dependency.alpha`

Controls ridge regularization. Recommended starting value: `1.0`. Increase it if
ridge formulas look unstable; decrease it if dependency edges are too weak.

`ridge_dependency.edge_threshold`

Controls the strength needed to form a ridge dependency edge. Higher values
produce fewer chain links. Recommended starting value: `0.9`.

`ridge_dependency.multicollinearity_r2_threshold`

Controls how well a feature must be explained by other same-group features before
it can be considered multicollinear. Recommended starting value: `0.9`.

`ridge_dependency.contributor_threshold`

Controls which ridge coefficients are counted as meaningful contributors.
Recommended starting value: `0.05` for standardized variables.

`ridge_dependency.min_multicollinearity_contributors`

Controls how many contributors are required for multicollinearity. Use `2` or
`3`. A larger value makes the multicollinearity label stricter.

`llm_chain_decomposition.enabled`

Controls whether step 5 calls an external LLM after deterministic MI/ridge chain
generation. Keep `false` for fully local screening. Set `true` to add LLM
review fields such as merge/split suggestions, literature/search basis, and
reaction-meaning summaries.

`llm_chain_decomposition.provider`

Use `openai_responses` to call `<base_url>/responses`, which can request a web
search tool for literature-aware review. Use `openai_compatible` to call a
generic `<base_url>/chat/completions` endpoint; this works for local or proxy
models but cannot guarantee real network/literature search.

`llm_chain_decomposition.base_url`

Chat-completions API base URL. Use `https://api.openai.com/v1` for OpenAI, or a
compatible local/proxy endpoint.

`llm_chain_decomposition.model`

Model name sent to the chat-completions endpoint.

`llm_chain_decomposition.api_key_env`

Environment variable that contains the API key. The key is read at runtime and
is not stored in run artifacts.

`llm_chain_decomposition.strict_validation`

When `true`, the LLM response must preserve known chain IDs and use only input
features in merge/split suggestions.

`llm_chain_decomposition.literature_search`

When `true` and `provider: openai_responses`, the request includes a web-search
tool so the model can use literature/network context when judging whether chains
can be merged or split. With `provider: openai_compatible`, this flag is passed
as review context but depends on the endpoint's own capabilities.

`llm_chain_decomposition.web_search_tool_type`

Tool type sent to the Responses API. The default is `web_search_preview`.

`llm_chain_decomposition.literature_context`

Optional user-supplied text sent to the LLM, such as reaction class, substrate
scope, catalyst family, or manually collected literature notes.

`llm_chain_decomposition.fallback_on_error`

When `true`, API or validation failures keep the deterministic MI/ridge chain
output and record the fallback reason. When `false`, such failures stop the
workflow.

`residual_check.threshold`

Controls the final compliance rule. The retained features should have pairwise
absolute Pearson correlations below this value. Recommended value: `0.8`.

`residual_check.max_iterations`

Maximum number of deletion/recheck iterations. This prevents accidental infinite
loops when the selected feature set cannot be made compliant automatically.
