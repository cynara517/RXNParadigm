# Agent Screening Protocol

This document defines how the agent and LLM-assisted steps participate in the
feature-screening workflow.

## Non-Negotiable Rules

- The final dataset column is the prediction target.
- All preceding columns are feature columns.
- `dataset_profile.json` is the source of truth for target name, feature names,
  and group membership.
- Variable base types are determined from numerical artifacts, not invented by
  the LLM.
- The LLM cannot create new ridge links, MI scores, or correlation values.
- Every LLM keep/remove explanation must cite available numerical evidence.

## Base Variable Types

Every feature receives exactly one base type:

- `completely_independent_variable`: no significant ridge dependency structure
  in its group.
- `chain_correlated_variable`: belongs to a ridge-linked chain or subchain.
- `multicollinear_variable`: can be explained by multiple same-group variables
  through ridge regression with high explanatory quality.

Retention is a decision, not a variable type. A feature decision is one of:

- `retain`
- `remove`
- `needs_review`

## LLM Responsibilities

The LLM receives:

- ridge dependency structures
- ridge edge values and formulas
- MI scores grouped by feature
- high-correlation evidence
- feature names and group names

The LLM returns:

- decomposed subchains
- chain importance labels based on ridge values
- retained variables per chain
- removed variables per chain
- reasons with MI and ridge evidence
- warnings for ambiguous cases

The LLM should favor reproducible numerical evidence over chemical speculation.
Chemical or literature reasoning is explanatory context only unless backed by
the current dataset artifacts.

## Residual Compliance Rule

After initial feature selection, selected features are repeatedly checked for
pairwise absolute Pearson correlation `>= 0.8`.

When the retained set violates this rule:

- remove as few variables as possible
- if one variable causes multiple violations, remove that conflict-center
  variable first
- otherwise compare MI score, variable type, previous chain role, and LLM
  explanation
- record every iteration

The loop stops when all retained features pass the threshold, or when remaining
violations require manual review.
