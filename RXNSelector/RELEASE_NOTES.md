# RXNSelector Release Notes

## Release Package

This directory is a clean release copy of RXNSelector. Runtime outputs such as
`runs/` and uploaded files are intentionally excluded from the package.

## Current Capabilities

- strict reaction descriptor dataset parsing;
- grouped correlation diagnosis;
- mutual-information scoring;
- ridge-regression dependency graph construction;
- deterministic reaction-chain decomposition;
- optional post-rule LLM chain review with merge/split suggestions and chain
  meaning summaries;
- chain-based feature selection;
- residual collinearity compliance checking;
- final screened dataset export;
- CLI and Streamlit user entry points.

## Recommended Release Validation

Run from this directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall dft_preprocess_agent skills scripts ui tests
python scripts/start_streamlit.py
```

Then open:

```text
http://127.0.0.1:8501
```

## Deployment Recommendation

Test the Streamlit entry locally before deployment. Deploy only after confirming
that the UI loads, the bundled config parses, and a demo run can be created.

For public demos, Streamlit Community Cloud is the most direct path. For private
or API-key-backed use, prefer an internal server, Docker, or a managed app
platform with secret management.

