# RXNSelector Release Checklist

Use this checklist before publishing RXNSelector.

## Local Validation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python scripts/run_reaction_screening.py
python -m compileall dft_preprocess_agent skills scripts ui tests
```

## CLI Smoke Test

```bash
rxnselector run \
  --dataset examples/00_01_data.csv \
  --config configs/reaction_dft_screening.yaml \
  --run-id release_smoke_test
```

Check:

- `runs/release_smoke_test/workflow_state.json`
- `runs/release_smoke_test/step_03_mutual_information/mi_scores_by_group.csv`
- `runs/release_smoke_test/step_04_ridge_dependency_graph/ridge_dependency_report.md`
- `runs/release_smoke_test/step_08_final_feature_export/selected_dataset.csv`

## UI Smoke Test

```bash
rxnselector-ui
```

Open:

```text
http://127.0.0.1:8501
```

Check every step:

1. Dataset profile
2. Correlation diagnosis
3. MI scoring
4. Ridge dependency graph
5. Chain decomposition
6. Chain selection
7. Compliance check
8. Export dataset

## GitHub Release

```bash
git init
git add .
git commit -m "Release RXNSelector v0.1.0"
git branch -M main
git remote add origin https://github.com/<your-name>/RXNSelector.git
git push -u origin main
```

Create a GitHub release:

```text
Tag: v0.1.0
Title: RXNSelector v0.1.0
```

## Notes

RXNSelector v0.1.0 is intended for source-tree or editable-install use. Public
PyPI publishing should wait until project data files such as `skills/`, `ui/`,
and `configs/` are packaged as installable resources.
