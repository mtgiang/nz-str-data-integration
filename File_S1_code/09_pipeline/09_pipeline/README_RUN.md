# One-command pipeline (v2, point-in-time conservative)

## Install (once, in the project root)
```
pip install -r 09_pipeline/requirements.txt
```

## Run everything
```
python 09_pipeline/run_all.py
```
Rebuild from a stage: `--force manifest|panel|features|models`. Outputs go to `09_pipeline_out/`.

## What it does (all from 01_raw)
| Stage | Output |
|---|---|
| manifest | `SNAPSHOT_MANIFEST.csv` (SHA-256 of every raw file) |
| panel | `panel.parquet`, `audit_origins.csv` (counts, labels, scrape delay, missing price, T+2 reappearance, TA match), `audit_snapshots.csv` |
| features | `features.parquet`, `audit_calendar.csv`, `audit_reviews.csv`, `audit_construction_timing.csv`, `audit_rental_timing.csv`, `audit_consents_columns.csv`, `seasonality_profile.csv` |
| models | `dev_by_window.csv`, `dev_summary.csv`, `FROZEN_SPEC.json` (written BEFORE T10), `T10_predictions.csv`, `T10_topk.csv`, `T10_deciles.csv`, `T10_shap_mean_abs.csv`, `secondary_dev.csv`, `RESULTS.json`, `results_macros.tex`, `figures/*.pdf` |

## Point-in-time rules (fixed in config.yaml before any v2 result is seen)
- Operational origin = latest `last_scraped` in the origin snapshot; checked to fall before the next snapshot.
- Listing fields: as published in the origin snapshot. Review features: only reviews dated <= nominal snapshot date. Calendar: first 28 dated rows; first date logged against the nominal date.
- Construction: latest consent month whose official release date <= origin (from `07_provenance/consents_release_calendar.csv`); if that file is absent, conservative M-3 and flagged UNVERIFIED.
- Rental bond: month M-3 (file released in month M covers to M-2; release day unknown).
- Seasonality: national 2023-2024 monthly profile only.
- A6 comparator: pooled 384-d sentence embeddings (all-MiniLM-L6-v2) of the 10 most recent reviews dated <= nominal origin; MLP (128,32) on A5 and A6.
- HPI: ex-post robustness model only (`A5+hpi`), never a selection candidate.
- Category vocabularies, imputation, scaling, threshold, calibration: development windows only.

## Checks before trusting results
1. `audit_consents_columns.csv`: exactly the intended "new dwellings / number" columns used, 67 TAs.
2. `audit_bond_unmatched.csv`: only non-TA rows (e.g. totals) unmatched.
3. `run_log.txt`: no `WARNING ... DROPPED` (incl. semantic).
4. `audit_origins.csv`: `ta_unmatched` ~ 0.

## Manuscript
Put `\input{results_macros}` in the preamble and write numbers as `\ResTtenROC`, `\ResTopTenRecall`, etc.
