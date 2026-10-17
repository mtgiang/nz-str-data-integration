# DATA LINEAGE AUDIT — 7th_NZ_STR_Construction_AI

Audit date: 12 October 2026. Auditor: AI-assisted independent audit (Claude, Anthropic; Claude Code session), executed on the author's workstation.
Scope: every file under the project root except `FINAL_MDPI_DATA/` (which holds this audit). No file outside `FINAL_MDPI_DATA/` was modified.
All numbers below were produced by the scripts in `../reproducibility/scripts/` (`audit_01`–`audit_09`); evidence tables are in `evidence/`.

## 1. Inventory

| Item | Result | Evidence |
|---|---|---|
| Files inventoried and hashed (SHA-256) | 668 files (excluding `.ipynb_checkpoints`, `__pycache__`) | `../reproducibility/FILE_CHECKSUMS_SHA256.csv` |
| Source-level inventory | 6 sources (5 used, 1 excluded) | `../reproducibility/DATA_SOURCE_INVENTORY.csv` |
| Raw Inside Airbnb files | 33 of 33 expected (11 snapshots × listings/calendar/reviews) | `../reproducibility/SNAPSHOT_MANIFEST.csv` |
| Code | 12 Python files (incl. `config.py` and an unused backup `02_clean_listings_TEST_backup.py`), 7 notebooks (incl. an unused `Untitled.ipynb`) | `04_code/` |
| Bibliography | **Not in the project folder.** Located at `Downloads/MDPI_DATA_FINAL_25P_SUBMISSION/references.bib` (byte-identical to two other copies, SHA-256 `B36B2ADC…3D24D6`); copied to `references/references_ORIGINAL_input.bib` | — |
| Figures used by `main_reviewed.tex` | `05_results/manuscript_final_L_to_O/figures/journal_ready/F0B, F0C, F1–F4 (.pdf)` | — |

Note: `Downloads/MDPI_DATA_FINAL_25P_SUBMISSION/` contains *different* binary versions of the six figure PDFs (dated 11 Oct 2026) whose generating script is not in the project. The final package uses the in-project `journal_ready` versions, which are traceable to `04_code/generate_manuscript_figures_WINDOWS_TIMES_DISPLAY.py` and `generate_F0_research_story.py`.

## 2. Reconstructed workflow (raw → reported results)

```
01_raw/airbnb/<date>/listings.csv.gz ─► 04_code/02_clean_listings.py ─► 02_intermediate/listings/listings_<date>.parquet (40 cols)
01_raw/airbnb/<date>/reviews.csv.gz  ─► 06_reviews_QC.ipynb cells 0–30 (STEP 24.1–24.7) ─► review_membership_primary365_frozen, review registry, MiniLM-256 embeddings, mean-pooled 365-d embeddings
01_raw/airbnb/<date>/calendar.csv.gz ─► 06_reviews_QC.ipynb cell 45 (Block F) ─► calendar/final_supported/calendar_supported_<date>.parquet
01_raw/geography (TA 2025 SHP)        ─► 06_reviews_QC.ipynb cells 33–35 ─► geography/listing_ta_<date>.parquet
01_raw/construction (Stats NZ CSV)    ─► 03_construction_QC.ipynb + 03_build_construction_temporal_alignment.py ─► construction_features_ta_transition_frozen.parquet (670 = 10 T × 67 TA)
01_raw/property_market (MBIE bonds)   ─► 04_property_market_QC.ipynb ─► rental_market_features_ta_transition_frozen.parquet (670)
01_raw/seasonality (ADP)              ─► 05_seasonality_QC.ipynb ─► seasonality_features_ta_transition_merge_ready_frozen.parquet (680)
        └─► 06_reviews_QC.ipynb cells 31–42 (STEP 24.8A–E): backbone (491,140) → +TA → +context → +reviews → outcomes
        └─► cell 46–48 (Blocks G, H, H2): modelling_panel_T1_T10.parquet (491,140 × 470), feature registry, 437-predictor whitelist
        └─► cell 49–52 (Blocks I, I-B, I-C): development T4–T9, ablations, MLP, calibration, threshold, freeze (SHA-256 d830c39d…)
        └─► cell 55 (Block J): final T1–T9 fit, single T10 evaluation
        └─► block_K_secondary.py (secondary), production_L_to_O.py (bootstrap, SHAP, deciles), figure scripts
        └─► 07_postfreeze_one_run_audit.py, 08_finalize_publication_package.py (post-freeze audits)
```

Step-level processing that exists only inside notebook cells (not as standalone scripts) is identified by cell index (see `FEATURE_PROVENANCE_REGISTER.csv`). The notebooks were inspected statically (code + stored outputs); they were **not re-executed end-to-end**. Instead, every intermediate quantity that feeds the frozen model was independently recomputed from raw or first-stage files (Section 3).

## 3. Verification checklist (prompt §5, items 1–11)

| # | Check | Result | Status |
|---|---|---|---|
| 1 | Snapshot count and dates | 11 snapshots, nominal 2025-10-05 … 2026-08-13; 10 adjacent transitions, horizons 23–37 days | PASS |
| 2 | Schema changes | listings 85 cols (5 snapshots) → 94 (2) → 96 (4); cleaned schema fixed at 40 cols in all 11 | PASS |
| 3 | Duplicate identifiers/keys | 0 duplicate listing IDs in every raw listings file; 0 duplicate review IDs per snapshot; panel 491,140 rows, 0 duplicate (transition, listing_id) | PASS |
| 4 | Join cardinality / geography | backbone = panel = cleaned origin snapshot for every transition (0 unmatched); TA missing 0; 68 TA categories incl. 999; contextual tables 1 row per (transition, TA); construction join mismatch 0 | PASS |
| 5 | Missingness | origin price missing 172,898 / 491,140 = 35.20% (T3–T5 100%: 145,410); construction missing 2,448 rows, rental 3,790 rows in panel; calendar `price`/`adjusted_price` empty in all 11 raw calendars | PASS (reproduced) |
| 6 | Row conservation | raw listing rows = cleaned rows for all 11 snapshots (542,465 total); origins T1–T10 = 491,140 | PASS |
| 7 | Label construction | disappearance recomputed from cleaned ID sets: 16,949 events; 0 mismatches vs panel and vs frozen outcome table | PASS |
| 8 | Outcome eligibility | secondary eligible 259,699 (T1 42,368; T6 43,063; T7 42,810; T8 43,624; T9 43,977; T10 43,857); T2–T5 structurally unavailable; range −0.960 to 0.998; peer TA×room 255,949 / TA-only 3,750 | PASS |
| 9 | Feature construction | construction (4), rental (2), seasonality index (1), month sin/cos recomputed from source: max abs diff = 0 for all; `review_count_365d` reproduced from raw origin review file for 99.89% of origins (562 residual differences, attributable to the registry's cross-snapshot review-ID de-duplication; not further resolved) | PASS (review count: PASS with minor residual) |
| 10 | Transformations / imputation | numeric median imputer (+ missing indicators) statistics in the frozen T1–T9 pipeline equal T1–T9 medians exactly (max diff 0) and differ from T1–T10 medians (max diff 7) → fitted without T10 | PASS |
| 11 | Model & prediction provenance | frozen spec SHA-256 matches; re-scoring T10 with the saved pipeline reproduces saved probabilities exactly (max diff 0.0); **refitting** the frozen specification on T1–T9 reproduces T10 probabilities bit-for-bit; Platt model re-application reproduces calibrated probabilities exactly | PASS |
| 11b | Development prediction provenance | refitting the A5 specification with the J pipeline on each expanding window reproduces development macro ROC-AUC 0.7387 vs frozen 0.7384, but per-window AUCs differ by up to 0.007 (T7) → frozen I-B development predictions are **not bit-reproducible** from the stored J pipeline (I-B implementation details only in notebook cell 50) | PARTIAL — documented |

## 4. Snapshot timing (operational facts)

| T | Origin nominal | Origin scrape window | Next scrape window | Gap completion→next first scrape (d) |
|---|---|---|---|---|
| T1 | 2025-10-05 | 10-07 → 10-13 | 11-11 → 11-24 | 29 |
| T2 | 2025-11-07 | 11-11 → 11-24 | 12-11 → 12-28 | 17 |
| T3 | 2025-12-11 | 12-11 → 12-28 | 01-23 → 01-24 | 26 |
| T4 | 2026-01-16 | 01-23 → 01-24 | 02-15 → 02-16 | 22 |
| T5 | 2026-02-13 | 02-15 → 02-16 | 03-21 → 03-23 | 33 |
| T6 | 2026-03-17 | 03-21 → 03-23 | 04-20 → 04-22 | 28 |
| T7 | 2026-04-16 | 04-20 → 04-22 | 05-24 → 06-03 | 32 |
| T8 | 2026-05-23 | 05-24 → 06-03 | 06-20 → 06-23 | 17 |
| T9 | 2026-06-19 | 06-20 → 06-23 | 07-14 → 07-16 | 21 |
| T10 | 2026-07-12 | 07-14 → 07-16 | 08-13 → 08-25 | 28 |

483,683 origins (98.48%) were scraped after their nominal date (max 17 days; none earlier); 7,457 T3 origins were scraped on the nominal date. Calendar files start on the scrape date; no calendar row precedes its nominal date. Review files contain 759–11,218 reviews per snapshot dated after the nominal date but none after the snapshot's last scrape.

## 5. Other lineage observations

* **T3 anomaly.** T3 disappearance is 10.6% (5,296) vs 1.9–3.3% elsewhere; 24.98% of T3 disappearances reappear at T+2 (12.76% overall, 2,016/15,802 reproduced exactly). The 2026-01-16 snapshot began scraping 7 days after its nominal date and finished within 2 days. This is consistent with a collection-coverage effect, but the cause is not determinable from the data.
* **Secondary outcome table.** In the frozen secondary table, `anchor_price` is null for all T2 rows although the raw T2 listings contain 45,569 prices; secondary eligibility requires both endpoints, so this does not change any reported count.
* **Unused artefacts.** `02_clean_listings_TEST_backup.py`, `Untitled.ipynb`, `05_results/development_modelling/` (an earlier 3-model development run, superseded by I-B) and the MCERT workbook are not inputs to any reported result.
* **Not re-executed.** Review-text language detection and MiniLM embedding generation (A6 comparator only) were not re-run; the A6 results are taken from the frozen development predictions, whose metrics were recomputed (max diff 1.6×10⁻⁷).

## 6. Conclusion

The analytical panel, labels, contextual features and locked T10 predictions are fully traceable to the raw files and reproduce exactly. The principal lineage risks are temporal (Section 4 and `LEAKAGE_RISK_REPORT.md`), not arithmetic.
