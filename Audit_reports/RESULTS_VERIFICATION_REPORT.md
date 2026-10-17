# RESULTS VERIFICATION REPORT

Audit date 12 October 2026. Every value below was recomputed from frozen artefacts or raw files by `../reproducibility/scripts/audit_03_panel_lineage.py`, `audit_04_results_verification.py`, `audit_06_review_timing.py` and `audit_07_horizon_free_sensitivity.py` (outputs in `evidence/`). No frozen output was overwritten, no model was re-tuned, and T10 was not used for any selection.

Verdicts: **VERIFIED** (independently reproduced to the reported precision), **VERIFIED–MINOR** (reproduced; a negligible numerical or wording difference noted), **NOT VERIFIED** (could not be reproduced with available files), **QUALIFIED** (number correct, interpretation needs revision).

## 1. Claim-by-claim audit of the frozen research facts

| # | Reported value | Independent value | Input / calculation | Verdict |
|---|---|---|---|---|
| 1 | 11 snapshots, 5 Oct 2025–13 Aug 2026 | 11 (`01_raw/airbnb/`) | directory + file manifest | VERIFIED |
| 2 | 10 transitions T1–T10 | 10 | panel `transition` | VERIFIED |
| 3 | 491,140 listing-origins | 491,140; 0 duplicate keys | modelling panel, backbone, outcome table | VERIFIED |
| 4 | Primary outcome = absence from next snapshot | recomputed from cleaned ID sets; 16,949 events, 0 mismatches | `listings_<date>.parquet` | VERIFIED |
| 5 | Secondary = change in within-peer advertised-price percentile | 259,699 eligible; formula in frozen table | secondary outcome parquet | VERIFIED |
| 6 | A5 XGBoost, 53 predictors | 53 unique in frozen spec and saved pipeline | spec JSON (SHA-256 d830c39d…), joblib | VERIFIED |
| 7 | Families 7/16/1/2/4/4/10/9 | 7/16/1/2/4/4/10/9 | `audit_08` family map | VERIFIED |
| 8 | A6: 437 predictors incl. 384 semantic | 437 in `MODEL_DATA_CONTRACT_FROZEN.json`; 384 REVIEW_SEMANTIC in family summary | contracts | VERIFIED |
| 9 | Development T4–T9 | dev predictions contain T4–T9 only, no T10 | IB predictions | VERIFIED |
| 10 | Locked holdout T10 | T10 rows only in J predictions; T1–T9 imputer fit | J predictions + pipeline | VERIFIED |
| 11 | Dev macro ROC-AUC 0.738356 | 0.7383562 | IB predictions, per-window mean | VERIFIED |
| 12 | Dev macro PR-AUC 0.074924 | 0.0749244 | idem (average precision) | VERIFIED |
| 13 | Dev raw Brier 0.0292427 | 0.0292427 | idem | VERIFIED |
| 14 | Dev Platt Brier ≈ 0.028042 | 0.028042 (frozen table); 0.0280435 on independent leave-one-window-out refit (unregularised LR) | IB calibration table; `audit_04` §3 | VERIFIED–MINOR (Δ 1.2×10⁻⁶, regularisation constant) |
| 15 | Threshold 0.0267906449735164 | 0.026790644973516 (Youden J = 0.3486) | pooled A5 dev OOF | VERIFIED |
| 16 | T10 n = 51,096 | 51,096 | J predictions | VERIFIED |
| 17 | T10 events 1,147 | 1,147 | idem | VERIFIED |
| 18 | T10 ROC-AUC 0.743199 | 0.7431991 | idem | VERIFIED |
| 19 | T10 PR-AUC 0.069483 | 0.0694828 | idem | VERIFIED |
| 20 | T10 raw Brier 0.021475 | 0.0214752 | idem | VERIFIED |
| 21 | T10 Platt Brier 0.021549 | 0.0215493 | idem | VERIFIED |
| 22 | TP 787, FP 16,839, FN 360, TN 33,110 | identical | raw p ≥ frozen threshold | VERIFIED |
| 23 | Top-10%: 5,110 / 363 / 7.10% / 31.65% / lift 3.16 | 5,110 / 363 / 7.104% / 31.648% / 3.165 | stable descending sort of raw p; k = ⌈0.1N⌉ | VERIFIED (261 tied probabilities exist; tie order = stored row order, as stated) |
| 24 | Secondary T10 n 43,857 | 43,857 | K predictions | VERIFIED |
| 25 | Zero-change baseline selected | development macro MAE 0.043513 (zero) < 0.043659 (train mean) < 0.051426 (XGB) | K dev predictions T6–T9 | VERIFIED |
| 26 | MAE 0.04653997 | 0.04653997 | idem | VERIFIED |
| 27 | RMSE 0.08092542 | 0.08092542 | idem | VERIFIED |
| 28 | R² −0.000167 | −0.00016712 | idem | VERIFIED |
| 29 | Price missing 172,898 / 35.20% | 172,898 / 35.20% | panel `price` | VERIFIED |
| 30 | T3–T5 complete price missingness, 145,410 | 49,899 + 46,819 + 48,692 = 145,410 | idem | VERIFIED |
| 31 | Scraped later 483,683 / 98.48% | 483,683 / 98.48% | `last_scraped` − `snapshot_date` | VERIFIED |
| 32 | Max discrepancy 17 days | 17 (T2, T3) | idem | VERIFIED |
| 33 | T+2 reappearance 2,016 / 15,802 = 12.76% | 2,016 / 15,802 = 12.76% | cleaned ID sets T+2 | VERIFIED |
| 34 | Construction: modest development gains, not causal | +0.0007 ROC (A1→A2); −0.0024 when removed from A6 | IB tables | QUALIFIED — the manuscript's "consistently positive" overstates two point comparisons without uncertainty; revise to "small" |
| 35 | Calendar and review history more informative than construction | calendar +0.0368 ROC; review history +0.0067 | IB increments | VERIFIED |
| 36 | Semantic and neural alternatives did not outperform | A6 0.7228, MLP 0.6786/0.6693 < A5 0.7384 | IB summary | VERIFIED |

## 2. Other manuscript numbers checked

All of the following match the recomputation: development sample 440,044 with 3.591% events; T4–T9 ranges ROC 0.7168–0.7597, PR 0.0594–0.0999; weighted ROC 0.7383, PR 0.0750, Brier 0.02922; Table 4 rows (A4 0.7317/0.0720/0.02753; A10 0.7297/0.0704/0.02768; A6 0.7228/0.0678/0.02933; MLP 0.6786/0.0561/0.03933 and 0.6693/0.0519/0.02714); pooled dev balanced accuracy 0.6743, sensitivity 0.7249, specificity 0.6237, window range 0.6302–0.6903; isotonic Brier 0.028175 and log loss 0.122179 vs Platt 0.122589; T10 log loss 0.098605 (raw) and 0.099893 (Platt); sensitivity 0.6861, specificity 0.6629, balanced accuracy 0.6745, precision 0.0447; deciles 8/5,110 and 363/5,110 (ratio ≈45); persistence 96.549%, T3 89.387%, T9 98.072%; horizons 23–37 days; secondary counts per transition and range −0.960…0.998; 542,465 cleaned rows; 32,706 consent values; 68 TA categories (66 in secondary).

Cluster bootstrap (600 TA-cluster replicates, seed 2026): re-executed and reproduced **exactly** for all nine intervals (e.g., ROC-AUC 0.72429–0.75937; PR-AUC 0.05629–0.07967; raw Brier 0.01738–0.02507; Platt Brier 0.01757–0.02589; MAE 0.04174–0.05200; RMSE 0.07287–0.08939; R² −0.000725 to −0.000001).

SHAP (exact tree contributions, same 3,000-row sample, seed 2026): the top seven predictors and their values reproduce exactly (open_ratio_w4 0.3081; month_cos 0.2108; listing_review_age_days 0.1683/0.1686; month_sin 0.1521; days_since_last_review 0.1266; open_ratio_w3 0.1113; host_is_superhost 0.1099). Ranks 8–10 depend on whether missing-indicator and one-hot columns are summed into their source predictor (VERIFIED–MINOR; convention should be stated).

## 3. Model-pipeline provenance

* Re-scoring T10 with the saved pipeline: max |Δp| = 0.0. Refitting the frozen specification on T1–T9: T10 probabilities identical (max |Δp| = 0.0).
* Development predictions: refitting the same specification per expanding window gives macro ROC-AUC 0.7387 vs 0.7384 frozen, with per-window differences up to 0.007 (T7) → the I-B development run is **not bit-reproducible** from stored artefacts (NOT VERIFIED at window level; macro conclusions unchanged).
* Selection rule (best macro PR-AUC; ties within 1% resolved by ROC-AUC then Brier) re-applied to the 13 configurations selects XGB-A5 uniquely.

## 4. Fairness of comparisons and possible overstatement

1. Hyper-parameters were fixed across feature sets and the MLP was not tuned. Comparisons therefore test representations under one fixed configuration; the conclusion "semantic/neural alternatives did not improve" holds only for these specifications (manuscript already says so).
2. Ablation differences are single development point estimates without uncertainty; differences below ≈0.005 ROC-AUC (construction, rental, seasonality) should be called small and unstable rather than "consistent".
3. The seasonality family includes the two ex-post horizon predictors; its development increment (+0.0047 ROC) cannot be attributed to historical seasonality alone.
4. `month_sin`/`month_cos` take 8 distinct values over one year, so in this panel they largely index transitions (period effects); their high SHAP rank should not be read as learned seasonality.
5. T10 PR-AUC (0.0695) is below development (0.0749) while ROC-AUC is higher; T10 prevalence (2.245%) is lower than development (2.73% in T4–T9), so PR-AUC is not directly comparable across periods.
6. The Platt calibration chosen in development worsened T10 Brier slightly (0.021549 vs 0.021475); the manuscript reports this correctly.
7. Bootstrap intervals are conditional on the single T10 window and TA clustering; they do not represent between-season uncertainty (stated correctly).

## 5. Discrepancies requiring correction

No numerical discrepancy affecting any table, figure or conclusion was found. Required changes are interpretive: (i) "consistently positive" construction increment → "small"; (ii) disclose the ex-post horizon predictors and their sensitivity; (iii) replace the nominal-date framing with the completion cutoff; (iv) state the SHAP aggregation convention. Rerunning the model is not necessary.
