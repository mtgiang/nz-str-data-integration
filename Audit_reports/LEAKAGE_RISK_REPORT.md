# LEAKAGE AND POINT-IN-TIME RISK REPORT

Audit date 12 October 2026. Machine-readable results: `POINT_IN_TIME_AUDIT.csv` (53 predictors × 10 transitions × 2 cutoff conventions) and `../reproducibility/FEATURE_PROVENANCE_REGISTER.csv` (53 rows). Status vocabulary: **PASS** = verified available at cutoff; **FAIL** = verified unavailable at cutoff; **UNVERIFIED** = insufficient evidence. An UNVERIFIED predictor is not treated as leakage-free.

## 1. Bottom line

1. **No outcome-label leakage was found.** Labels are built only from the next snapshot; T10 rows never enter the training backbone; the next snapshot's first scrape always begins ≥17 days after the origin snapshot's last scrape; preprocessing, calibration and threshold were fitted on development data only (verified numerically).
2. **The nominal snapshot date is not a valid information cutoff.** Listing, calendar and Inside Airbnb review-aggregate predictors (42 of 53) were captured 0–17 days *after* the nominal date (98.48% of origins); 25,284 origins (5.15%) carry review aggregates that include reviews posted after the nominal date. Under a nominal-date cutoff these predictors FAIL.
3. **A defensible operational cutoff exists:** the *snapshot-completion cutoff* C_t = end of the day of the last listing scrape in the origin snapshot (T1 13 Oct 2025 … T10 16 Jul 2026). Under C_t, all listing, calendar and review predictors PASS.
4. **Two predictors FAIL under any cutoff:** `forecast_horizon_days` and `historical_seasonality_index_horizon` are functions of the *next* snapshot date (unknown at the origin; horizons 23–37 days). They are transition-level design variables, not outcome information. Sensitivity analyses (Section 4) show they do not drive the reported T10 discrimination or screening results.
5. **Contextual sources are retrospective vintages.** Construction release timing PASSES for all transitions; rental-bond release timing is PASS for T5, T8, T10 (under C_t) and UNVERIFIED elsewhere; both series, and the ADP seasonality profile, were downloaded after the study period (Sept–Oct 2026) and the rental series is demonstrably revised between vintages.
6. **Consequence:** the study must be presented as a **retrospective temporal evaluation** with an explicitly defined completion cutoff; it does not demonstrate real-time prospective deployment. No re-tuning or re-selection on T10 is required, and the frozen results are retained.

## 2. Status counts per transition (53 predictors)

| T | Completion cutoff PASS / FAIL / UNVERIFIED | Nominal cutoff PASS / FAIL / UNVERIFIED |
|---|---|---|
| T1–T4, T6, T7, T9 | 49 / 2 / 2 | 8 / 42 / 3 |
| T5 | 51 / 2 / 0 | 8 / 42 / 3 |
| T8 | 51 / 2 / 0 | 10 / 42 / 1 |
| **T10 (locked)** | **51 / 2 / 0** | 8 / 42 / 3 |

(For T3, 85.06% of origins were scraped after the nominal date; nominal-cutoff FAIL applies to those origins only.)

## 3. Channel-by-channel findings

**A. Listing and platform attributes (property 7, platform 16, geography 1).** Values are as observed at each listing's `last_scraped` (origin scrape windows in `DATA_LINEAGE_AUDIT.md` §4). No origin was scraped before its nominal date; all origin scrapes finished before the next snapshot started. *Nominal:* FAIL. *Completion:* PASS. `listing_review_age_days` and `days_since_last_review` are explicitly computed from `last_scraped` (`02_clean_listings.py`). `ta_code` derives from scrape-time coordinates and TA 2025 boundaries (UNVERIFIED nominal / PASS completion).

**B. Review history (9).** `review_count_365d` and `review_available` use only origin-file reviews dated in (nominal − 365 d, nominal]; reproduced from raw for 99.89% of origins → PASS under both cutoffs. The seven `review_scores_*` fields and the Inside Airbnb review aggregates in the platform family are scrape-time values: `number_of_reviews_l30d` matches a scrape-anchored 30-day window for 95.4% of origins versus 87.2% for a nominal-anchored window. No review dated after a snapshot's last scrape exists in any review file; review membership is snapshot-specific (no future-snapshot backfilling).

**C. Calendar (10).** Raw calendar prices are empty in all 11 snapshots, so features describe availability configuration and minimum-night rules only. The first calendar date equals the scrape date (no calendar row precedes the nominal date), so the 28-day window covers dates after the origin; for T4, T6, T8 and T9 the 28th observed date falls on or after the next snapshot's nominal date. These are host settings captured at the origin scrape, not realised bookings or occupancy, and contain no outcome information. *Nominal:* FAIL. *Completion:* PASS.

**D. Construction (4).** The scripted release calendar in `03_build_construction_temporal_alignment.py` was checked against the official Stats NZ `PublicationDate` of all 13 releases (July 2025–July 2026, each 10:45 NZT; `external_evidence/statsnz_building_consents_release_pages.json`): **all 13 dates match**. For every transition the consent month used (2025M08–2026M05) was published on or before the nominal origin, and the following month had not yet been released → PASS under both cutoffs. The four features were recomputed from the monthly table (max abs diff 0). Values come from an extract downloaded 1 Oct 2026; revisions between first release and that extract were not quantified (vintage UNVERIFIED).

**E. Residential rental market (2).** The provider states that files released in month M contain data to the end of M−2 (Tenancy Services page, accessed 12 Oct 2026). The M−2 rule was implemented exactly (lag = 2 months for all 670 TA×T rows; features recomputed, max diff 0). Release-day evidence: 13 Feb 2026 (Dec-2025 data), 13 May 2026 (Mar-2026 data), a file with May-2026 data captured by the Internet Archive at 09:22 NZST on 13 Jul 2026, and 10 Sep 2026. Hence T8 PASS (both cutoffs); T5 and T10 PASS under the completion cutoff but UNVERIFIED under the nominal cutoff; T1 and T2 UNVERIFIED with probable FAIL at the nominal cutoff (origins on the 5th/7th, observed releases on the 10th–13th); remaining transitions UNVERIFIED (probably PASS at completion). Separately, the single Sept-2026 vintage used is revised relative to earlier vintages: only 22–82% of TA-month geometric-mean rents are identical, mean |Δlog| 0.1–0.9%, max 14% (`evidence/rental_vintage_revision_check.csv`), and the provider labels recent data provisional because of a bond-system migration.

**F. Historical seasonality (4).** `month_sin`/`month_cos` encode the origin month (verified) → PASS. The ADP profile uses reference months 2023–2024 only (all before T1); 87% of TA profiles are TA-specific, the rest NZ fallback. However, `historical_seasonality_index_horizon` weights months over [origin, next snapshot) and `forecast_horizon_days` equals next − origin date (verified for all transitions) → **FAIL** under both cutoffs. ADP licence and vintage revisions: UNVERIFIED.

**G. Preprocessing and modelling.** Imputer statistics equal T1–T9 medians exactly; one-hot categories fitted in the pipeline on T1–T9; hyper-parameters fixed in the frozen spec and identical in the saved model; threshold = Youden-J maximiser on pooled T4–T9 out-of-time predictions (recomputed 0.026790645 = frozen); Platt fitted on development out-of-time predictions (re-application reproduces T10 calibrated probabilities exactly). Development prediction files contain no T10 rows. Feature-set selection used development macro PR-AUC only. → PASS.

**H. Labels.** T10 labels come only from the 13 Aug 2026 snapshot; 0 label mismatches on reconstruction; the T10 backbone contains only 12 Jul 2026 origins. → PASS.

## 4. Sensitivity of results to the two FAIL predictors (no re-tuning; frozen outputs preserved)

| Analysis | ROC-AUC | PR-AUC | Top-10% recall | Source |
|---|---|---|---|---|
| Frozen model, T10 (reported) | 0.7432 | 0.0695 | 31.65% | J predictions |
| Frozen model re-scored with assumed horizons 23/28/30/35/37 d | 0.7388–0.7436 | 0.0690–0.0696 | 31.12–32.43% | `evidence/T10_horizon_sensitivity_frozen_model.csv` |
| Frozen specification minus 2 horizon predictors (51), development T4–T9 macro | 0.7368 (vs 0.7387 for the same-code 53-predictor refit) | 0.0792 (vs 0.0770) | — | `evidence/horizon_free_sensitivity.csv` |
| Same 51-predictor variant, post-hoc T10 (sensitivity only) | 0.7405 | 0.0694 | 31.65% | idem |

Interpretation: the ex-post horizon predictors change T10 discrimination by ≤0.005 ROC-AUC and do not change top-10% recall; development PR-AUC and Brier score are not worse without them. Their presence nevertheless contradicts a strictly prospective claim and must be disclosed.

## 5. Recommended corrections

1. Define and report the operational cutoff as the origin snapshot-completion date C_t (Table in `DATA_LINEAGE_AUDIT.md` §4); state that the nominal label is only an identifier.
2. Disclose that two seasonality-family predictors use the realised next-snapshot date and report the sensitivity results above; recommend fixed-horizon definitions for any prospective use.
3. Report rental release timing as partially verified (T5, T8, T10 at C_t), and all contextual series as post-study vintages; describe rental vintage revisions.
4. Retain the frozen A5 model and all T10 numbers; label the study a retrospective temporal evaluation; remove any wording implying real-time or verified prospective availability.
5. For a future prospective study: archive contextual vintages at each origin; use fixed-horizon labels; schedule predictions after scrape completion; recompute review aggregates at the cutoff from the review file.

## 6. Is rerunning required?

No for the retrospective claims: no outcome leakage, preprocessing leakage or T10 selection was found, and the only verified-unavailable predictors have a negligible, quantified effect. Yes, but outside the scope of this frozen study, for any claim of prospective deployment (fixed-horizon specification and real-time vintages would be needed). The sensitivity re-fits were run solely as robustness checks and do not replace the frozen model.
