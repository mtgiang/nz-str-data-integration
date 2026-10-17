"""Phase 2 - build FEATURE_PROVENANCE_REGISTER.csv (53 rows) and POINT_IN_TIME_AUDIT.csv (53 x 10 transitions x 2 cutoffs).

Status vocabulary: PASS (verified available at cutoff), FAIL (verified unavailable at cutoff), UNVERIFIED (insufficient evidence).
Two cutoff conventions are evaluated:
  N  nominal: 00:00 on the Inside Airbnb snapshot label date (the convention implied by the original manuscript);
  C  completion: end of the day on which the last listing in the origin snapshot was scraped (max last_scraped),
     which the lineage audit shows precedes the first scrape of the next snapshot by >= 17 days.
Evidence files referenced are produced by audit_02/03/04/06 and are in ../audit/evidence/.

Usage: python -I audit_08_point_in_time_register.py <PROJECT_ROOT> <AUDIT_DIR> <REPRO_DIR>
"""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]).resolve(); aud = Path(sys.argv[2]).resolve(); rep = Path(sys.argv[3]).resolve()
ev = aud / "evidence"
spec = json.loads((root / "05_results/development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json").read_text(encoding="utf-8"))
FEAT = spec["final_primary_model"]["predictors"]
lin = pd.read_csv(ev / "panel_lineage_by_transition.csv")
T = lin.transition.tolist(); NOM = dict(zip(T, lin.origin_nominal)); COMP = dict(zip(T, lin.origin_scrape_completion))
LAG = dict(zip(T, lin.scraped_after_pct))

PROPERTY = ["property_type", "room_type", "accommodates", "bathrooms", "bedrooms", "beds", "amenities_count"]
PLATFORM = ["host_is_superhost", "host_listings_count", "host_has_profile_pic", "host_identity_verified", "minimum_nights", "maximum_nights",
            "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "calculated_host_listings_count",
            "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
            "calculated_host_listings_count_shared_rooms", "reviews_per_month", "listing_review_age_days", "days_since_last_review"]
GEO = ["ta_code"]; RENT = ["rental_geomean_rent_log", "rental_geomean_rent_change_12m_log"]
CONS = ["construction_dwelling_3m_log", "construction_dwelling_12m_log", "construction_momentum_3m_vs_prior9m", "construction_multiunit_share_12m"]
SEAS = ["forecast_horizon_days", "historical_seasonality_index_horizon", "month_sin", "month_cos"]
CAL = [f for f in FEAT if f.startswith("open_ratio") or f.startswith("median_minimum_nights")]
REV = ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
       "review_scores_location", "review_scores_value", "review_count_365d", "review_available"]
fam = {**{f: "property" for f in PROPERTY}, **{f: "platform" for f in PLATFORM}, **{f: "geography" for f in GEO}, **{f: "rental" for f in RENT},
       **{f: "construction" for f in CONS}, **{f: "seasonality" for f in SEAS}, **{f: "calendar" for f in CAL}, **{f: "review_history" for f in REV}}
assert set(fam) == set(FEAT) and len(FEAT) == 53, set(FEAT) ^ set(fam)
counts = pd.Series(fam).value_counts().to_dict()

LISTING_SRC = "01_raw/airbnb/<snapshot>/listings.csv.gz"
SCRIPT_LIST = "04_code/02_clean_listings.py (clean_snapshot); 04_code/06_reviews_QC.ipynb cell 32 (STEP 24.8A-1 backbone)"
derived = {
    "amenities_count": ("amenities", "count of items in the amenities JSON list (02_clean_listings.py)"),
    "listing_review_age_days": ("last_scraped, first_review", "last_scraped - first_review in days (02_clean_listings.py lines ~278-285)"),
    "days_since_last_review": ("last_scraped, last_review", "last_scraped - last_review in days (02_clean_listings.py lines ~287-293)"),
}

# rental release evidence (MBIE Tenancy Services): files released in month M contain data to end of M-2
# observed release/availability evidence: 13 Feb 2026 (data to Dec 2025, Wayback page capture 2026-02-17);
# 13 May 2026 (data to Mar 2026, page capture 2026-06-11 + file capture 2026-05-13 22:19 UTC);
# file containing May 2026 captured 2026-07-12 21:22 UTC = 13 Jul 2026 09:22 NZST; 10 Sep 2026 (current page).
RENT_STATUS = {
    #      nominal cutoff                                                   completion cutoff
    "T1": ("UNVERIFIED (probable FAIL: Aug-2025 data first released in Oct-2025 file; observed release days are the 10th-13th)", "UNVERIFIED (release day of Oct-2025 file unknown; completion 13 Oct)"),
    "T2": ("UNVERIFIED (probable FAIL: Sep-2025 data in Nov-2025 file; origin 7 Nov)", "UNVERIFIED (probable PASS: completion 24 Nov)"),
    "T3": ("UNVERIFIED (origin 11 Dec; Dec-2025 file release day unknown)", "UNVERIFIED (probable PASS: completion 28 Dec)"),
    "T4": ("UNVERIFIED (origin 16 Jan; Jan-2026 file release day unknown)", "UNVERIFIED (probable PASS: completion 24 Jan)"),
    "T5": ("UNVERIFIED (Feb-2026 file released 13 Feb 2026 = origin date; time of day unknown)", "PASS (released 13 Feb 2026 <= completion 16 Feb)"),
    "T6": ("UNVERIFIED (probable PASS: origin 17 Mar)", "UNVERIFIED (probable PASS: completion 23 Mar)"),
    "T7": ("UNVERIFIED (probable PASS: origin 16 Apr)", "UNVERIFIED (probable PASS: completion 22 Apr)"),
    "T8": ("PASS (May-2026 file with Mar-2026 data released 13 May 2026 < 23 May)", "PASS"),
    "T9": ("UNVERIFIED (probable PASS: origin 19 Jun)", "UNVERIFIED (probable PASS: completion 23 Jun)"),
    "T10": ("UNVERIFIED (May-2026 data available by 13 Jul 2026 09:22 NZST; origin 12 Jul)", "PASS (available by 13 Jul 09:22 NZST < completion 16 Jul)"),
}
CONS_REL = pd.read_csv(ev / "construction_release_alignment_verified.csv").set_index("transition")

reg, pit = [], []
for f in FEAT:
    fa = fam[f]
    if fa in ("property", "platform"):
        src, col, scr = LISTING_SRC, derived.get(f, (f, ""))[0], SCRIPT_LIST
        tr = derived.get(f, ("", "copied from Inside Airbnb listings field after type cleaning"))[1]
        obs = "listing last_scraped (per listing)"; coll = "last_scraped: 0-17 days after nominal snapshot date (98.48% later)"
        pub = "Inside Airbnb snapshot publication (date not recorded in project)"; avail = "on or after listing last_scraped"
        aggregate_review = f in ("number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "reviews_per_month", "listing_review_age_days", "days_since_last_review")
        evid = "evidence/panel_lineage_by_transition.csv" + ("; evidence/review_timing_by_transition.csv (Inside Airbnb review aggregates are scrape-anchored: l30d matches scrape-anchored window for 95.4% vs nominal 87.2%; 25,284 origins include reviews dated after the nominal date)" if aggregate_review else "")
    elif fa == "geography":
        src, col, scr = LISTING_SRC + "; 01_raw/geography/statsnz-territorial-authority-2025-SHP", "latitude, longitude; TA2025_V1_00", "06_reviews_QC.ipynb cells 33-35 (STEP 24.8B)"
        tr = "point-in-polygon assignment to Stats NZ TA 2025 boundaries; code 999 = Area Outside TA"; obs = "listing last_scraped (coordinates)"
        coll = "as listings"; pub = "TA 2025 boundary release (Stats NZ; static classification)"; avail = "boundaries: before T1; coordinates: at last_scraped"
        evid = "evidence/panel_lineage_by_transition.csv (68 categories, 0 missing)"
    elif fa == "rental":
        src, col = "01_raw/property_market/detailed-monthly-tla-tenancy-september.csv (MBIE, file released 10 Sep 2026, data to Jul 2026)", "GeometricMeanRent by TimeFrame x location_id"
        scr = "04_code/04_property_market_QC.ipynb; output 02_intermediate/property_market/rental_market_features_ta_transition_frozen.parquet"
        tr = "log(geometric mean weekly rent) at origin month - 2" + (" ; minus log value 12 months earlier" if "change" in f else "")
        obs = "reference month = origin calendar month - 2"; coll = "bond lodgements by tenancy start date"
        pub = "monthly file; month-M file contains data to end of M-2; release day 10th-13th where observed"; avail = "see transition-specific status"
        evid = "evidence/rental_features_recomputed.csv (max abs diff 0); evidence/rental_vintage_revision_check.csv; external_evidence/mbie_rental_bond_archived_vintages/"
    elif fa == "construction":
        src, col = "01_raw/construction/statsnz_building_consents_ta_monthly_2020M01_2026M07.csv (Stats NZ, extract downloaded 1 Oct 2026)", "New dwelling units / houses / apartments / townhouses etc., Number"
        scr = "04_code/03_build_construction_temporal_alignment.py; 04_code/03_construction_QC.ipynb (STEP 23.x)"
        tr = {"construction_dwelling_3m_log": "log1p(sum of new dwelling units, 3 months ending at latest released month)",
              "construction_dwelling_12m_log": "log1p(sum, 12 months ending at latest released month)",
              "construction_momentum_3m_vs_prior9m": "log1p(3m monthly pace) - log1p(prior 9m monthly pace)",
              "construction_multiunit_share_12m": "(apartments + townhouses/flats/units/other) / dwelling units, 12 months"}[f]
        obs = "consent reference month 2025M08-2026M05 (T1-T10)"; coll = "administrative consents issued"
        pub = "Stats NZ 'Building consents issued' information release, 10:45 NZT on dates in construction_release_alignment_verified.csv"
        avail = "official release date <= nominal origin for all T1-T10; latest released month used"
        evid = "evidence/construction_release_alignment_verified.csv; evidence/construction_features_recomputed.csv (max abs diff 0); external_evidence/statsnz_building_consents_release_pages.json"
    elif fa == "seasonality":
        if f in ("month_sin", "month_cos"):
            src, col, tr = "panel anchor_date", "anchor month", f"{f[-3:]}(2*pi*origin_month/12)"
            obs = avail = "origin calendar month"; coll = pub = "n/a (deterministic calendar)"; evid = "evidence/month_cyclical_check.csv"
        elif f == "forecast_horizon_days":
            src, col, tr = "config SNAPSHOT_DATES", "anchor_date, future_date", "future_snapshot_date - origin_snapshot_date (days)"
            obs = "requires the NEXT snapshot date"; coll = "n/a"; pub = "next Inside Airbnb snapshot (23-37 days later)"; avail = "only after the next snapshot exists"
            evid = "05_seasonality_QC.ipynb STEP 23.11-23.12; evidence/panel_lineage_by_transition.csv (horizon_equals_future_minus_anchor = True)"
        else:
            src, col = "01_raw/seasonality/ADP_All_Measures.csv (Accommodation Data Programme; extract downloaded 4 Oct 2026)", "monthly measures 2023-2024 by area"
            tr = "day-weighted mean of the TA (or NZ fallback) 2023-2024 monthly seasonal index over [origin date, next snapshot date)"
            obs = "reference years 2023-2024; weights depend on next snapshot date"; coll = "ADP monthly"; pub = "historical months published before T1; vintage downloaded Oct 2026"
            avail = "profile: before T1; horizon weights: only after next snapshot date is known"
            evid = "05_seasonality_QC.ipynb STEP 23.11-23.12; evidence/seasonality_features_recomputed.csv (max abs diff 0)"
        scr = "04_code/05_seasonality_QC.ipynb; 06_reviews_QC.ipynb cell 47 (Block H, month_sin/cos)"
    elif fa == "calendar":
        src, col = "01_raw/airbnb/<snapshot>/calendar.csv.gz", "date, available, minimum_nights"
        scr = "04_code/06_reviews_QC.ipynb cell 45 (Production Block F) -> 02_intermediate/calendar/final_supported/"
        tr = "first 28 observed calendar dates per listing; w1-w4 = 7-day blocks; open ratio = share available; median minimum nights"
        obs = "calendar capture (calendar_last_scraped, after nominal date)"; coll = "first calendar date = scrape date (all origins except partial T3)"
        pub = "Inside Airbnb snapshot"; avail = "at calendar capture"; evid = "evidence/panel_lineage_by_transition.csv (calendar_first_date_* columns); SNAPSHOT_MANIFEST.csv (calendar price fields empty)"
    else:  # review history
        if f in ("review_count_365d", "review_available"):
            src, col = "01_raw/airbnb/<snapshot>/reviews.csv.gz (origin snapshot only)", "listing_id, id, date"
            scr = "04_code/06_reviews_QC.ipynb cell 11 (STEP 24.6A temporal membership, window (anchor-365d, anchor])"
            tr = "count of origin-file reviews dated in (nominal origin - 365 d, nominal origin]; available = count > 0"
            obs = "review date <= nominal origin"; coll = "origin snapshot review file"; pub = "Inside Airbnb snapshot"; avail = "review dates precede nominal origin"
            evid = "evidence/review_timing_by_transition.csv (99.89% reproduced from raw nominal window)"
        else:
            src, col, scr = LISTING_SRC, f, SCRIPT_LIST
            tr = "Inside Airbnb aggregate review score at scrape time"; obs = "listing last_scraped"; coll = "as listings"; pub = "Inside Airbnb snapshot"
            avail = "at last_scraped"; evid = "evidence/review_timing_by_transition.csv (scrape-anchored aggregates)"
    reg.append(dict(feature=f, family=fa, source_file=src, source_columns=col, transformation=tr, code_location=scr,
                    observation_date=obs, collection_date=coll, official_publication=pub, reasonable_availability=avail, evidence=evid))

    for t in T:
        lagp = LAG[t]
        if fa in ("property", "platform", "calendar") or (fa == "review_history" and f not in ("review_count_365d", "review_available")) or fa == "geography":
            if fa == "geography":
                n_ = "UNVERIFIED (coordinates observed at scrape after nominal date; TA assignment of a fixed listing is time-invariant)"
            else:
                n_ = "FAIL" if lagp == 100 else f"FAIL for {lagp}% of origins scraped after nominal date; PASS for remainder"
            c_ = "PASS"
            why = f"captured at listing/calendar scrape {lagp}% after nominal; origin scrape completed {COMP[t]} before next snapshot"
        elif f in ("review_count_365d", "review_available"):
            n_, c_ = "PASS", "PASS"; why = "only reviews dated <= nominal origin from the origin snapshot file"
        elif fa == "construction":
            r = CONS_REL.loc[t]
            n_ = c_ = "PASS" if bool(r.released_on_or_before_origin) else "FAIL"
            why = f"consent month {r.consent_month_used} released {r.official_release_nzt} NZT; values from Oct-2026 extract (revision effect not verified)"
        elif fa == "rental":
            n_, c_ = RENT_STATUS[t]; why = "MBIE release timing (see evidence); values taken from Sep-2026 vintage, which revises earlier vintages"
        elif f in ("month_sin", "month_cos"):
            n_ = c_ = "PASS"; why = "origin calendar month"
        else:
            n_ = c_ = "FAIL"; why = "value depends on the date of the next snapshot, which is unknown at the origin (horizon 23-37 days)"
        pit.append(dict(feature=f, family=fa, transition=t, nominal_origin=NOM[t], scrape_completion_origin=COMP[t],
                        status_nominal_cutoff=n_, status_completion_cutoff=c_, evidence_note=why,
                        vintage_note={"rental": "RETROSPECTIVE VINTAGE (Sep-2026 file; 22-82% of TA-month values differ from earlier vintages)",
                                      "construction": "RETROSPECTIVE VINTAGE (Oct-2026 extract; revisions not quantified)",
                                      "seasonality": "RETROSPECTIVE VINTAGE (Oct-2026 ADP extract of 2023-2024 months)" if f == "historical_seasonality_index_horizon" else ""}.get(fa, "")))
pd.DataFrame(reg).to_csv(rep / "FEATURE_PROVENANCE_REGISTER.csv", index=False)
pit = pd.DataFrame(pit); pit.to_csv(aud / "POINT_IN_TIME_AUDIT.csv", index=False)
summ = pit.assign(n=pit.status_nominal_cutoff.str.split(" ").str[0], c=pit.status_completion_cutoff.str.split(" ").str[0])
tab = pd.concat([summ.groupby(["transition", "c"]).size().unstack(fill_value=0).add_prefix("completion_"),
                 summ.groupby(["transition", "n"]).size().unstack(fill_value=0).add_prefix("nominal_")], axis=1).reindex(T)
tab.to_csv(ev / "point_in_time_status_counts.csv"); print(counts); print(tab.to_string())
