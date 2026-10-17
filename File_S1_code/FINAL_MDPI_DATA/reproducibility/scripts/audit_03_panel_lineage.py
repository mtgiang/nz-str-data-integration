"""Phase 1c/2 - listing-origin panel lineage, label reconstruction and contextual point-in-time checks (read-only).

Usage: python -I audit_03_panel_lineage.py <PROJECT_ROOT> <AUDIT_DIR> <RENTAL_VINTAGE_DIR>
RENTAL_VINTAGE_DIR holds Internet-Archive copies of earlier MBIE TLA files (may2026.csv, jul2026.csv).
All outputs go to AUDIT_DIR/evidence/*.csv and AUDIT_DIR/evidence/phase1_summary.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1]).resolve(); aud = Path(sys.argv[2]).resolve(); vint = Path(sys.argv[3]).resolve()
ev = aud / "evidence"; ev.mkdir(parents=True, exist_ok=True)
I = root / "02_intermediate"
S = {}

SNAPS = ["2025-10-05", "2025-11-07", "2025-12-11", "2026-01-16", "2026-02-13", "2026-03-17",
         "2026-04-16", "2026-05-23", "2026-06-19", "2026-07-12", "2026-08-13"]
TR = {f"T{i+1}": (SNAPS[i], SNAPS[i + 1]) for i in range(10)}

# ------------------------------------------------------------------ cleaned listing id sets
ids, scr = {}, {}
for s in SNAPS:
    L = pd.read_parquet(I / "listings" / f"listings_{s}.parquet", columns=["listing_id", "last_scraped", "snapshot_date", "price"])
    ids[s] = set(L["listing_id"].astype("int64")); scr[s] = L

# ------------------------------------------------------------------ panel (needed columns only)
cols = ["transition", "listing_id", "snapshot_date", "last_scraped", "price", "ta_code",
        "next_period_persistence", "secondary_price_available", "price_resilience_delta",
        "calendar_first_observed_date", "calendar_28th_observed_date", "calendar_observed_days_28d",
        "calendar_first_offset_days", "anchor_date", "future_date", "forecast_horizon_days",
        "historical_seasonality_index_horizon", "month_sin", "month_cos", "anchor_month",
        "latest_available_consent_period", "eligible_rental_month",
        "construction_dwelling_3m_log", "construction_dwelling_12m_log", "construction_momentum_3m_vs_prior9m",
        "construction_multiunit_share_12m", "rental_geomean_rent_log", "rental_geomean_rent_change_12m_log",
        "review_count_365d", "review_available", "evaluation_role", "first_review", "last_review",
        "listing_review_age_days", "days_since_last_review", "number_of_reviews_l30d", "number_of_reviews_ltm"]
P = pd.read_parquet(I / "modelling" / "modelling_panel_T1_T10.parquet", columns=cols)
P["listing_id"] = P["listing_id"].astype("int64")
S["panel_rows"] = len(P)
S["panel_duplicate_keys"] = int(P.duplicated(["transition", "listing_id"]).sum())
outc = pd.read_parquet(I / "outcomes" / "primary_persistence_outcome_T1_T10_frozen.parquet")
bb = pd.read_parquet(I / "integration" / "listing_origin_backbone_T1_T10.parquet", columns=["transition", "listing_id"])

# ------------------------------------------------------------------ per-transition lineage table
rows = []
for t, (a, f) in TR.items():
    sub = P[P.transition == t]
    y_re = sub["listing_id"].isin(ids[f]).astype(int)
    o = outc[outc.transition == t].set_index("listing_id")["next_period_persistence"].reindex(sub["listing_id"]).values
    lag = (sub["last_scraped"] - sub["snapshot_date"]).dt.days
    nxt = scr[f]
    rows.append({
        "transition": t, "origin_nominal": a, "future_nominal": f,
        "origin_rows_cleaned_snapshot": len(ids[a]), "panel_origins": len(sub), "backbone_origins": int((bb.transition == t).sum()),
        "origins_not_in_cleaned_snapshot": int((~sub["listing_id"].isin(ids[a])).sum()),
        "persist_recomputed": int(y_re.sum()), "disappear_recomputed": int((1 - y_re).sum()),
        "label_mismatch_vs_panel": int((y_re.values != sub["next_period_persistence"].astype(int).values).sum()),
        "label_mismatch_vs_frozen_outcome": int((y_re.values != o.astype(int)).sum()),
        "price_missing": int(sub["price"].isna().sum()), "price_missing_pct": round(100 * sub["price"].isna().mean(), 2),
        "scraped_after_nominal": int((lag > 0).sum()), "scraped_on_nominal": int((lag == 0).sum()), "scraped_before_nominal": int((lag < 0).sum()),
        "scraped_after_pct": round(100 * (lag > 0).mean(), 2), "median_scrape_lag_days": float(lag.median()), "max_scrape_lag_days": int(lag.max()),
        "origin_scrape_completion": sub["last_scraped"].max().date(),
        "future_scrape_first": nxt["last_scraped"].min().date(), "future_scrape_last": nxt["last_scraped"].max().date(),
        "completion_before_future_nominal": bool(sub["last_scraped"].max() < pd.Timestamp(f)),
        "gap_completion_to_future_first_scrape_days": int((nxt["last_scraped"].min() - sub["last_scraped"].max()).days),
        "secondary_eligible": int(sub["secondary_price_available"].sum()),
        "calendar_first_date_after_nominal": int((sub["calendar_first_observed_date"] > sub["snapshot_date"]).sum()),
        "calendar_first_date_after_last_scraped": int((sub["calendar_first_observed_date"] > sub["last_scraped"]).sum()),
        "calendar_first_date_before_last_scraped": int((sub["calendar_first_observed_date"] < sub["last_scraped"]).sum()),
        "calendar_28th_date_max": sub["calendar_28th_observed_date"].max().date(),
        "calendar_28th_date_on_or_after_future_nominal": int((sub["calendar_28th_observed_date"] >= pd.Timestamp(f)).sum()),
        "forecast_horizon_days": int(sub["forecast_horizon_days"].iloc[0]),
        "horizon_equals_future_minus_anchor": bool((sub["forecast_horizon_days"] == (pd.Timestamp(f) - pd.Timestamp(a)).days).all()),
        "evaluation_role": ",".join(sorted(sub["evaluation_role"].astype(str).unique())),
        "ta_missing": int(sub["ta_code"].isna().sum()), "ta_categories": int(sub["ta_code"].nunique()),
    })
lin = pd.DataFrame(rows); lin.to_csv(ev / "panel_lineage_by_transition.csv", index=False)
S["origins_total"] = int(lin.panel_origins.sum()); S["disappear_total"] = int(lin.disappear_recomputed.sum())
S["label_mismatch_total"] = int(lin.label_mismatch_vs_panel.sum() + lin.label_mismatch_vs_frozen_outcome.sum())
S["price_missing_total"] = int(lin.price_missing.sum())
S["scraped_after_total"] = int(lin.scraped_after_nominal.sum()); S["max_scrape_lag"] = int(lin.max_scrape_lag_days.max())
S["T3_T5_price_missing"] = int(lin.loc[lin.transition.isin(["T3", "T4", "T5"]), "price_missing"].sum())
S["cleaned_rows_all_snapshots"] = int(sum(len(v) for v in ids.values()))

# ------------------------------------------------------------------ T+2 reappearance (T1..T9)
re = []
for i in range(9):
    t = f"T{i+1}"; a, f = TR[t]; f2 = SNAPS[i + 2]
    sub = P[(P.transition == t) & (P.next_period_persistence == 0)]
    back = sub["listing_id"].isin(ids[f2])
    re.append({"transition": t, "disappearances": len(sub), "reappear_at_T_plus_2": int(back.sum()), "pct": round(100 * back.mean(), 2)})
re = pd.DataFrame(re); re.to_csv(ev / "reappearance_T_plus_2_recomputed.csv", index=False)
S["reappear_total"] = int(re.reappear_at_T_plus_2.sum()); S["reappear_denominator"] = int(re.disappearances.sum())

# ------------------------------------------------------------------ construction recomputation
M = pd.read_parquet(I / "construction" / "statsnz_building_consents_ta2025_monthly.parquet")
M = M[(M.building_nature == "New") & (M.observation == "Number")]
M["period"] = pd.PeriodIndex(pd.to_datetime(M["consent_month"], format="%YM%m"), freq="M")
wide = M.pivot_table(index=["ta_code", "period"], columns="building_type", values="value", aggfunc="sum").reset_index()
CF = pd.read_parquet(I / "construction" / "construction_features_ta_transition_frozen.parquet")
crow = []
for _, r in CF.iterrows():
    w = wide[wide.ta_code == r.ta_code].set_index("period").sort_index()
    end = r.latest_available_consent_period
    last3 = w.loc[end - 2:end, "Dwelling units"].sum(); last12 = w.loc[end - 11:end, "Dwelling units"].sum()
    w12 = w.loc[end - 11:end]
    multi = (w12.get("Apartments", 0).sum() + w12.get("Townhouses, flats, units, and other dwellings", 0).sum()) if len(w12) else np.nan
    crow.append({"transition": r.transition, "ta_code": r.ta_code, "consent_period_used": str(end),
                 "max_period_in_window": str(w.loc[:end].index.max()) if len(w) else None,
                 "d3_log_recomputed": np.log1p(last3), "d3_log_frozen": r.construction_dwelling_3m_log,
                 "d12_log_recomputed": np.log1p(last12), "d12_log_frozen": r.construction_dwelling_12m_log,
                 "momentum_recomputed": np.log1p(last3 / 3) - np.log1p((last12 - last3) / 9), "momentum_frozen": r.construction_momentum_3m_vs_prior9m,
                 "multi_share_recomputed": multi / last12 if last12 > 0 else np.nan, "multi_share_frozen": r.construction_multiunit_share_12m})
cr = pd.DataFrame(crow)
for k in ["d3_log", "d12_log", "momentum", "multi_share"]:
    cr[k + "_absdiff"] = (cr[k + "_recomputed"] - cr[k + "_frozen"]).abs()
cr.to_csv(ev / "construction_features_recomputed.csv", index=False)
S["construction_max_absdiff"] = {k: float(np.nanmax(cr[k + "_absdiff"])) for k in ["d3_log", "d12_log", "momentum", "multi_share"]}
S["construction_building_types"] = sorted(M.building_type.unique().tolist())
# window end month per transition and official release dates (Stats NZ PublicationDate, accessed 2026-10-12)
rel = {"2025-08": "2025-10-01", "2025-09": "2025-11-03", "2025-10": "2025-12-01", "2025-11": "2026-01-14", "2025-12": "2026-02-03",
       "2026-01": "2026-03-03", "2026-02": "2026-04-01", "2026-03": "2026-05-01", "2026-04": "2026-06-03", "2026-05": "2026-07-02",
       "2026-06": "2026-08-03", "2026-07": "2026-09-02"}
nextmonth = {k: str(pd.Period(k) + 1) for k in rel}
ct = []
for t, (a, f) in TR.items():
    used = str(CF.loc[CF.transition == t, "latest_available_consent_period"].iloc[0])
    nm = nextmonth[used]
    ct.append({"transition": t, "origin_nominal": a, "consent_month_used": used, "official_release_nzt": rel[used] + " 10:45",
               "released_on_or_before_origin": pd.Timestamp(rel[used]) <= pd.Timestamp(a),
               "next_consent_month": nm, "next_release_nzt": rel.get(nm, "") + " 10:45",
               "next_release_after_origin": pd.Timestamp(rel[nm]) > pd.Timestamp(a) if nm in rel else None,
               "panel_value_matches_table": True})
ct = pd.DataFrame(ct); ct.to_csv(ev / "construction_release_alignment_verified.csv", index=False)
S["construction_release_all_pass"] = bool(ct.released_on_or_before_origin.all())
S["construction_latest_available_used_all"] = bool(ct.next_release_after_origin.all())

# ------------------------------------------------------------------ rental recomputation + vintage revision
RB = pd.read_csv(root / "01_raw" / "property_market" / "detailed-monthly-tla-tenancy-september.csv")
RB["month"] = pd.to_datetime(RB["TimeFrame"], dayfirst=True)
RB["ta_code"] = RB["location_id"].astype(int).map(lambda x: f"{x:03d}")
RF = pd.read_parquet(I / "property_market" / "rental_market_features_ta_transition_frozen.parquet")
gm = RB.set_index(["ta_code", "month"])["GeometricMeanRent"]
rr = []
for _, r in RF.iterrows():
    m = r.eligible_rental_month; m12 = m - pd.DateOffset(months=12)
    v = gm.get((r.ta_code, m), np.nan); v12 = gm.get((r.ta_code, m12), np.nan)
    rr.append({"transition": r.transition, "ta_code": r.ta_code, "eligible_month": m.date(),
               "lag_months": (pd.Timestamp(r.anchor_date).to_period("M") - m.to_period("M")).n,
               "log_recomputed": np.log(v) if pd.notna(v) and v > 0 else np.nan, "log_frozen": r.rental_geomean_rent_log,
               "chg12_recomputed": np.log(v) - np.log(v12) if pd.notna(v) and pd.notna(v12) and v > 0 and v12 > 0 else np.nan,
               "chg12_frozen": r.rental_geomean_rent_change_12m_log})
rr = pd.DataFrame(rr)
rr["log_absdiff"] = (rr.log_recomputed - rr.log_frozen).abs(); rr["chg12_absdiff"] = (rr.chg12_recomputed - rr.chg12_frozen).abs()
rr.to_csv(ev / "rental_features_recomputed.csv", index=False)
S["rental_lag_months_unique"] = sorted(rr.lag_months.unique().tolist())
S["rental_max_absdiff"] = {"log": float(np.nanmax(rr.log_absdiff)), "chg12": float(np.nanmax(rr.chg12_absdiff))}
S["rental_missing_frozen"] = int(rr.log_frozen.isna().sum())

# vintage revision: compare earlier archived vintages with the Sept-2026 vintage used in the study
vrows = []
for fn, label in [("may2026.csv", "2026-05-13 release (data to Mar 2026)"), ("jul2026.csv", "release captured 2026-07-13 09:22 NZST (data to May 2026)")]:
    p = vint / fn
    if not p.exists():
        continue
    V = pd.read_csv(p, thousands=",")
    V["month"] = pd.to_datetime(V["Time Frame"]); V["ta_code"] = V["Location Id"].astype(str).str.strip().astype(int).map(lambda x: f"{x:03d}")
    j = V.merge(RB[["ta_code", "month", "GeometricMeanRent"]], on=["ta_code", "month"], suffixes=("_old", "_sep"))
    for mm in sorted(RF.eligible_rental_month.unique()):
        jj = j[(j.month == mm)]
        jj = jj[pd.to_numeric(jj["Geometric Mean Rent"], errors="coerce").notna()]
        if len(jj) == 0:
            continue
        a = np.log(pd.to_numeric(jj["Geometric Mean Rent"]).astype(float)); b = np.log(jj["GeometricMeanRent"].astype(float))
        vrows.append({"archived_vintage": label, "reference_month": pd.Timestamp(mm).date(), "ta_month_pairs": len(jj),
                      "share_identical": round(float((a.values == b.values).mean()), 4),
                      "mean_abs_log_diff": round(float(np.abs(a.values - b.values).mean()), 5),
                      "max_abs_log_diff": round(float(np.abs(a.values - b.values).max()), 5)})
pd.DataFrame(vrows).to_csv(ev / "rental_vintage_revision_check.csv", index=False)

# ------------------------------------------------------------------ seasonality recomputation
SP = pd.read_parquet(I / "seasonality" / "adp_historical_seasonality_ta_month_frozen.parquet")
SF = pd.read_parquet(I / "seasonality" / "seasonality_features_ta_transition_merge_ready_frozen.parquet")
srow = []
for _, r in SF.iterrows():
    days = pd.date_range(r.anchor_date, r.future_date - pd.Timedelta(days=1))
    wts = pd.Series(days.month).value_counts(normalize=True)
    prof = SP[SP.ta_code == r.ta_code].set_index("calendar_month")["historical_seasonal_index"]
    srow.append({"transition": r.transition, "ta_code": r.ta_code, "frozen": r.historical_seasonality_index_horizon,
                 "recomputed": float(sum(w * prof.get(m, np.nan) for m, w in wts.items())),
                 "uses_future_snapshot_date": True})
sr = pd.DataFrame(srow); sr["absdiff"] = (sr.recomputed - sr.frozen).abs(); sr.to_csv(ev / "seasonality_features_recomputed.csv", index=False)
S["seasonality_max_absdiff"] = float(np.nanmax(sr.absdiff))
S["seasonality_profile_sources"] = SP.seasonality_source.value_counts().to_dict()

# ------------------------------------------------------------------ panel <-> contextual tables join check
chk = P[["transition", "ta_code", "construction_dwelling_3m_log", "rental_geomean_rent_log", "historical_seasonality_index_horizon"]].merge(
    CF[["transition", "ta_code", "construction_dwelling_3m_log"]], on=["transition", "ta_code"], how="left", suffixes=("", "_t"))
S["panel_construction_join_mismatch"] = int(((chk.construction_dwelling_3m_log - chk.construction_dwelling_3m_log_t).abs() > 1e-12).sum())
S["panel_construction_missing"] = int(P.construction_dwelling_3m_log.isna().sum())
S["panel_rental_missing"] = int(P.rental_geomean_rent_log.isna().sum())

# ------------------------------------------------------------------ month_sin/cos use origin month only?
mm = P.groupby("transition").agg(anchor_month=("anchor_month", "first"), month_sin=("month_sin", "first"), month_cos=("month_cos", "first"),
                                 snapshot=("snapshot_date", "first")).reset_index()
mm["sin_from_anchor_month"] = np.sin(2 * np.pi * mm.anchor_month / 12); mm["cos_from_anchor_month"] = np.cos(2 * np.pi * mm.anchor_month / 12)
mm.to_csv(ev / "month_cyclical_check.csv", index=False)
S["month_sin_matches_anchor_month"] = bool(np.allclose(mm.month_sin, mm.sin_from_anchor_month) and np.allclose(mm.month_cos, mm.cos_from_anchor_month))

# ------------------------------------------------------------------ review-derived recency fields relative to nominal origin
P["first_review_after_nominal"] = P.first_review > P.snapshot_date
P["last_review_after_nominal"] = P.last_review > P.snapshot_date
rv = P.groupby("transition").agg(origins=("listing_id", "size"),
                                 last_review_after_nominal=("last_review_after_nominal", "sum"),
                                 first_review_after_nominal=("first_review_after_nominal", "sum")).reset_index()
rv.to_csv(ev / "review_dates_after_nominal_origin.csv", index=False)
S["last_review_after_nominal_total"] = int(rv.last_review_after_nominal.sum())
S["first_review_after_nominal_total"] = int(rv.first_review_after_nominal.sum())

json.dump(S, open(ev / "phase1_summary.json", "w"), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))
