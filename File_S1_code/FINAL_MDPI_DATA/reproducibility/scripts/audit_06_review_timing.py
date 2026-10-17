"""Phase 2 (channel B) - review-derived predictors versus the nominal origin and the listing scrape time.

Usage: python -I audit_06_review_timing.py <PROJECT_ROOT> <AUDIT_DIR>
Needs AUDIT_DIR/evidence/raw_review_counts_by_listing.parquet and raw_listing_scrape_fields.parquet (audit_02).
"""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]).resolve(); ev = Path(sys.argv[2]).resolve() / "evidence"
SN = ["2025-10-05", "2025-11-07", "2025-12-11", "2026-01-16", "2026-02-13", "2026-03-17", "2026-04-16", "2026-05-23", "2026-06-19", "2026-07-12"]
tmap = {s: f"T{i+1}" for i, s in enumerate(SN)}
rc = pd.read_parquet(ev / "raw_review_counts_by_listing.parquet"); rc = rc[rc.snapshot.isin(SN)]; rc["transition"] = rc.snapshot.map(tmap)
sf = pd.read_parquet(ev / "raw_listing_scrape_fields.parquet"); sf = sf[sf.snapshot.isin(SN)]; sf["transition"] = sf.snapshot.map(tmap)
P = pd.read_parquet(root / "02_intermediate" / "modelling" / "modelling_panel_T1_T10.parquet", columns=["transition", "listing_id", "review_count_365d", "review_available", "number_of_reviews_l30d"])
P["listing_id"] = P.listing_id.astype("int64")
m = P.merge(rc, on=["transition", "listing_id"], how="left").merge(sf[["transition", "listing_id", "number_of_reviews_l30d"]], on=["transition", "listing_id"], how="left", suffixes=("", "_raw"))
for c in ["in_365_nominal", "in_30_nominal", "in_30_scrape", "after_nominal"]:
    m[c] = m[c].fillna(0).astype(int)
out = m.groupby("transition").apply(lambda g: pd.Series({
    "origins": len(g),
    "review_count_365d_equals_raw_nominal_window": int((g.review_count_365d == g.in_365_nominal).sum()),
    "origins_with_reviews_after_nominal_in_origin_file": int((g.after_nominal > 0).sum()),
    "reviews_after_nominal_total": int(g.after_nominal.sum()),
    "l30d_matches_scrape_anchored_window": int((g.number_of_reviews_l30d == g.in_30_scrape).sum()),
    "l30d_matches_nominal_anchored_window": int((g.number_of_reviews_l30d == g.in_30_nominal).sum()),
})).reset_index()
out.to_csv(ev / "review_timing_by_transition.csv", index=False)
S = {"review_count_365d_match_rate": float(out.review_count_365d_equals_raw_nominal_window.sum() / out.origins.sum()),
     "origins_with_post_nominal_reviews": int(out.origins_with_reviews_after_nominal_in_origin_file.sum()),
     "l30d_scrape_anchored_match_rate": float(out.l30d_matches_scrape_anchored_window.sum() / out.origins.sum()),
     "l30d_nominal_anchored_match_rate": float(out.l30d_matches_nominal_anchored_window.sum() / out.origins.sum())}
json.dump(S, open(ev / "review_timing_summary.json", "w"), indent=1)
print(out.to_string()); print(json.dumps(S, indent=1))
