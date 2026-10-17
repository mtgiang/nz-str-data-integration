"""Phase 1b - raw Inside Airbnb snapshot manifest and review/calendar timing evidence (read-only).

Usage: python -I audit_02_raw_snapshots.py <PROJECT_ROOT> <OUT_DIR>
Outputs (OUT_DIR):
  SNAPSHOT_MANIFEST.csv                       one row per snapshot x file
  evidence/raw_review_counts_by_listing.parquet  per snapshot/listing review counts in audit windows
  evidence/raw_listing_scrape_fields.parquet     listing_id, last_scraped, calendar_last_scraped per snapshot
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]).resolve()
(out / "evidence").mkdir(parents=True, exist_ok=True)
SNAPS = sorted(p.name for p in (root / "01_raw" / "airbnb").iterdir() if p.is_dir())
manifest, rev_frames, scrape_frames = [], [], []

for s in SNAPS:
    nominal = pd.Timestamp(s)
    d = root / "01_raw" / "airbnb" / s
    print("snapshot", s, flush=True)

    # ---------------- listings ----------------
    L = pd.read_csv(d / "listings.csv.gz", low_memory=False)
    ls = pd.to_datetime(L["last_scraped"], errors="coerce")
    cls = pd.to_datetime(L["calendar_last_scraped"], errors="coerce") if "calendar_last_scraped" in L else pd.Series(pd.NaT, index=L.index)
    price_nonmissing = L["price"].notna().sum() if "price" in L else 0
    manifest.append({
        "snapshot": s, "file": "listings.csv.gz", "rows": len(L), "columns": L.shape[1],
        "unique_listing_id": L["id"].nunique(), "duplicate_listing_id": int(L["id"].duplicated().sum()),
        "last_scraped_min": ls.min().date(), "last_scraped_median": ls.median().date(), "last_scraped_max": ls.max().date(),
        "rows_last_scraped_after_nominal": int((ls > nominal).sum()), "rows_last_scraped_before_nominal": int((ls < nominal).sum()),
        "calendar_last_scraped_min": cls.min().date() if cls.notna().any() else None,
        "calendar_last_scraped_max": cls.max().date() if cls.notna().any() else None,
        "price_nonmissing": int(price_nonmissing),
        "cleaned_parquet_rows": len(pd.read_parquet(root / "02_intermediate" / "listings" / f"listings_{s}.parquet", columns=["listing_id"])),
    })
    scrape_frames.append(pd.DataFrame({"snapshot": s, "listing_id": L["id"].astype("int64"), "last_scraped": ls,
                                       "calendar_last_scraped": cls,
                                       "number_of_reviews_l30d": L.get("number_of_reviews_l30d"),
                                       "number_of_reviews_ltm": L.get("number_of_reviews_ltm"),
                                       "number_of_reviews": L.get("number_of_reviews")}))

    # ---------------- calendar ----------------
    C = pd.read_csv(d / "calendar.csv.gz", usecols=lambda c: c in {"listing_id", "date", "available", "price", "adjusted_price", "minimum_nights"}, low_memory=False)
    cd = pd.to_datetime(C["date"], errors="coerce")
    manifest.append({
        "snapshot": s, "file": "calendar.csv.gz", "rows": len(C), "columns": None,
        "unique_listing_id": C["listing_id"].nunique(), "duplicate_listing_id": None,
        "calendar_date_min": cd.min().date(), "calendar_date_max": cd.max().date(),
        "calendar_rows_dated_before_nominal": int((cd < nominal).sum()),
        "calendar_price_nonmissing": int(C["price"].notna().sum()) if "price" in C else 0,
        "calendar_adjusted_price_nonmissing": int(C["adjusted_price"].notna().sum()) if "adjusted_price" in C else 0,
    })
    del C

    # ---------------- reviews ----------------
    R = pd.read_csv(d / "reviews.csv.gz", usecols=["listing_id", "id", "date"], low_memory=False)
    rd = pd.to_datetime(R["date"], errors="coerce")
    max_ls = ls.max()
    manifest.append({
        "snapshot": s, "file": "reviews.csv.gz", "rows": len(R), "columns": None,
        "unique_listing_id": R["listing_id"].nunique(), "duplicate_review_id": int(R["id"].duplicated().sum()),
        "review_date_min": rd.min().date(), "review_date_max": rd.max().date(),
        "reviews_dated_after_nominal": int((rd > nominal).sum()),
        "reviews_dated_after_max_last_scraped": int((rd > max_ls).sum()),
    })
    # per-listing counts in audit windows; joined later to listing-level last_scraped
    R = R.assign(date=rd).merge(pd.DataFrame({"listing_id": L["id"].astype("int64"), "ls": ls}), on="listing_id", how="left")
    g = pd.DataFrame({
        "listing_id": R["listing_id"],
        "in_365_nominal": (R["date"] > nominal - pd.Timedelta(days=365)) & (R["date"] <= nominal),
        "in_30_nominal": (R["date"] > nominal - pd.Timedelta(days=30)) & (R["date"] <= nominal),
        "in_30_scrape": (R["date"] > R["ls"] - pd.Timedelta(days=30)) & (R["date"] <= R["ls"]),
        "after_nominal": R["date"] > nominal,
    }).groupby("listing_id").sum().reset_index()
    g.insert(0, "snapshot", s)
    rev_frames.append(g)
    del R, L

pd.DataFrame(manifest).to_csv(out / "SNAPSHOT_MANIFEST.csv", index=False)
pd.concat(rev_frames).to_parquet(out / "evidence" / "raw_review_counts_by_listing.parquet", index=False)
pd.concat(scrape_frames).to_parquet(out / "evidence" / "raw_listing_scrape_fields.parquet", index=False)
print("done")
