#!/usr/bin/env python
"""
One-command, point-in-time-conservative pipeline for the NZ short-term-rental paper.

    raw Inside Airbnb + Stats NZ + MBIE + MCERT files
      -> manifest & audits -> listing-origin panel & labels -> TA join
      -> listing / calendar / review / construction / rental / seasonality features
      -> rolling-origin development (T4-T9) -> frozen choices -> one locked T10 evaluation
      -> tables (CSV), figures (PDF), results_macros.tex for the manuscript

Every number that goes into the paper is computed here from raw files and written to
<out_dir>/results_macros.tex, so the manuscript never contains hand-copied numbers.

Usage (from the project root, i.e. the folder containing 01_raw/):
    python 09_pipeline/run_all.py                 # run everything, reuse cached stages
    python 09_pipeline/run_all.py --force panel   # rebuild from a stage onward
Stages: manifest, panel, features, models
"""
import argparse, hashlib, json, re, sys, time, unicodedata, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
STAGES = ["manifest", "panel", "features", "models"]
CFG, OUT = None, None


# ----------------------------------------------------------------------------- utils
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_file(folder, stem):
    hits = sorted(Path(folder).glob(stem + "*"))
    if not hits:
        raise FileNotFoundError(f"{stem}* not found in {folder}")
    return hits[0]


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(district|city|council|territory|territorial authority)\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def num(s):
    return pd.to_numeric(pd.Series(s, dtype="object").astype(str)
                         .str.replace(r"[$,%]", "", regex=True), errors="coerce").values


def tf(s):
    return pd.Series(s, dtype="object").map({"t": 1.0, "f": 0.0, "True": 1.0, "False": 0.0}).astype(float).values


def read_csv_any(p, **kw):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise


def save(df, name):
    df.to_parquet(OUT / f"{name}.parquet", index=False)


def load(name):
    return pd.read_parquet(OUT / f"{name}.parquet")


def snapshots():
    d = Path(CFG["paths"]["airbnb_dir"])
    snaps = sorted(p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    if len(snaps) < 3:
        raise SystemExit(f"Expected dated snapshot folders in {d}, found {len(snaps)}")
    return snaps


# ----------------------------------------------------------------------------- stage 1
def stage_manifest():
    rows = []
    for s in snapshots():
        for kind in ("listings", "calendar", "reviews"):
            p = find_file(s, kind + ".csv")
            rows.append({"snapshot": s.name, "file": kind, "path": p.as_posix(),
                         "bytes": p.stat().st_size, "sha256": sha256(p)})
    for key in ("consents_file", "bond_file", "hpi_file", "adp_file"):
        p = Path(CFG["paths"][key])
        if p.exists():
            rows.append({"snapshot": "", "file": key, "path": p.as_posix(),
                         "bytes": p.stat().st_size, "sha256": sha256(p)})
        else:
            log(f"WARNING: {key} not found at {p}")
    for p in Path(CFG["paths"]["ta_dir"]).rglob("*"):
        if p.is_file():
            rows.append({"snapshot": "", "file": "ta_boundary", "path": p.as_posix(),
                         "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(OUT / "SNAPSHOT_MANIFEST.csv", index=False)
    log(f"manifest: {len(rows)} files hashed")


# ----------------------------------------------------------------------------- stage 2
LCOLS = ["id", "last_scraped", "host_since", "host_response_rate", "host_acceptance_rate",
         "host_is_superhost", "host_total_listings_count", "host_listings_count",
         "host_has_profile_pic", "host_identity_verified", "latitude", "longitude",
         "property_type", "room_type", "accommodates", "bathrooms", "bathrooms_text",
         "bedrooms", "beds", "price", "minimum_nights", "maximum_nights", "availability_30",
         "availability_90", "availability_365", "number_of_reviews", "review_scores_rating",
         "instant_bookable", "calculated_host_listings_count", "license"]


def read_listings(snap):
    p = find_file(snap, "listings.csv")
    df = pd.read_csv(p, dtype=str, usecols=lambda c: c in LCOLS, low_memory=False)
    for c in LCOLS:
        if c not in df:
            df[c] = np.nan
    df = df.astype(object)
    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    n0 = len(df)
    df = df.drop_duplicates("id", keep="first")
    return df, n0 - len(df), list(pd.read_csv(p, nrows=0).columns)


def ta_join(df):
    import geopandas as gpd
    shp = sorted(Path(CFG["paths"]["ta_dir"]).rglob("*.shp"))
    if not shp:
        raise SystemExit("No .shp found under ta_dir")
    ta = gpd.read_file(shp[0]).to_crs(2193)
    # Shapefile field names are truncated to 10 chars (e.g. TA2025_V1_), so detect by content.
    strcols = [c for c in ta.columns if c != "geometry" and not pd.api.types.is_numeric_dtype(ta[c])]
    code = next(c for c in strcols if ta[c].astype(str).str.fullmatch(r"\d{3}").all())
    names = [c for c in strcols if c != code and ta[c].astype(str).str.contains(r"[A-Za-z]").all()]
    ascii_ok = [c for c in names if ta[c].astype(str).map(str.isascii).all()]
    name = next((c for c in names if "ASCII" in c.upper()), (ascii_ok or names)[-1])
    log(f"TA boundary: code field '{code}', name field '{name}', {len(ta)} polygons")
    ta = ta[[code, name, "geometry"]].rename(columns={code: "ta_code", name: "ta_name"})
    pts = df[["latitude", "longitude"]].astype(float)
    g = gpd.GeoDataFrame(df[["_rid"]].copy(),
                         geometry=gpd.points_from_xy(pts.longitude, pts.latitude), crs=4326).to_crs(2193)
    j = gpd.sjoin(g, ta, how="left", predicate="within").drop_duplicates("_rid")
    miss = j["ta_code"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(g[g["_rid"].isin(j.loc[miss, "_rid"])], ta, how="left",
                                 max_distance=2000).drop_duplicates("_rid")
        j = j.set_index("_rid")
        j.loc[near["_rid"], ["ta_code", "ta_name"]] = near.set_index("_rid")[["ta_code", "ta_name"]].values
        j = j.reset_index()
        j["ta_method"] = np.where(j["_rid"].isin(near.dropna(subset=["ta_code"])["_rid"]), "nearest<=2km", "within")
    else:
        j["ta_method"] = "within"
    j.loc[j["ta_code"].isna(), "ta_method"] = "unmatched"
    return j[["_rid", "ta_code", "ta_name", "ta_method"]], ta[["ta_code", "ta_name"]].drop_duplicates()


def peer_percentile(snap_df):
    """Midrank percentile of advertised price within TA x room type (n>=30), else TA (n>=30)."""
    d = snap_df[["id", "ta_code", "room_type", "price_num"]].copy()
    ok = d["price_num"] > 0
    out = pd.Series(np.nan, index=d.index)
    v = d[ok]
    for keys in (["ta_code", "room_type"], ["ta_code"]):
        g = v.groupby(keys, dropna=False)["price_num"]
        n = g.transform("size")
        r = (g.rank(method="average") - 0.5) / n
        fill = out.loc[v.index].isna() & (n >= 30)
        out.loc[v.index[fill.values]] = r[fill.values].values
    return out.values


def stage_panel():
    snaps = snapshots()
    frames, audit, schemas = [], [], {}
    for k, s in enumerate(snaps):
        df, ndup, cols = read_listings(s)
        schemas[s.name] = len(cols)
        df["snap_idx"], df["nominal"] = k, pd.Timestamp(s.name)
        df["last_scraped_dt"] = pd.to_datetime(df["last_scraped"], errors="coerce")
        df["price_num"] = num(df["price"])
        frames.append(df)
        audit.append({"snapshot": s.name, "rows": len(df), "dup_ids_dropped": ndup, "n_columns": len(cols)})
        log(f"listings {s.name}: {len(df):,} rows ({ndup} duplicate ids dropped, {len(cols)} columns)")
    allL = pd.concat(frames, ignore_index=True)
    allL["_rid"] = np.arange(len(allL))
    tj, ta_list = ta_join(allL)
    allL = allL.merge(tj, on="_rid", how="left")
    ta_list.to_csv(OUT / "ta_list.csv", index=False)
    allL["peer_pct"] = np.nan
    for k in range(len(snaps)):
        m = allL["snap_idx"] == k
        allL.loc[m, "peer_pct"] = peer_percentile(allL[m])

    ids = [set(allL.loc[allL.snap_idx == k, "id"]) for k in range(len(snaps))]
    pct = {k: allL[allL.snap_idx == k].set_index("id")["peer_pct"] for k in range(len(snaps))}
    asof = allL.groupby("snap_idx")["last_scraped_dt"].max()
    panel = []
    for k in range(len(snaps) - 1):
        d = allL[allL.snap_idx == k].copy()
        d["origin"] = k + 1
        d["next_nominal"] = pd.Timestamp(snaps[k + 1].name)
        d["asof"] = asof[k] if pd.notna(asof[k]) else d["nominal"].iloc[0]
        if d["asof"].iloc[0] >= d["next_nominal"].iloc[0]:
            raise SystemExit(f"T{k+1}: as-of date {d['asof'].iloc[0]} is not before the next snapshot")
        d["disappear"] = (~d["id"].isin(ids[k + 1])).astype(int)
        if k + 2 < len(snaps):
            d["reappear_t2"] = np.where(d.disappear == 1, d["id"].isin(ids[k + 2]).astype(float), np.nan)
        else:
            d["reappear_t2"] = np.nan
        nxt = d["id"].map(pct[k + 1])
        d["y_price"] = np.where(d.disappear == 0, nxt.values - d["peer_pct"].values, np.nan)
        panel.append(d)
    panel = pd.concat(panel, ignore_index=True)
    panel["scrape_delay_days"] = (panel["last_scraped_dt"] - panel["nominal"]).dt.days
    save(panel, "panel")
    pd.DataFrame(audit).to_csv(OUT / "audit_snapshots.csv", index=False)

    a = panel.groupby("origin").agg(
        origins=("id", "size"), disappear=("disappear", "sum"),
        price_missing=("price_num", lambda x: x.isna().sum()),
        scraped_later=("scrape_delay_days", lambda x: (x > 0).sum()),
        scraped_missing=("scrape_delay_days", lambda x: x.isna().sum()),
        median_delay=("scrape_delay_days", "median"), max_delay=("scrape_delay_days", "max"),
        reappear_t2=("reappear_t2", "sum"), price_labels=("y_price", lambda x: x.notna().sum()),
        ta_unmatched=("ta_method", lambda x: (x == "unmatched").sum()),
        ta_nearest=("ta_method", lambda x: (x == "nearest<=2km").sum()),
        asof=("asof", "first"), nominal=("nominal", "first"), next_nominal=("next_nominal", "first"))
    a.to_csv(OUT / "audit_origins.csv")
    log("origin audit:\n" + a[["origins", "disappear", "price_missing", "scraped_later", "max_delay"]].to_string())


# ----------------------------------------------------------------------------- stage 3
def listing_features(p):
    f = pd.DataFrame(index=p.index)
    f["accommodates"] = num(p["accommodates"])
    f["bedrooms"] = num(p["bedrooms"])
    f["beds"] = num(p["beds"])
    bt = p["bathrooms_text"].astype(str).str.lower()
    b = num(p["bathrooms"])
    bt_num = pd.to_numeric(bt.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    bt_num = bt_num.where(~bt.str.contains("half"), 0.5)
    f["bathrooms_num"] = np.where(np.isnan(b), bt_num, b)
    f["bath_shared"] = bt.str.contains("shared").astype(float)
    f["room_type"] = p["room_type"].astype(str)
    pt = p["property_type"].astype(str).str.lower()
    f["ptype_grp"] = np.select(
        [pt.str.contains("room") & pt.str.contains("private|shared"),
         pt.str.contains("hotel|hostel|resort|bed and breakfast|boutique"),
         pt.str.contains("condo|apartment|rental unit|serviced|loft|flat"),
         pt.str.contains("home|house|townhouse|cottage|villa|bungalow|cabin|chalet")],
        ["room", "hospitality", "apartment", "house"], "other")
    hs = pd.to_datetime(p["host_since"], errors="coerce")
    f["host_tenure_days"] = (p["nominal"] - hs).dt.days
    f["host_is_superhost"] = tf(p["host_is_superhost"])
    f["host_total_listings"] = num(p["host_total_listings_count"])
    f["calc_host_listings"] = num(p["calculated_host_listings_count"])
    f["host_response_rate"] = num(p["host_response_rate"])
    f["host_acceptance_rate"] = num(p["host_acceptance_rate"])
    f["host_identity_verified"] = tf(p["host_identity_verified"])
    f["host_has_profile_pic"] = tf(p["host_has_profile_pic"])
    f["instant_bookable"] = tf(p["instant_bookable"])
    f["minimum_nights"] = num(p["minimum_nights"])
    f["maximum_nights"] = np.minimum(num(p["maximum_nights"]), 1125)
    f["availability_90"] = num(p["availability_90"])
    f["availability_365"] = num(p["availability_365"])
    f["number_of_reviews"] = num(p["number_of_reviews"])
    f["review_scores_rating"] = num(p["review_scores_rating"])
    f["has_license"] = p["license"].notna().astype(float)
    f["ta_code"] = p["ta_code"].astype(str)
    return f


def calendar_features(panel):
    import duckdb
    snaps = snapshots()
    out, audit = [], []
    for k in sorted(panel["origin"].unique()):
        s = snaps[k - 1]
        p = find_file(s, "calendar.csv").as_posix()
        q = f"""
        WITH c AS (SELECT CAST(listing_id AS VARCHAR) AS id, TRY_CAST(date AS DATE) AS d,
                    CASE WHEN lower(available) IN ('t','true') THEN 1 ELSE 0 END AS av,
                    TRY_CAST(minimum_nights AS DOUBLE) AS minn, TRY_CAST(maximum_nights AS DOUBLE) AS maxn
                   FROM read_csv_auto('{p}', all_varchar=true)),
        r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY d) AS rn,
                     LAG(av) OVER (PARTITION BY id ORDER BY d) AS prev FROM c)
        SELECT id, MIN(d) AS cal_first_date,
          COUNT(*) FILTER (WHERE rn<=28) AS cal_n28,
          AVG(av) FILTER (WHERE rn<=7) AS cal_avail_7,
          AVG(av) FILTER (WHERE rn<=28) AS cal_avail_28,
          AVG(av) FILTER (WHERE rn BETWEEN 8 AND 28) AS cal_avail_8_28,
          SUM(CASE WHEN rn BETWEEN 2 AND 28 AND av<>prev THEN 1 ELSE 0 END) AS cal_switch_28,
          MEDIAN(minn) FILTER (WHERE rn<=28) AS cal_minn_med_28,
          MAX(minn) FILTER (WHERE rn<=28) AS cal_minn_max_28,
          MEDIAN(maxn) FILTER (WHERE rn<=28) AS cal_maxn_med_28
        FROM r GROUP BY id"""
        c = duckdb.sql(q).df()
        pr = duckdb.sql(f"""SELECT COUNT(*) n, SUM(CASE WHEN price IS NOT NULL AND trim(price)<>'' THEN 1 ELSE 0 END) p
                             FROM read_csv_auto('{p}', all_varchar=true)""").fetchone()
        nominal = pd.Timestamp(s.name)
        c["cal_first_date"] = pd.to_datetime(c["cal_first_date"])
        c["cal_first_lag_days"] = (c["cal_first_date"] - nominal).dt.days
        c["origin"] = k
        audit.append({"origin": k, "calendar_rows": pr[0], "calendar_price_nonnull": pr[1],
                      "first_date_min": c.cal_first_date.min(), "first_date_max": c.cal_first_date.max(),
                      "first_date_before_nominal": int((c.cal_first_lag_days < 0).sum())})
        out.append(c.drop(columns="cal_first_date"))
        log(f"calendar T{k}: {len(c):,} listings, price non-null {pr[1]:,}/{pr[0]:,}")
    pd.DataFrame(audit).to_csv(OUT / "audit_calendar.csv", index=False)
    return pd.concat(out, ignore_index=True)


def review_features(panel):
    import duckdb
    snaps = snapshots()
    out, audit = [], []
    for k in sorted(panel["origin"].unique()):
        s = snaps[k - 1]
        p = find_file(s, "reviews.csv").as_posix()
        cut = s.name  # conservative cutoff: nominal snapshot date (before the operational as-of date)
        q = f"""
        WITH v AS (SELECT CAST(listing_id AS VARCHAR) AS id, TRY_CAST(date AS DATE) AS d
                   FROM read_csv_auto('{p}', all_varchar=true, ignore_errors=true))
        SELECT id,
          COUNT(*) FILTER (WHERE d <= DATE '{cut}') AS rev_total,
          COUNT(*) FILTER (WHERE d <= DATE '{cut}' AND d > DATE '{cut}' - INTERVAL 30 DAY) AS rev_n_30,
          COUNT(*) FILTER (WHERE d <= DATE '{cut}' AND d > DATE '{cut}' - INTERVAL 90 DAY) AS rev_n_90,
          COUNT(*) FILTER (WHERE d <= DATE '{cut}' AND d > DATE '{cut}' - INTERVAL 365 DAY) AS rev_n_365,
          MAX(d) FILTER (WHERE d <= DATE '{cut}') AS last_d,
          MIN(d) FILTER (WHERE d <= DATE '{cut}') AS first_d,
          COUNT(*) FILTER (WHERE d > DATE '{cut}') AS rev_after_cut
        FROM v GROUP BY id"""
        r = duckdb.sql(q).df()
        nominal = pd.Timestamp(cut)
        r["rev_days_since_last"] = (nominal - pd.to_datetime(r.pop("last_d"))).dt.days
        r["rev_days_since_first"] = (nominal - pd.to_datetime(r.pop("first_d"))).dt.days
        r["rev_share_365"] = r["rev_n_365"] / r["rev_total"].replace(0, np.nan)
        r["rev_trend_90"] = r["rev_n_90"] - r["rev_n_365"] / 4.0
        r["origin"] = k
        audit.append({"origin": k, "listings_with_reviews": len(r), "reviews_after_cutoff_excluded": int(r.rev_after_cut.sum())})
        out.append(r.drop(columns="rev_after_cut"))
        log(f"reviews T{k}: {len(r):,} listings; {int(r.rev_after_cut.sum()):,} post-cutoff reviews excluded")
    pd.DataFrame(audit).to_csv(OUT / "audit_reviews.csv", index=False)
    return pd.concat(out, ignore_index=True)


def parse_consents(ta_list):
    """Returns tidy (ta_code, month: Period[M], value). Supports a tidy override or Infoshare CSV."""
    pth = CFG["paths"].get("consents_tidy")
    name2code = {norm_name(n): c for c, n in zip(ta_list.ta_code, ta_list.ta_name)}
    if pth:
        t = read_csv_any(pth)
        t["ta_code"] = t["ta_name"].map(lambda x: name2code.get(norm_name(x)))
        t["month"] = pd.PeriodIndex(t["month"].astype(str).str.replace("M", "-"), freq="M")
        return t[["ta_code", "month", "value"]], "tidy override"
    raw = read_csv_any(CFG["paths"]["consents_file"], header=None, dtype=str)
    is_data = raw[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")
    first = int(is_data.idxmax())
    hdr = raw.iloc[:first].ffill(axis=1).fillna("")
    labels = [" | ".join(x for x in hdr[c].astype(str) if x and x != "nan") for c in raw.columns]
    inc = re.compile(CFG["consents"]["include_regex"], re.I)
    exc = re.compile(CFG["consents"]["exclude_regex"], re.I)
    keep, mapping = [], []
    names_sorted = sorted(name2code, key=len, reverse=True)
    for c, lab in zip(raw.columns[1:], labels[1:]):
        nl = norm_name(lab)
        hit = next((n for n in names_sorted if n and n in nl), None)
        ok = hit is not None and inc.search(lab) and not exc.search(lab)
        mapping.append({"column": c, "header": lab, "ta_code": name2code.get(hit) if hit else None, "used": bool(ok)})
        if ok:
            keep.append((c, name2code[hit]))
    pd.DataFrame(mapping).to_csv(OUT / "audit_consents_columns.csv", index=False)
    data = raw[is_data]
    recs = []
    for c, code in keep:
        v = pd.to_numeric(data[c], errors="coerce")
        recs.append(pd.DataFrame({"ta_code": code, "month": data[0].str.replace("M", "-"), "value": v}))
    if not recs:
        raise SystemExit("Consents parser matched no columns. Inspect audit_consents_columns.csv, "
                         "adjust consents.include_regex/exclude_regex, or set paths.consents_tidy.")
    t = pd.concat(recs).groupby(["ta_code", "month"], as_index=False)["value"].sum(min_count=1)
    t["month"] = pd.PeriodIndex(t["month"], freq="M")
    log(f"consents: {t.ta_code.nunique()} TAs, {t.month.min()}..{t.month.max()}, {len(keep)} columns used")
    return t, "Infoshare parser"


def consents_available_month(asof):
    cal = Path(CFG["paths"]["consents_release_calendar"])
    if cal.exists():
        rc = pd.read_csv(cal).dropna(subset=["release_date"])
        rc["release_date"] = pd.to_datetime(rc["release_date"])
        ok = rc[rc.release_date <= asof]
        if len(ok):
            return pd.Period(str(ok.ref_month.max()).replace("M", "-"), freq="M"), "release calendar"
    return pd.Period(asof, freq="M") - int(CFG["consents"]["fallback_lag_months"]), "fallback rule (UNVERIFIED)"


def construction_features(origins, ta_list):
    t, how = parse_consents(ta_list)
    wide = t.pivot_table(index="month", columns="ta_code", values="value", aggfunc="sum").sort_index()
    rows, aud = [], []
    for o, asof in origins.items():
        m, rule = consents_available_month(asof)
        aud.append({"origin": o, "asof": asof.date(), "consent_ref_month": str(m), "rule": rule, "parser": how})
        def s(a, b):
            return wide.loc[(wide.index > m - a) & (wide.index <= m - b)].sum(min_count=1)
        c3, c12, c3ly = s(3, 0), s(12, 0), s(15, 12)
        df = pd.DataFrame({"cons_3m": c3, "cons_12m": c12,
                           "cons_yoy_3m": np.log1p(c3) - np.log1p(c3ly),
                           "cons_mom": np.log1p(c3) - np.log1p(c12 / 4.0)})
        df["origin"] = o
        rows.append(df.reset_index().rename(columns={"index": "ta_code"}))
    pd.DataFrame(aud).to_csv(OUT / "audit_construction_timing.csv", index=False)
    out = pd.concat(rows, ignore_index=True)
    out["ta_code"] = out["ta_code"].astype(str)
    return out


def rental_features(origins, ta_list):
    b = read_csv_any(CFG["paths"]["bond_file"])
    tcol = next(c for c in b.columns if "time" in c.lower() or "date" in c.lower())
    lcol = next(c for c in b.columns if c.lower() in ("location", "location name", "tla", "territorial authority"))
    rcol = next(c for c in b.columns if "geometric" in c.lower()) if any("geometric" in c.lower() for c in b.columns) \
        else next(c for c in b.columns if "median" in c.lower())
    b["month"] = pd.to_datetime(b[tcol], errors="coerce", dayfirst=True).dt.to_period("M")
    name2code = {norm_name(n): c for c, n in zip(ta_list.ta_code, ta_list.ta_name)}
    b["ta_code"] = b[lcol].map(lambda x: name2code.get(norm_name(x)))
    unmatched = sorted(set(b.loc[b.ta_code.isna(), lcol].astype(str)))
    pd.Series(unmatched, name="bond_location_unmatched").to_csv(OUT / "audit_bond_unmatched.csv", index=False)
    b["rent"] = pd.to_numeric(b[rcol], errors="coerce")
    w = b.dropna(subset=["ta_code"]).pivot_table(index="month", columns="ta_code", values="rent", aggfunc="mean")
    rows, aud = [], []
    lag = int(CFG["bond"]["lag_months"])
    for o, asof in origins.items():
        m = pd.Period(asof, freq="M") - lag
        aud.append({"origin": o, "asof": asof.date(), "bond_ref_month": str(m), "rule": f"M-{lag} conservative"})
        cur = w.loc[m] if m in w.index else pd.Series(dtype=float)
        ly = w.loc[m - 12] if (m - 12) in w.index else pd.Series(dtype=float)
        df = pd.DataFrame({"rent_log_gm": np.log(cur), "rent_yoy": np.log(cur) - np.log(ly.reindex(cur.index))})
        df["origin"] = o
        rows.append(df.reset_index().rename(columns={"index": "ta_code"}))
    pd.DataFrame(aud).to_csv(OUT / "audit_rental_timing.csv", index=False)
    log(f"rental: rent column '{rcol}', {len(unmatched)} unmatched bond locations")
    out = pd.concat(rows, ignore_index=True)
    out["ta_code"] = out["ta_code"].astype(str)
    return out


def seasonality_features(origins, next_nominal):
    sc = CFG["seasonality"]
    a = read_csv_any(CFG["paths"]["adp_file"])
    for col, val in (sc.get("filters") or {}).items():
        a = a[a[col].astype(str) == str(val)]
    tcol = sc.get("time_col") or next((c for c in a.columns if re.search(r"period|month|date|time", c, re.I)), None)
    vcol = sc.get("value_col") or next((c for c in a.columns if c.lower() in ("value", "obs_value", "data_value")), None)
    if tcol is None or vcol is None:
        log(f"WARNING seasonality: cannot detect columns {list(a.columns)}; family DROPPED. Set seasonality.time_col/value_col.")
        return None
    a["month"] = pd.to_datetime(a[tcol].astype(str).str.replace("M", "-"), errors="coerce").dt.to_period("M")
    a["v"] = pd.to_numeric(a[vcol], errors="coerce")
    yrs = sc["profile_years"]
    a = a[a.month.dt.year.isin(yrs)]
    if a.month.nunique() < 12:
        log("WARNING seasonality: <12 profile months after filtering; family DROPPED. Set seasonality.filters.")
        return None
    prof = a.groupby(a.month.dt.month)["v"].sum() / a.groupby(a.month.dt.month)["v"].sum().mean()
    rows = []
    for o, asof in origins.items():
        m0, m1 = asof.month, next_nominal[o].month
        rows.append({"origin": o, "season_idx": prof.get(m0), "season_idx_next": prof.get(m1),
                     "season_change": prof.get(m1) - prof.get(m0)})
    pd.DataFrame({"month": prof.index, "season_index": prof.values}).to_csv(OUT / "seasonality_profile.csv", index=False)
    log(f"seasonality: profile {yrs} from column '{vcol}' ({len(a):,} rows)")
    return pd.DataFrame(rows)


def semantic_features(panel):
    """Pooled sentence embeddings (A6 comparator) of the K most recent reviews dated <= nominal origin."""
    sc = CFG.get("semantic") or {}
    if not sc.get("enabled"):
        return None
    import duckdb
    if sc["model"] == "__hash__":          # test hook only: deterministic hashed bag-of-words
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(n_features=384, alternate_sign=True, norm="l2")
        encode = lambda t: hv.transform(t).toarray()
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log("WARNING semantic: sentence-transformers not installed; A6/MLP-A6 DROPPED")
            return None
        st = SentenceTransformer(sc["model"], device="cpu")
        encode = lambda t: st.encode(t, batch_size=int(sc["batch_size"]), normalize_embeddings=True,
                                     show_progress_bar=True, convert_to_numpy=True)
    snaps, K = snapshots(), int(sc["max_reviews_per_listing"])
    sel = []
    for k in sorted(panel["origin"].unique()):
        s = snaps[k - 1]
        p = find_file(s, "reviews.csv").as_posix()
        q = f"""SELECT id, rid, comments FROM (
                  SELECT CAST(listing_id AS VARCHAR) AS id, CAST(id AS VARCHAR) AS rid, comments,
                         ROW_NUMBER() OVER (PARTITION BY listing_id ORDER BY TRY_CAST(date AS DATE) DESC, id DESC) AS rn
                  FROM read_csv_auto('{p}', all_varchar=true, ignore_errors=true)
                  WHERE TRY_CAST(date AS DATE) <= DATE '{s.name}' AND comments IS NOT NULL AND trim(comments) <> '')
                WHERE rn <= {K}"""
        r = duckdb.sql(q).df()
        r["origin"] = k
        sel.append(r)
    sel = pd.concat(sel, ignore_index=True)
    cache = OUT / "review_embeddings.parquet"
    uniq = sel.drop_duplicates("rid")[["rid", "comments"]]
    if cache.exists():
        E = pd.read_parquet(cache)
    else:
        E = pd.DataFrame({"rid": []})
    todo = uniq[~uniq.rid.isin(E.rid)] if len(E) else uniq
    if len(todo):
        log(f"semantic: encoding {len(todo):,} review texts with {sc['model']} (cached afterwards)")
        parts = []
        for i in range(0, len(todo), 50000):
            chunk = todo.iloc[i:i + 50000]
            emb = np.asarray(encode(chunk.comments.str.slice(0, int(sc["max_chars"])).tolist()), dtype=np.float16)
            parts.append(pd.DataFrame(emb, columns=[f"e{j:03d}" for j in range(emb.shape[1])]).assign(rid=chunk.rid.values))
            log(f"semantic: {min(i + 50000, len(todo)):,}/{len(todo):,}")
        E = pd.concat([E] + parts, ignore_index=True) if len(E) else pd.concat(parts, ignore_index=True)
        E.to_parquet(cache, index=False)
    ecols = [c for c in E.columns if c.startswith("e")]
    pos = pd.Series(np.arange(len(E)), index=E.rid.values)
    M = E[ecols].to_numpy(dtype=np.float32)
    out = []
    for k, g in sel.groupby("origin"):
        codes, ids = pd.factorize(g["id"])
        acc = np.zeros((len(ids), len(ecols)), dtype=np.float32)
        np.add.at(acc, codes, M[pos.loc[g.rid].values])
        acc /= np.bincount(codes)[:, None]
        df = pd.DataFrame(acc, columns=[f"sem_{j:03d}" for j in range(len(ecols))])
        df["id"], df["origin"] = ids, k
        out.append(df)
    log(f"semantic: {len(ecols)} pooled dimensions for {sum(len(o) for o in out):,} listing-origins")
    return pd.concat(out, ignore_index=True)


def hpi_features(origins, ta_list):
    try:
        x = pd.read_excel(CFG["paths"]["hpi_file"], sheet_name=None)
        hsheet = next(v for k, v in x.items() if "price index" in k.lower())
        ssheet = next(v for k, v in x.items() if k.lower().strip() == "sales")
        name2code = {norm_name(n): c for c, n in zip(ta_list.ta_code, ta_list.ta_name)}
        def prep(d):
            d = d[d["area_type"].astype(str).str.contains("territorial", case=False)].copy()
            if "stat_type" in d:
                st = d["stat_type"].astype(str)
                d = d[st == (st[st.str.contains("3")].iloc[0] if st.str.contains("3").any() else st.iloc[0])]
            d["ta_code"] = d["area"].map(lambda v: name2code.get(norm_name(v)))
            d["month"] = pd.to_datetime(d["period"], errors="coerce").dt.to_period("M")
            return d.dropna(subset=["ta_code", "month"])
        h = prep(hsheet)
        if "value_quartile" in h:
            vq = h["value_quartile"].astype(str)
            h = h[vq.str.contains("all", case=False)] if vq.str.contains("all", case=False).any() else h
        s = prep(ssheet)
        hw = h.pivot_table(index="month", columns="ta_code", values="house_price_index")
        sw = s.pivot_table(index="month", columns="ta_code", values="residential_sales")
        rows = []
        for o, asof in origins.items():
            m = pd.Period(asof, freq="M") - int(CFG["hpi_robustness"]["lag_months"])
            df = pd.DataFrame({"hpi_change_12m": np.log(hw.loc[m]) - np.log(hw.loc[m - 12]),
                               "sales_activity_yoy": np.log1p(sw.loc[m]) - np.log1p(sw.loc[m - 12])})
            df["origin"] = o
            rows.append(df.reset_index().rename(columns={"index": "ta_code"}))
        out = pd.concat(rows, ignore_index=True)
        out["ta_code"] = out["ta_code"].astype(str)
        log("hpi robustness features built (ex-post vintage; NOT available at T1-T10)")
        return out
    except Exception as e:
        log(f"WARNING hpi robustness skipped: {e!r}")
        return None


FAMILIES = {}


def stage_features():
    panel = load("panel")
    ta_list = pd.read_csv(OUT / "ta_list.csv", dtype=str)
    origins = panel.groupby("origin")["asof"].first()
    next_nom = panel.groupby("origin")["next_nominal"].first()
    lf = listing_features(panel)
    base = panel[["id", "origin", "disappear", "reappear_t2", "y_price", "ta_code", "asof"]].copy()
    base["ta_code"] = base["ta_code"].astype(str)
    X = pd.concat([base, lf.drop(columns=["ta_code"])], axis=1)
    X = X.merge(calendar_features(panel), on=["id", "origin"], how="left")
    X = X.merge(review_features(panel), on=["id", "origin"], how="left")
    for c in ["rev_total", "rev_n_30", "rev_n_90", "rev_n_365"]:
        X[c] = X[c].fillna(0)
    X = X.merge(construction_features(origins, ta_list), on=["origin", "ta_code"], how="left")
    X = X.merge(rental_features(origins, ta_list), on=["origin", "ta_code"], how="left")
    se = seasonality_features(origins, next_nom)
    if se is not None:
        X = X.merge(se, on="origin", how="left")
    sm = semantic_features(panel)
    if sm is not None:
        X = X.merge(sm, on=["id", "origin"], how="left")
    hp = hpi_features(origins, ta_list) if CFG["hpi_robustness"]["enabled"] else None
    if hp is not None:
        X = X.merge(hp, on=["origin", "ta_code"], how="left")
    assert len(X) == len(panel), "feature merge changed row count"
    save(X, "features")
    log(f"features: {X.shape[0]:,} rows x {X.shape[1]} columns")


def families(X):
    fam = {
        "property": ["accommodates", "bedrooms", "beds", "bathrooms_num", "bath_shared", "room_type", "ptype_grp"],
        "platform": ["host_tenure_days", "host_is_superhost", "host_total_listings", "calc_host_listings",
                     "host_response_rate", "host_acceptance_rate", "host_identity_verified",
                     "host_has_profile_pic", "instant_bookable", "minimum_nights", "maximum_nights",
                     "availability_90", "availability_365", "number_of_reviews", "review_scores_rating",
                     "has_license"],
        "geography": ["ta_code"],
        "rental": ["rent_log_gm", "rent_yoy"],
        "construction": ["cons_3m", "cons_12m", "cons_yoy_3m", "cons_mom"],
        "seasonality": ["season_idx", "season_idx_next", "season_change"],
        "calendar": ["cal_first_lag_days", "cal_n28", "cal_avail_7", "cal_avail_28", "cal_avail_8_28",
                     "cal_switch_28", "cal_minn_med_28", "cal_minn_max_28", "cal_maxn_med_28"],
        "review": ["rev_total", "rev_n_30", "rev_n_90", "rev_n_365", "rev_days_since_last",
                   "rev_days_since_first", "rev_share_365", "rev_trend_90"],
        "hpi": ["hpi_change_12m", "sales_activity_yoy"],
        "semantic": [c for c in X.columns if c.startswith("sem_")],
    }
    return {k: [c for c in v if c in X.columns] for k, v in fam.items()}


# ----------------------------------------------------------------------------- stage 4
CATS = ["room_type", "ptype_grp", "ta_code"]


def to_matrix(df, cols, cats):
    M = df[cols].copy()
    for c in cols:
        if c in CATS:
            M[c] = pd.Categorical(df[c].astype(str), categories=cats[c])
        else:
            M[c] = pd.to_numeric(M[c], errors="coerce").astype(float)
    return M


def fit_predict(kind, cols, tr, te, cats, task="clf"):
    from xgboost import XGBClassifier, XGBRegressor
    p = CFG["model"]["xgb"]
    seed = CFG["model"]["seed"]
    if kind == "xgb":
        M = (XGBClassifier if task == "clf" else XGBRegressor)(
            **p, tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=-1,
            **({"eval_metric": "logloss"} if task == "clf" else {}))
        y = tr["disappear"] if task == "clf" else tr["y_price"]
        M.fit(to_matrix(tr, cols, cats), y)
        Xt = to_matrix(te, cols, cats)
        return (M.predict_proba(Xt)[:, 1] if task == "clf" else M.predict(Xt)), M
    if kind in ("logit", "mlp"):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        nc = [c for c in cols if c not in CATS]
        cc = [c for c in cols if c in CATS]
        pre = ColumnTransformer([
            ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                              keep_empty_features=True), StandardScaler()), nc),
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cc)])
        if kind == "logit":
            est = LogisticRegression(max_iter=2000, C=1.0)
        else:
            from sklearn.neural_network import MLPClassifier
            est = MLPClassifier(hidden_layer_sizes=(128, 32), activation="relu", alpha=1e-4, batch_size=512,
                                learning_rate_init=1e-3, max_iter=80, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=7, random_state=seed)
        M = make_pipeline(pre, est)
        f = lambda d: d[cols].assign(**{c: d[c].astype(str) for c in cc})
        M.fit(f(tr), tr["disappear"])
        return M.predict_proba(f(te))[:, 1], M
    raise ValueError(kind)


def clf_metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), "n": len(y), "events": int(y.sum())}


def platt_fit(p, y):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return lambda q: m.predict_proba(np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1]


def iso_fit(p, y):
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p, y)
    return m.predict


def cluster_bootstrap(df, fn, reps, seed):
    rng = np.random.default_rng(seed)
    groups = df.groupby("ta_code").indices
    keys = list(groups)
    vals = []
    for _ in range(reps):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        try:
            vals.append(fn(df.iloc[idx]))
        except ValueError:
            continue
    v = pd.DataFrame(vals)
    return v.quantile(0.025), v.quantile(0.975), len(v)


def stage_models():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve
    X = load("features")
    X["ta_code"] = X["ta_code"].astype(str)
    fam = families(X)
    mc = CFG["model"]
    dev, test = mc["dev_windows"], mc["test_window"]
    devX = X[X.origin < test]
    cats = {c: sorted(devX[c].astype(str).unique()) for c in CATS}  # vocabularies from T1-T9 only

    order = ["property", "platform", "geography", "rental", "construction", "seasonality", "calendar", "review"]
    sets, acc = {}, []
    for i, f in enumerate(order):
        acc = acc + fam[f]
        if i >= 2:
            sets[f"A{i-2}"] = list(acc)       # A0=prop+plat+geo, A1=+rental, ..., A5=+review
    full = sets["A5"]
    if fam["semantic"]:
        sets["A6"] = full + fam["semantic"]
    lofo = {f"A5-{f}": [c for c in full if c not in fam[f]] for f in ["rental", "construction", "seasonality", "calendar", "review"] if fam[f]}
    if fam["hpi"]:
        lofo["A5+hpi (ex-post robustness)"] = full + fam["hpi"]
    specs = [("xgb", k, v) for k, v in sets.items()] + [("logit", "A5", full), ("mlp", "A5", full)]
    if "A6" in sets:
        specs.append(("mlp", "A6", sets["A6"]))
    specs += [("xgb", k, v) for k, v in lofo.items()]

    preds, rows = [], []
    for kind, name, cols in specs:
        for w in dev:
            tr, te = X[X.origin < w], X[X.origin == w]
            p, _ = fit_predict(kind, cols, tr, te, cats)
            m = clf_metrics(te["disappear"].values, p)
            rows.append({"model": kind, "set": name, "n_pred": len(cols), "window": w, **m})
            preds.append(pd.DataFrame({"model": kind, "set": name, "window": w, "id": te["id"].values,
                                       "y": te["disappear"].values, "p": p}))
        log(f"dev {kind}:{name} ({len(cols)} predictors) done")
    dev_win = pd.DataFrame(rows)
    dev_win.to_csv(OUT / "dev_by_window.csv", index=False)
    devp = pd.concat(preds, ignore_index=True)
    devp.to_parquet(OUT / "dev_predictions.parquet", index=False)
    summ = dev_win.groupby(["model", "set", "n_pred"], as_index=False)[["roc_auc", "pr_auc", "brier"]].mean()
    summ.to_csv(OUT / "dev_summary.csv", index=False)

    # selection among candidates (A0-A5 xgb + logit A5); ablations are not candidates
    cand = summ[summ.set.isin(sets)].copy()   # A0-A6 for every model class; ablations excluded
    best = cand.pr_auc.max()
    near = cand[cand.pr_auc >= best * 0.99].sort_values(["roc_auc", "brier"], ascending=[False, True])
    sel = near.iloc[0]
    sel_kind, sel_set = sel.model, sel.set
    sel_cols = sets[sel_set]
    log(f"SELECTED: {sel_kind} {sel_set} ({len(sel_cols)} predictors)")

    sp = devp[(devp.model == sel_kind) & (devp.set == sel_set)]
    fpr, tpr, thr = roc_curve(sp.y, sp.p)
    threshold = float(thr[np.argmax(tpr - fpr)])

    # calibration choice: leave-one-window-out on dev predictions
    cal = {"raw": [], "platt": [], "isotonic": []}
    for w in dev:
        a, b = sp[sp.window != w], sp[sp.window == w]
        cal["raw"].append(np.mean((b.p - b.y) ** 2))
        cal["platt"].append(np.mean((platt_fit(a.p.values, a.y.values)(b.p.values) - b.y) ** 2))
        cal["isotonic"].append(np.mean((iso_fit(a.p.values, a.y.values)(b.p.values) - b.y) ** 2))
    cal = {k: float(np.mean(v)) for k, v in cal.items()}
    best_cal = min(("platt", "isotonic"), key=lambda k: cal[k])
    calib = best_cal if (cal["raw"] - cal[best_cal]) / cal["raw"] >= mc["calib_min_rel_gain"] else "raw"
    calibrator = {"platt": platt_fit, "isotonic": iso_fit}.get(calib)
    calibrator = calibrator(sp.p.values, sp.y.values) if calibrator else None
    freeze = {"model": sel_kind, "set": sel_set, "predictors": sel_cols, "threshold": threshold,
              "calibration": calib, "calibration_dev_brier": cal, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "FROZEN_SPEC.json").write_text(json.dumps(freeze, indent=2))
    log(f"FROZEN before T10: threshold={threshold:.6f}, calibration={calib}")

    # ---- single locked T10 evaluation
    tr, te = X[X.origin < test], X[X.origin == test].reset_index(drop=True)
    p, model = fit_predict(sel_kind, sel_cols, tr, te, cats)
    te = te.assign(p=p, p_cal=calibrator(p) if calibrator else p)
    te[["id", "ta_code", "disappear", "p", "p_cal"]].to_csv(OUT / "T10_predictions.csv", index=False)
    y = te.disappear.values
    m_raw, m_cal = clf_metrics(y, p), clf_metrics(y, te.p_cal.values)
    yhat = (p >= threshold).astype(int)
    cm = {"TP": int(((yhat == 1) & (y == 1)).sum()), "FP": int(((yhat == 1) & (y == 0)).sum()),
          "FN": int(((yhat == 0) & (y == 1)).sum()), "TN": int(((yhat == 0) & (y == 0)).sum())}
    lo, hi, nb = cluster_bootstrap(te.rename(columns={"disappear": "y"}),
                                   lambda d: {k: v for k, v in clf_metrics(d.y.values, d.p.values).items() if k in ("roc_auc", "pr_auc", "brier")},
                                   mc["bootstrap_reps"], mc["seed"])
    order_idx = np.argsort(-p, kind="mergesort")
    topk = []
    for q in mc["topk"]:
        k = int(np.ceil(q * len(te)))
        ev = int(y[order_idx[:k]].sum())
        topk.append({"share": q, "listings": k, "events": ev, "precision": ev / k,
                     "recall": ev / y.sum(), "lift": (ev / k) / y.mean()})
    topk = pd.DataFrame(topk)
    topk.to_csv(OUT / "T10_topk.csv", index=False)
    te["decile"] = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False) + 1
    dec = te.groupby("decile").agg(n=("disappear", "size"), events=("disappear", "sum"), mean_p=("p", "mean"))
    dec["rate"] = dec.events / dec.n
    dec.to_csv(OUT / "T10_deciles.csv")

    shap_top = None
    if sel_kind == "xgb":
        import xgboost as xgb
        smp = te.sample(n=min(3000, len(te)), random_state=mc["seed"])
        contrib = model.get_booster().predict(xgb.DMatrix(to_matrix(smp, sel_cols, cats), enable_categorical=True), pred_contribs=True)
        shap_top = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=sel_cols).sort_values(ascending=False)
        shap_top.to_csv(OUT / "T10_shap_mean_abs.csv", header=["mean_abs_shap"])

    # ---- secondary outcome: within-peer advertised-price percentile change
    sec_rows = []
    lab = X[X.y_price.notna()]
    for w in [w for w in dev if (lab.origin == w).any() and (lab.origin < w).any()]:
        trs, tes = lab[lab.origin < w], lab[lab.origin == w]
        ps, _ = fit_predict("xgb", full, trs, tes, cats, task="reg")
        for nm, pr in (("zero", np.zeros(len(tes))), ("xgb_A5", ps)):
            sec_rows.append({"model": nm, "window": w, "mae": mean_absolute_error(tes.y_price, pr),
                             "rmse": mean_squared_error(tes.y_price, pr) ** 0.5})
    sec_dev = pd.DataFrame(sec_rows)
    sec_dev.to_csv(OUT / "secondary_dev.csv", index=False)
    sec_choice = sec_dev.groupby("model").mae.mean().idxmin() if len(sec_dev) else "zero"
    tes = lab[lab.origin == test]
    if sec_choice == "zero":
        ps = np.zeros(len(tes))
    else:
        ps, _ = fit_predict("xgb", full, lab[lab.origin < test], tes, cats, task="reg")
    sec = {"choice": sec_choice, "n": len(tes), "mae": mean_absolute_error(tes.y_price, ps),
           "rmse": mean_squared_error(tes.y_price, ps) ** 0.5, "r2": r2_score(tes.y_price, ps)}

    # ---- data audits feeding the paper
    ao = pd.read_csv(OUT / "audit_origins.csv")
    tot = len(X)
    reap = X[X.origin < test]
    res = {
        "n_origins": tot, "n_events": int(X.disappear.sum()),
        "price_missing": int(ao.price_missing.sum()), "price_missing_pct": 100 * ao.price_missing.sum() / tot,
        "scraped_later": int(ao.scraped_later.sum()), "scraped_later_pct": 100 * ao.scraped_later.sum() / tot,
        "max_delay": int(ao.max_delay.max()),
        "reappear": int(reap.reappear_t2.sum()), "reappear_den": int((reap.disappear == 1).sum()),
        "selected": f"{sel_kind} {sel_set}", "n_predictors": len(sel_cols), "threshold": threshold,
        "calibration": calib, "dev": summ.to_dict("records"),
        "t10_raw": m_raw, "t10_cal": m_cal, "t10_ci_low": lo.to_dict(), "t10_ci_high": hi.to_dict(),
        "bootstrap_valid": nb, "confusion": cm, "secondary": sec,
    }
    (OUT / "RESULTS.json").write_text(json.dumps(res, indent=2, default=str))
    write_macros(res, topk, sel_cols)
    make_figures(summ, sets, lofo, dec)
    log("DONE. See RESULTS.json, results_macros.tex, *.csv, figures/*.pdf")


def write_macros(r, topk, cols):
    f = lambda x, d=4: f"{x:.{d}f}"
    t10 = r["t10_raw"]
    top10 = topk[np.isclose(topk.share, 0.10)].iloc[0]
    sel_dev = [d for d in r["dev"] if f"{d['model']} {d['set']}" == r["selected"]][0]
    m = {
        "NOrigins": f"{r['n_origins']:,}", "NEvents": f"{r['n_events']:,}",
        "PriceMissing": f"{r['price_missing']:,}", "PriceMissingPct": f(r["price_missing_pct"], 2),
        "ScrapedLater": f"{r['scraped_later']:,}", "ScrapedLaterPct": f(r["scraped_later_pct"], 2),
        "MaxDelay": str(r["max_delay"]), "Reappear": f"{r['reappear']:,}", "ReappearDen": f"{r['reappear_den']:,}",
        "ReappearPct": f(100 * r["reappear"] / max(r["reappear_den"], 1), 2),
        "NPredictors": str(r["n_predictors"]), "Threshold": f(r["threshold"], 6), "Calibration": r["calibration"],
        "DevROC": f(sel_dev["roc_auc"]), "DevPR": f(sel_dev["pr_auc"]), "DevBrier": f(sel_dev["brier"], 5),
        "TtenN": f"{t10['n']:,}", "TtenEvents": f"{t10['events']:,}",
        "TtenROC": f(t10["roc_auc"]), "TtenPR": f(t10["pr_auc"]), "TtenBrier": f(t10["brier"], 5),
        "TtenBrierCal": f(r["t10_cal"]["brier"], 5),
        "TtenROClo": f(r["t10_ci_low"]["roc_auc"]), "TtenROChi": f(r["t10_ci_high"]["roc_auc"]),
        "TtenPRlo": f(r["t10_ci_low"]["pr_auc"]), "TtenPRhi": f(r["t10_ci_high"]["pr_auc"]),
        "TP": str(r["confusion"]["TP"]), "FP": f"{r['confusion']['FP']:,}",
        "FN": str(r["confusion"]["FN"]), "TN": f"{r['confusion']['TN']:,}",
        "TopTenN": f"{int(top10.listings):,}", "TopTenEvents": str(int(top10.events)),
        "TopTenPrec": f(100 * top10.precision, 2), "TopTenRecall": f(100 * top10.recall, 2), "TopTenLift": f(top10.lift, 2),
        "SecN": f"{r['secondary']['n']:,}", "SecChoice": r["secondary"]["choice"],
        "SecMAE": f(r["secondary"]["mae"], 5), "SecRMSE": f(r["secondary"]["rmse"], 5), "SecRtwo": f(r["secondary"]["r2"], 6),
    }
    lines = ["% AUTO-GENERATED by 09_pipeline/run_all.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\Res{k}}}{{{v}}}" for k, v in m.items()]
    (OUT / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(summ, sets, lofo, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = OUT / "figures"
    fd.mkdir(exist_ok=True)
    s = summ[summ.model == "xgb"].set_index("set")
    names = [n for n in list(sets) + list(lofo) if n in s.index]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].barh(names, s.loc[names, "roc_auc"], color="0.35"); ax[0].set_xlabel("Macro ROC-AUC (T4-T9)")
    ax[1].barh(names, s.loc[names, "pr_auc"], color="0.35"); ax[1].set_xlabel("Macro PR-AUC (T4-T9)")
    ax[1].set_yticklabels([])
    lo = s.loc[names, "roc_auc"].min()
    ax[0].set_xlim(lo - 0.02, s.loc[names, "roc_auc"].max() + 0.01)
    for a in ax: a.invert_yaxis()
    fig.tight_layout(); fig.savefig(fd / "fig_dev_feature_sets.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(dec.index, 100 * dec.rate, color="0.35", label="Observed")
    ax.plot(dec.index, 100 * dec.mean_p, "k.-", label="Mean predicted")
    ax.set_xlabel("Predicted-risk decile (T10)"); ax.set_ylabel("Disappearance rate (%)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fd / "fig_t10_deciles.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    global CFG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--force", choices=STAGES, help="rebuild from this stage onward")
    a = ap.parse_args()
    CFG = yaml.safe_load(open(a.config, encoding="utf-8"))
    OUT = Path(CFG["paths"]["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(CFG["model"]["seed"])
    done = {"manifest": (OUT / "SNAPSHOT_MANIFEST.csv").exists(), "panel": (OUT / "panel.parquet").exists(),
            "features": (OUT / "features.parquet").exists(), "models": (OUT / "RESULTS.json").exists()}
    if a.force:
        for s in STAGES[STAGES.index(a.force):]:
            done[s] = False
    for s, fn in zip(STAGES, [stage_manifest, stage_panel, stage_features, stage_models]):
        if done[s]:
            log(f"stage {s}: cached, skipped (use --force {s} to rebuild)")
            continue
        log(f"stage {s}: start"); t0 = time.time(); fn(); log(f"stage {s}: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
