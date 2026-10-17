#!/usr/bin/env python
"""
R6 -- Construction extension for the MDPI Data manuscript (post hoc, exploratory).

Nothing here changes the frozen A5 model, its threshold or any reported T10 result. All analyses use
files that already exist in the project and are reported in full, whatever their sign.

  E0  Alignment audit: re-derives the release-aware consent month of every transition from the raw
      Stats NZ consent file and checks it reproduces the construction features stored in the panel.
  E1  Pipeline-stage (lag) structure at area level: consents 0-3, 0-12, 12-24 and 24-36 months before
      the release-aware reference month (12-36 months ~ dwellings that have been completed).
  E2  Scale: (a) partial association controlling for TA size (log listings), (b) consents per 1,000
      dwellings (2023 Census dwelling stock; optional file).
  E3  Outcome robustness: association with the CONFIRMED exit rate (absent at t+1 and t+2).
  E4  Listing type: area association and listing-level gain (A2 vs A1, A6 vs A7) by room type.
  E5  Area-level monitoring: rolling-origin forecast of TA x transition disappearance rates with and
      without the four model construction features (same expanding design as the listing model).
  FIG Figure 7 (two panels) for the manuscript.

Run from the project root (the folder that contains 01_raw/):
    python 10_revision/construction_extension.py
Outputs: 10_revision_out/R6_construction/ (CSV, figure, R6_SUMMARY.txt). Send R6_SUMMARY.txt back.
"""
import re, sys, time, unicodedata, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================ CONFIG
CONFIG = {
    "airbnb_dir": "01_raw/airbnb",
    "out_dir": "10_revision_out/R6_construction",
    "shareable_dir": "10_revision_out/shareable_data",
    # same files as revision_addons (R1-R4)
    "panel_path": "10_revision_out/panel_small.parquet",
    "dev_pred_path": "05_results/development_modelling_IB/predictions/IB_all_development_predictions.parquet",
    "dev_model_col": "model", "dev_model_value": "xgb", "dev_spec_col": "feature_set",
    # file with room_type per listing-origin (any backbone parquet that has transition, listing_id, room_type)
    "roomtype_path": "02_intermediate/integration/listing_origin_backbone_T1_T10_with_TA.parquet",
    # Stats NZ monthly building consents by TA (raw). None = search 01_raw for *consent* files.
    "consents_file": None,
    # 2023 Census dwelling counts by TA (optional). None = search 01_raw for *dwelling*/*census* files.
    "dwellings_file": None,
    # column names
    "id_col": "listing_id", "t_col": "transition", "y_col": "y_disappear", "p_col": "p_disappear",
    "ta_col": "ta_code",
    "cons3_col": "construction_dwelling_3m_log", "cons12_col": "construction_dwelling_12m_log",
    "construction_cols": ["construction_dwelling_3m_log", "construction_dwelling_12m_log",
                          "construction_momentum_3m_vs_prior9m", "construction_multiunit_share_12m"],
    "rental_cols": ["rental_geomean_rent_log", "rental_geomean_rent_change_12m_log"],
    "min_cell": 30,            # TA x transition cells with fewer listings are not analysed (as in R4)
    "bootstrap_reps": 400,
    "seed": 2026,
}
DEV_WINDOWS = [4, 5, 6, 7, 8, 9]

# Stats NZ 2025 Territorial Authority codes (fallback only; a project file with ta_name is used first)
TA2025 = {"001": "Far North", "002": "Whangarei", "003": "Kaipara", "011": "Thames-Coromandel", "012": "Hauraki",
          "013": "Waikato", "015": "Matamata-Piako", "016": "Hamilton", "017": "Waipa", "018": "Otorohanga",
          "019": "South Waikato", "020": "Waitomo", "021": "Taupo", "022": "Western Bay of Plenty", "023": "Tauranga",
          "024": "Rotorua", "025": "Whakatane", "026": "Kawerau", "027": "Opotiki", "028": "Gisborne", "029": "Wairoa",
          "030": "Hastings", "031": "Napier", "032": "Central Hawke's Bay", "033": "New Plymouth", "034": "Stratford",
          "035": "South Taranaki", "036": "Ruapehu", "037": "Whanganui", "038": "Rangitikei", "039": "Manawatu",
          "040": "Palmerston North", "041": "Tararua", "042": "Horowhenua", "043": "Kapiti Coast", "044": "Porirua",
          "045": "Upper Hutt", "046": "Lower Hutt", "047": "Wellington", "048": "Masterton", "049": "Carterton",
          "050": "South Wairarapa", "051": "Tasman", "052": "Nelson", "053": "Marlborough", "054": "Kaikoura",
          "055": "Buller", "056": "Grey", "057": "Westland", "058": "Hurunui", "059": "Waimakariri",
          "060": "Christchurch", "062": "Selwyn", "063": "Ashburton", "064": "Timaru", "065": "Mackenzie",
          "066": "Waimate", "067": "Chatham Islands", "068": "Waitaki", "069": "Central Otago",
          "070": "Queenstown-Lakes", "071": "Dunedin", "072": "Clutha", "073": "Southland", "074": "Gore",
          "075": "Invercargill", "076": "Auckland"}
ALIASES = {"hutt": "lowerhutt", "wanganui": "whanganui", "queenstownlakes": "queenstownlakes"}

LINES = []   # summary lines


# ============================================================================ helpers
def out():
    o = Path(CONFIG["out_dir"]); o.mkdir(parents=True, exist_ok=True); return o


def log(m=""):
    print(m, flush=True)
    LINES.append(str(m))


def tnum(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(district|city|territory|council)\b", " ", s)
    s = re.sub(r"[^a-z]", "", s)
    return ALIASES.get(s, s)


def read_any(p, columns=None):
    p = Path(p)
    if p.suffix == ".parquet":
        if columns:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(p).names)
            columns = [c for c in columns if c in have]
        return pd.read_parquet(p, columns=columns)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, header=None, dtype=str)
    return pd.read_csv(p, usecols=columns, low_memory=False)


def within_ranks(df, col, by):
    r = df.groupby(by)[col].rank(pct=True)
    return r - r.groupby([df[b] for b in by]).transform("mean")


def wt_corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def resid(y, Z):
    ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    r = np.full(len(y), np.nan)
    if ok.sum() > Z.shape[1] + 2:
        A = np.column_stack([np.ones(ok.sum()), Z[ok]])
        b = np.linalg.lstsq(A, y[ok], rcond=None)[0]
        r[ok] = y[ok] - A @ b
    return r


def assoc(cells, xcol, ycol, by=("t",), controls=(), reps=None, seed=None):
    """Within-group Spearman (ranks demeaned within group); optional partial on controls; TA bootstrap."""
    reps = reps or CONFIG["bootstrap_reps"]
    z = cells.dropna(subset=[xcol, ycol] + list(controls)).copy()
    by = list(by)
    z["_rx"] = within_ranks(z, xcol, by); z["_ry"] = within_ranks(z, ycol, by)
    cz = [f"_rz{i}" for i in range(len(controls))]
    for c, nm in zip(controls, cz):
        z[nm] = within_ranks(z, c, by)

    def stat(d):
        x, y = d["_rx"].values, d["_ry"].values
        if cz:
            Z = d[cz].values; x, y = resid(x, Z), resid(y, Z)
        return wt_corr(x, y)

    rho = stat(z)
    rng = np.random.default_rng(seed or CONFIG["seed"])
    g = {k: v for k, v in z.groupby("ta")}
    tas = np.array(sorted(g))
    bs = []
    for _ in range(reps):
        s = pd.concat([g[k] for k in rng.choice(tas, len(tas))])
        bs.append(stat(s))
    return {"rho": rho, "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5),
            "cells": len(z), "tas": len(tas)}


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def f3(x):
    return "NA" if x is None or pd.isna(x) else f"{x:+.3f}"


def f4(x):
    return "NA" if x is None or pd.isna(x) else f"{x:+.4f}"


# ============================================================================ loading
def snapshots():
    d = Path(CONFIG["airbnb_dir"])
    return sorted(p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))


def load_panel():
    C = CONFIG
    want = [C["id_col"], C["t_col"], C["ta_col"], C["y_col"]] + C["construction_cols"] + C["rental_cols"]
    pan = read_any(C["panel_path"], columns=want)
    miss = [c for c in want if c not in pan.columns]
    if miss:
        raise SystemExit(f"panel {C['panel_path']} is missing {miss}; edit CONFIG")
    pan = pan.rename(columns={C["id_col"]: "id", C["t_col"]: "t", C["ta_col"]: "ta", C["y_col"]: "y"})
    pan["t"] = tnum(pan["t"]).astype(int)
    pan["id"] = pan["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    pan["ta"] = pan["ta"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
    log(f"panel: {len(pan):,} listing-origins, T{pan.t.min()}-T{pan.t.max()}, {pan.ta.nunique()} TA codes, "
        f"disappearances {int(pan.y.sum()):,}")
    p = Path(C["roomtype_path"])
    if p.exists():
        rt = read_any(p, columns=[C["t_col"], C["id_col"], "room_type"])
        if "room_type" in rt.columns:
            rt = rt.rename(columns={C["t_col"]: "t", C["id_col"]: "id"})
            rt["t"] = tnum(rt["t"]).astype(int); rt["id"] = rt["id"].astype(str).str.replace(r"\.0$", "", regex=True)
            pan = pan.merge(rt.drop_duplicates(["t", "id"]), on=["t", "id"], how="left")
            log(f"room_type merged: {pan.room_type.notna().mean():.4%} of origins matched")
    if "room_type" not in pan.columns:
        log("room_type not available -> E4 area/listing split skipped")
    return pan


def load_dev():
    C = CONFIG
    p = Path(C["dev_pred_path"])
    if not p.exists():
        log(f"dev predictions not found ({p}) -> E4 listing-level part skipped"); return None
    d = read_any(p, columns=[C["t_col"], C["id_col"], C["y_col"], C["p_col"], C["dev_model_col"], C["dev_spec_col"]])
    if C["dev_model_col"] in d:
        d = d[d[C["dev_model_col"]].astype(str).str.lower().str.contains(C["dev_model_value"].lower())]
    d["spec"] = d[C["dev_spec_col"]].astype(str).str.extract(r"(A\d+)")[0]
    d = d.rename(columns={C["t_col"]: "t", C["id_col"]: "id", C["y_col"]: "y", C["p_col"]: "p"})
    d["t"] = tnum(d["t"]).astype(int); d["id"] = d["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d[d.t.isin(DEV_WINDOWS) & d.spec.isin(["A1", "A2", "A6", "A7"])][["spec", "t", "id", "y", "p"]]
    log(f"dev predictions (A1, A2, A6, A7): {len(d):,} rows")
    return d


def ta_name_map():
    """{normalised name: ta_code} from a project file if possible, else the Stats NZ 2025 list."""
    cands = sorted(Path("02_intermediate").rglob("*.parquet")) if Path("02_intermediate").exists() else []
    cands = sorted([p for p in cands if "embedding" not in p.as_posix()],
                   key=lambda p: (not re.search(r"integration|_TA|context", p.as_posix()), p.stat().st_size))
    for p in cands[:60]:
        try:
            import pyarrow.parquet as pq
            names = pq.read_schema(p).names
        except Exception:
            continue
        if "ta_name" in names and CONFIG["ta_col"] in names:
            t = pd.read_parquet(p, columns=[CONFIG["ta_col"], "ta_name"]).drop_duplicates().dropna()
            m = {norm(n): str(c).replace(".0", "").zfill(3) for c, n in zip(t[CONFIG["ta_col"]], t["ta_name"])}
            if len(m) >= 60:
                log(f"TA names from {p.as_posix()} ({len(m)} TAs)")
                return m
    log("TA names: Stats NZ 2025 TA list (built-in)")
    return {norm(n): c for c, n in TA2025.items()}


# ============================================================================ consents
def find_file(cfg_key, patterns):
    if CONFIG[cfg_key]:
        return Path(CONFIG[cfg_key])
    hits = []
    for d in ("01_raw", "02_intermediate", "03_processed", "07_provenance"):
        if Path(d).exists():
            for p in Path(d).rglob("*"):
                if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".parquet") and \
                        any(re.search(pt, p.name, re.I) for pt in patterns):
                    hits.append(p)
    hits = sorted(hits, key=lambda p: (not p.as_posix().startswith("01_raw"), -p.stat().st_size))
    if hits:
        log(f"{cfg_key}: candidates " + "; ".join(h.as_posix() for h in hits[:6]) + f" -> using {hits[0].as_posix()}")
        return hits[0]
    return None


MONTH_RE = re.compile(r"^\s*(\d{4})\s*[-M/]\s*(\d{1,2})")


def to_month(v):
    m = MONTH_RE.match(str(v))
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Period(f"{m.group(1)}-{int(m.group(2)):02d}", freq="M")
    try:
        return pd.Period(pd.Timestamp(str(v)), freq="M")
    except Exception:
        return None


def parse_consents(path, names):
    """Return tidy DataFrame: ta (code), month (Period), kind (label), value."""
    if Path(path).suffix == ".csv":       # rows of unequal length (Infoshare titles/footers) -> csv module
        import csv
        with open(path, newline="", encoding="utf-8-sig", errors="ignore") as fh:
            rows = list(csv.reader(fh))
        width = max(len(r) for r in rows)
        raw = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str).replace({"": np.nan})
    elif Path(path).suffix == ".parquet":
        raw = pd.read_parquet(path)
        raw = pd.concat([pd.DataFrame([raw.columns], columns=raw.columns), raw.astype(str)], ignore_index=True)
        raw.columns = range(raw.shape[1])
    else:
        raw = read_any(path)
    raw = raw.astype(str).replace({"nan": np.nan, "None": np.nan})
    first = raw.iloc[:, 0].astype(str)
    is_wide = first.str.fullmatch(r"\s*\d{4}M\d{2}\s*").sum() >= 12
    recs = []
    if is_wide:   # Infoshare: first column 2020M01..., header rows above
        dmask = first.str.fullmatch(r"\s*\d{4}M\d{2}\s*")
        top = raw.iloc[: int(dmask.values.argmax())].dropna(how="all")
        top = top.ffill(axis=1)
        data = raw[dmask]
        months = data.iloc[:, 0].map(to_month)
        for c in raw.columns[1:]:
            cells = [str(x) for x in top[c].tolist() if isinstance(x, str) and x.strip()]
            code = next((names[norm(x)] for x in cells if norm(x) in names), None)
            if code is None:
                continue
            kind = " | ".join(x for x in cells if norm(x) not in names) or "value"
            v = pd.to_numeric(data[c].str.replace(",", ""), errors="coerce")
            recs.append(pd.DataFrame({"ta": code, "month": months.values, "kind": kind, "value": v.values}))
        fmt = "Infoshare wide"
    else:         # long (Aotearoa Data Explorer / tidy): find header row, then area, time, value columns
        hdr_row = next(i for i in range(min(20, len(raw))) if raw.iloc[i].notna().sum() >= 3)
        df = raw.iloc[hdr_row + 1:].copy(); df.columns = [str(c) for c in raw.iloc[hdr_row]]
        df = df.dropna(how="all")
        score = {c: df[c].map(lambda x: norm(x) in names).mean() for c in df.columns}
        area = max(score, key=score.get)
        tscore = {c: df[c].map(lambda x: to_month(x) is not None and bool(MONTH_RE.match(str(x)))).mean()
                  for c in df.columns if c != area}
        tcol = max(tscore, key=tscore.get)
        vcand = [c for c in df.columns if c not in (area, tcol)]
        vnum = {c: pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").notna().mean() for c in vcand}
        pref = [c for c in vcand if re.search(r"obs_value|^value$|count|number", c, re.I) and vnum[c] > .5]
        vcol = pref[0] if pref else max(vnum, key=vnum.get)
        dims = [c for c in vcand if c != vcol and 1 < df[c].nunique() <= 30 and vnum[c] < .5]
        if score[area] < .3 or tscore[tcol] < .5:
            raise SystemExit(f"Could not recognise the consent file layout ({path}). Columns: {list(df.columns)}")
        df["_kind"] = df[dims].astype(str).agg(" | ".join, axis=1) if dims else "value"
        df = df[df[area].map(lambda x: norm(x) in names)]
        recs.append(pd.DataFrame({"ta": df[area].map(lambda x: names[norm(x)]).values,
                                  "month": df[tcol].map(to_month).values, "kind": df["_kind"].values,
                                  "value": pd.to_numeric(df[vcol].astype(str).str.replace(",", ""), errors="coerce").values}))
        fmt = f"long (area='{area}', time='{tcol}', value='{vcol}', dims={dims})"
    if not recs:
        raise SystemExit(f"No TA columns recognised in {path}; check the file or set CONFIG['consents_file']")
    t = pd.concat(recs, ignore_index=True).dropna(subset=["month"])
    log(f"consents file: {Path(path).as_posix()} | format {fmt} | {t.ta.nunique()} TAs | "
        f"{t.month.min()}..{t.month.max()} | {len(t):,} values | kinds: {sorted(t.kind.unique())[:10]}")
    return t


def to_wide(t, ks):
    """ta x month matrix of summed values (NaN when every component is missing); works on any pandas version."""
    d = t[t.kind.isin(ks)]
    return d.groupby(["month", "ta"])["value"].sum(min_count=1).unstack("ta").sort_index()


def total_variants(t):
    """Candidate definitions of 'new dwellings'. E0 alignment picks the one that reproduces the panel exactly."""
    kinds = sorted(t.kind.unique())
    bad = re.compile(r"value|\$|floor|area|square|metre|percent|%|non-residential|nonresidential", re.I)
    dw = [k for k in kinds if not bad.search(k)]
    tot = [k for k in dw if re.search(r"\btotal\b|\ball\b|dwelling units|new dwellings", k, re.I)]
    parts = [k for k in dw if k not in tot]
    V = {}
    for k in dw:                                   # every single dwelling series (incl. a total such as 'Dwelling units')
        V[f"single:{k}"] = to_wide(t, [k])
    if len(parts) > 1:                             # sum of dwelling types (excluding totals and non-residential)
        V["sum_of:" + ",".join(parts)] = to_wide(t, parts)
    houses = [k for k in parts if re.search(r"house", k, re.I) and not re.search(r"town", k, re.I)]
    return V, houses, parts


def window_sum(W, m, a, b):
    """Sum of months in (m-a, m-b]; NaN if any month is missing."""
    idx = pd.period_range(m - a + 1, m - b, freq="M")
    sub = W.reindex(idx)
    s = sub.sum(min_count=1)
    s[sub.isna().any()] = np.nan
    return s


def align(pan, t, origins):
    """E0: find the definition, transform and reference month that reproduce the stored features."""
    C = CONFIG
    V, houses, parts = total_variants(t)
    tl = pan.groupby(["t", "ta"])[[C["cons3_col"], C["cons12_col"]]].first().reset_index()
    best = None
    for vname, W in V.items():
        for tr_name, tr in (("log1p", np.log1p), ("log", lambda x: np.log(x.where(x > 0)))):
            rows = []
            for tt, om in origins.items():
                g = tl[tl.t == tt].set_index("ta")
                cand = []
                for k in range(0, 10):
                    m = om - k
                    if m - 11 < W.index.min():
                        continue
                    v3, v12 = tr(window_sum(W, m, 3, 0)), tr(window_sum(W, m, 12, 0))
                    j = g.join(pd.DataFrame({"v3": v3, "v12": v12}), how="inner").replace([np.inf, -np.inf], np.nan)
                    j = j.dropna()
                    if len(j) < 10:
                        continue
                    d = np.maximum((j.v3 - j[C["cons3_col"]]).abs(), (j.v12 - j[C["cons12_col"]]).abs())
                    cand.append((float(d.median()), float(d.max()), float((d < 1e-6).mean()), str(m), len(j)))
                if cand:
                    c = min(cand)
                    rows.append({"t": tt, "origin_month": str(om), "ref_month": c[3], "median_abs_diff": c[0],
                                 "max_abs_diff": c[1], "share_exact": c[2], "tas_compared": c[4]})
            if not rows:
                continue
            r = pd.DataFrame(rows)
            score = (r.median_abs_diff.mean(), -r.share_exact.mean())
            if best is None or score < best[0]:
                best = (score, vname, tr_name, r)
    if best is None:
        raise SystemExit("E0 alignment failed: consent file does not overlap the panel months")
    _, vname, tr_name, r = best
    r.to_csv(out() / "E0_alignment_audit.csv", index=False)
    log("")
    log("E0 ALIGNMENT (raw consents -> stored panel features)")
    log(f"  definition: {vname} | transform: {tr_name}")
    log("  " + r.assign(median_abs_diff=r.median_abs_diff.map(lambda x: f"{x:.2e}"),
                        max_abs_diff=r.max_abs_diff.map(lambda x: f"{x:.2e}"),
                        share_exact=r.share_exact.map(lambda x: f"{x:.3f}")).to_string(index=False).replace("\n", "\n  "))
    ok = (r.share_exact >= 0.95).all()
    log(f"  verdict: {'EXACT reproduction for every transition' if ok else 'NOT exact -- check before using E1/E2 (see max_abs_diff)'}")
    W = V[vname]
    Wh = None
    if houses:
        Wh = to_wide(t, houses)
    ref = {int(x.t): pd.Period(x.ref_month, freq="M") for x in r.itertuples()}
    return W, Wh, ref, ok


def load_dwellings(names):
    p = find_file("dwellings_file", [r"dwelling", r"census"])
    if p is None or not p.exists():
        log("E2b: no 2023 Census dwelling file found -> per-dwelling intensity skipped "
            "(place it in 01_raw/census/, see README)")
        return None
    raw = pd.read_excel(p, dtype=str) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p, dtype=str, encoding_errors="ignore")
    score = {c: raw[c].map(lambda x: norm(x) in names).mean() for c in raw.columns}
    area = max(score, key=score.get)
    if score[area] < .3:
        log(f"E2b: could not find a TA name column in {p} -> skipped"); return None
    d = raw[raw[area].map(lambda x: norm(x) in names)].copy()
    ycols = [c for c in d.columns if re.search(r"year|census", c, re.I) and d[c].astype(str).str.contains("2023").any()]
    if ycols:
        d = d[d[ycols[0]].astype(str).str.contains("2023")]
    num = {c: pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce") for c in d.columns if c != area}
    vcols = [c for c, v in num.items() if v.notna().mean() > .8 and not re.search(r"year|code", c, re.I)]
    if not vcols:
        log(f"E2b: no numeric dwelling column in {p} -> skipped"); return None
    vcol = next((c for c in vcols if re.search(r"value|dwelling|count|total", c, re.I)), vcols[-1])
    dims = [c for c in d.columns if c not in [area, vcol] + ycols and 1 < d[c].nunique() <= 20 and num[c].notna().mean() < .5]
    used = "all rows"
    if dims:
        lab = d[dims].astype(str).agg(" | ".join, axis=1)
        tot = lab.str.contains(r"total", case=False)
        if tot.any():
            d, used = d[tot], f"rows labelled {sorted(lab[tot].unique())[:3]}"
        else:
            used = f"sum over {sorted(lab.unique())[:6]}"
    d["ta"] = d[area].map(lambda x: names[norm(x)])
    s = pd.to_numeric(d[vcol].astype(str).str.replace(",", ""), errors="coerce").groupby(d["ta"]).sum(min_count=1)
    log(f"E2b dwelling stock: {p.as_posix()} | column '{vcol}' | {used} | {s.notna().sum()} TAs, total {s.sum():,.0f}")
    return s.rename("dwellings_2023")


# ============================================================================ cells
def confirmed(pan):
    snaps = snapshots()
    ids = []
    for s in snaps:
        f = sorted(s.glob("listings.csv*"))[0]
        ids.append(set(pd.read_csv(f, usecols=["id"], dtype=str)["id"].str.replace(r"\.0$", "", regex=True)))
    t = pan.t.values
    nxt = np.array([i in ids[k] if k < len(ids) else np.nan for i, k in zip(pan.id, t)], dtype=object)
    agree = float(np.mean(pan.y.values == (~nxt.astype(bool)).astype(int)))
    pres2 = np.array([(i in ids[k + 1]) if k + 1 < len(ids) else np.nan for i, k in zip(pan.id, t)], dtype=float)
    pan["y_conf"] = np.where(np.isnan(pres2), np.nan, ((pan.y.values == 1) & (pres2 == 0)).astype(float))
    log(f"E3: stored label agrees with raw next-snapshot absence for {100 * agree:.3f}% of origins; "
        f"confirmed exits T1-T9: {int(np.nansum(pan.y_conf)):,} of {int(pan.loc[pan.t <= 9, 'y'].sum()):,} disappearances")
    return pan


def build_cells(pan, W, Wh, ref, stock):
    C = CONFIG
    g = pan.groupby(["t", "ta"])
    cells = g.agg(n=("y", "size"), rate=("y", "mean"), conf_rate=("y_conf", "mean"),
                  **{c: (c, "first") for c in C["construction_cols"] + C["rental_cols"]}).reset_index()
    rows = []
    for tt, m in ref.items():
        d = pd.DataFrame({"c_0_3": window_sum(W, m, 3, 0), "c_0_12": window_sum(W, m, 12, 0),
                          "c_12_24": window_sum(W, m, 24, 12), "c_24_36": window_sum(W, m, 36, 24)})
        if Wh is not None:
            h12 = window_sum(Wh, m, 12, 0)
            d["multi_share_0_12"] = 1 - h12 / d.c_0_12.replace(0, np.nan)
        d["t"] = tt; rows.append(d.rename_axis("ta").reset_index())
    k = pd.concat(rows, ignore_index=True)
    cells = cells.merge(k, on=["t", "ta"], how="left")
    cells["log_n"] = np.log(cells.n)
    if stock is not None:
        cells = cells.merge(stock.rename_axis("ta").reset_index(), on="ta", how="left")
        for c in ("c_0_3", "c_0_12", "c_12_24", "c_24_36"):
            cells[c + "_per1000"] = 1000 * cells[c] / cells.dwellings_2023
        cells["log_dwellings"] = np.log(cells.dwellings_2023)
    prev = cells[["t", "ta", "rate", "n"]].copy(); prev["t"] += 1
    cells = cells.merge(prev.rename(columns={"rate": "prev_rate", "n": "prev_n"}), on=["t", "ta"], how="left")
    cells.loc[cells.prev_n < 10, "prev_rate"] = np.nan
    return cells


# ============================================================================ analyses
LAG_LABEL = {"c_0_3": "Consents 0-3 months", "c_0_12": "Consents 0-12 months",
             "c_12_24": "Consents 12-24 months", "c_24_36": "Consents 24-36 months",
             "multi_share_0_12": "Multi-unit share 0-12 months"}


def e1_e2_e3(cells, stock):
    C = CONFIG
    dev = cells[(cells.t <= 9) & (cells.n >= C["min_cell"])]
    hold = cells[(cells.t == 10) & (cells.n >= C["min_cell"])]
    rows = []
    def add(block, label, x, y="rate", data=dev, controls=()):
        a = assoc(data, x, y, controls=controls)
        rows.append({"block": block, "measure": label, "x": x, "outcome": y, "controls": "+".join(controls) or "-",
                     "sample": "T10" if data is hold else "T1-T9", **a})
    for x in [c for c in LAG_LABEL if c in cells]:
        add("E1 lag structure", LAG_LABEL[x], x)
    for x in [c for c in LAG_LABEL if c in cells]:
        add("E1 replication on T10", LAG_LABEL[x], x, data=hold)
    for x in ("c_0_12", "c_12_24", "c_24_36"):
        add("E2a size control", LAG_LABEL[x], x, controls=("log_n",))
    if stock is not None:
        for x in ("c_0_12", "c_12_24", "c_24_36"):
            add("E2b per 1,000 dwellings", LAG_LABEL[x] + " per 1,000 dwellings", x + "_per1000")
            add("E2b per 1,000 dwellings, size control", LAG_LABEL[x] + " per 1,000 dwellings", x + "_per1000",
                controls=("log_n",))
        add("E2b stock control", "Consents 0-12 months | dwelling stock", "c_0_12", controls=("log_dwellings",))
    for x in ("c_0_12", "c_12_24", "multi_share_0_12"):
        if x in cells:
            add("E3 confirmed exit", LAG_LABEL[x], x, y="conf_rate")
    r = pd.DataFrame(rows)
    r.to_csv(out() / "E1_E3_area_associations.csv", index=False)
    # persistence of consent intensity across pipeline stages (TA level, T1 reference month)
    t1 = cells[cells.t == cells.t.min()].dropna(subset=["c_0_12", "c_12_24", "c_24_36"])
    pers = t1[["c_0_12", "c_12_24", "c_24_36"]].rank().corr().round(3)
    pers.to_csv(out() / "E1_pipeline_persistence.csv")
    log("")
    log("E1-E3 AREA-LEVEL ASSOCIATIONS (within-transition Spearman, 95% TA-cluster bootstrap CI)")
    for _, x in r.iterrows():
        log(f"  [{x.block}] {x.measure:52s} y={x.outcome:9s} ctrl={x.controls:13s} {x['sample']:6s} "
            f"rho {f3(x.rho)} ({f3(x.lo)}, {f3(x.hi)})  cells {x.cells}, TAs {x.tas}")
    log(f"  persistence across pipeline stages (Spearman between TAs, T1): 0-12 vs 12-24 = {pers.loc['c_0_12', 'c_12_24']:.3f}; "
        f"0-12 vs 24-36 = {pers.loc['c_0_12', 'c_24_36']:.3f}")
    return r


def e4(pan, cells_fn, dev):
    C = CONFIG
    rows, lrows = [], []
    if "room_type" in pan.columns:
        top = pan.room_type.value_counts().index[:2].tolist()
        for rt in top:
            sub = pan[pan.room_type == rt]
            c = sub.groupby(["t", "ta"]).agg(n=("y", "size"), rate=("y", "mean")).reset_index()
            c = c.merge(cells_fn[["t", "ta", "c_0_12", "c_12_24"]], on=["t", "ta"], how="left")
            c = c[(c.t <= 9) & (c.n >= C["min_cell"])]
            for x in ("c_0_12", "c_12_24"):
                a = assoc(c, x, "rate")
                rows.append({"room_type": rt, "measure": LAG_LABEL[x], "share_of_origins": len(sub) / len(pan),
                             "event_rate": sub.y.mean(), **a})
        if dev is not None:
            d = dev.merge(pan[["t", "id", "room_type"]], on=["t", "id"], how="left")
            for base, aug in (("A1", "A2"), ("A7", "A6")):
                a = d[d.spec == base].set_index(["t", "id"]); b = d[d.spec == aug].set_index(["t", "id"])
                j = a[["y", "p", "room_type"]].join(b[["p"]], rsuffix="_b", how="inner").reset_index()
                for rt in top:
                    per = []
                    for w, h in j[j.room_type == rt].groupby("t"):
                        r0, q0 = metrics(h.y.values, h.p.values); r1, q1 = metrics(h.y.values, h.p_b.values)
                        per.append((r1 - r0, q1 - q0))
                    per = np.array(per)
                    lrows.append({"comparison": f"{aug} vs {base}", "room_type": rt,
                                  "macro_droc": np.nanmean(per[:, 0]), "macro_dpr": np.nanmean(per[:, 1]),
                                  "wins_roc": int((per[:, 0] > 0).sum()), "n_windows": len(per)})
    r, lr = pd.DataFrame(rows), pd.DataFrame(lrows)
    r.to_csv(out() / "E4_roomtype_area.csv", index=False); lr.to_csv(out() / "E4_roomtype_listing_gain.csv", index=False)
    log("")
    log("E4 ROOM TYPE")
    for _, x in r.iterrows():
        log(f"  area  {x.room_type:18s} ({x.share_of_origins:.1%} of origins, event rate {x.event_rate:.2%}) "
            f"{x.measure:24s} rho {f3(x.rho)} ({f3(x.lo)}, {f3(x.hi)}) cells {x.cells}")
    for _, x in lr.iterrows():
        log(f"  listing {x.comparison} {x.room_type:18s} dROC {f4(x.macro_droc)} (wins {x.wins_roc}/{x.n_windows}) dPR {f4(x.macro_dpr)}")
    return r, lr


def e5(cells):
    """Area-level rolling-origin forecast of TA x transition disappearance rates."""
    C = CONFIG
    cons = C["construction_cols"]; rent = C["rental_cols"]
    specs = {"M0 persistence only": None,
             "M1 persistence + size + rental": ["prev_rate", "log_n"] + rent,
             "M2 = M1 + construction": ["prev_rate", "log_n"] + rent + cons,
             "M3 size + rental": ["log_n"] + rent,
             "M4 = M3 + construction": ["log_n"] + rent + cons}
    data = cells[(cells.n >= C["min_cell"]) & cells.prev_rate.notna()].copy()
    preds = []
    for w in DEV_WINDOWS + [10]:
        tr = data[data.t < w]; te = data[data.t == w].copy()
        if len(tr) < 30 or len(te) < 10:
            continue
        for nm, cols in specs.items():
            if cols is None:
                te[nm] = te.prev_rate.values; continue
            med = tr[cols].median()
            def prep(d):
                X = d[cols].fillna(med)
                return X - X.groupby(d.t.values).transform("mean")
            Xtr, Xte = prep(tr), prep(te)
            ytr = tr.rate - tr.groupby("t").rate.transform("mean")
            A = np.column_stack([np.ones(len(Xtr)), Xtr.values]); sw = np.sqrt(tr.n.values)
            b = np.linalg.lstsq(A * sw[:, None], ytr.values * sw, rcond=None)[0]
            te[nm] = np.column_stack([np.ones(len(Xte)), Xte.values]) @ b
        preds.append(te[["t", "ta", "n", "rate"] + list(specs)])
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(out() / "E5_area_forecasts.csv", index=False)

    def score(d, nm):
        out_ = {}
        for w, g in d.groupby("t"):
            if len(g) >= 5:
                out_[w] = pd.Series(g[nm].values).rank().corr(pd.Series(g.rate.values).rank())
        return out_
    rows = []
    for nm in specs:
        s = score(P, nm)
        rows.append({"model": nm, **{f"T{w}": v for w, v in s.items()},
                     "dev_macro_spearman": np.nanmean([s[w] for w in DEV_WINDOWS if w in s]),
                     "T10_spearman": s.get(10, np.nan)})
    R = pd.DataFrame(rows)
    # paired TA bootstrap for construction increments (predictions fixed; intervals conditional on fitted models)
    rng = np.random.default_rng(C["seed"])
    tas = np.array(sorted(P.ta.unique())); gi = {k: v for k, v in P.groupby("ta")}
    pairs = [("M2 = M1 + construction", "M1 persistence + size + rental"), ("M4 = M3 + construction", "M3 size + rental")]
    inc = []
    for a, b in pairs:
        for samp, ws in (("T4-T9", DEV_WINDOWS), ("T10", [10])):
            pt = np.nanmean([score(P, a).get(w, np.nan) - score(P, b).get(w, np.nan) for w in ws])
            wins = sum((score(P, a).get(w, np.nan) - score(P, b).get(w, np.nan)) > 0 for w in ws)
            bs = []
            for _ in range(C["bootstrap_reps"]):
                s = pd.concat([gi[k] for k in rng.choice(tas, len(tas))]); s = s[s.t.isin(ws)]
                sa, sb = score(s, a), score(s, b)
                bs.append(np.nanmean([sa[w] - sb[w] for w in sa if w in sb]))
            inc.append({"comparison": f"{a.split()[0]} vs {b.split()[0]}", "sample": samp, "d_spearman": pt,
                        "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5), "wins": f"{wins}/{len(ws)}"})
    I = pd.DataFrame(inc)
    R.to_csv(out() / "E5_area_forecast_scores.csv", index=False); I.to_csv(out() / "E5_construction_increment.csv", index=False)
    log("")
    log(f"E5 AREA-LEVEL ROLLING FORECAST of TA x transition disappearance rate (cells >= {C['min_cell']} listings; "
        "within-transition Spearman between predicted and observed rates)")
    log("  " + R.round(3).to_string(index=False).replace("\n", "\n  "))
    log("  construction increment (paired TA bootstrap, predictions fixed):")
    for _, x in I.iterrows():
        log(f"    {x.comparison} {x['sample']:6s} dSpearman {f3(x.d_spearman)} ({f3(x.lo)}, {f3(x.hi)}) wins {x.wins}")
    return R, I


# ============================================================================ figure
def figure(cells, assoc_tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    d = cells[(cells.t <= 9) & (cells.n >= CONFIG["min_cell"])].dropna(subset=["c_0_12"]).copy()
    d["rx"] = d.groupby("t").c_0_12.rank(pct=True); d["ry"] = d.groupby("t").rate.rank(pct=True)
    ax[0].scatter(d.rx, d.ry, s=np.clip(d.n / 40, 6, 120), alpha=.35, color="#2F5D8A", edgecolor="none")
    bins = pd.cut(d.rx, np.linspace(0, 1, 6), include_lowest=True)
    m = d.groupby(bins, observed=True).agg(x=("rx", "mean"), y=("ry", "mean"))
    ax[0].plot(m.x, m.y, "-o", color="#B23A48", lw=2, ms=5, label="Quintile means")
    ax[0].set_xlabel("TA rank: dwellings consented, last 12 months")
    ax[0].set_ylabel("TA rank: listing disappearance rate")
    ax[0].set_title("(a) Area level, T1–T9 (within transition)")
    ax[0].legend(frameon=False, loc="upper left")
    t = assoc_tab[assoc_tab["sample"] == "T1-T9"].copy()
    keep = [("E1 lag structure", None), ("E2a size control", None), ("E2b per 1,000 dwellings", None), ("E3 confirmed exit", None)]
    t = t[t.block.isin([k for k, _ in keep])].reset_index(drop=True)
    lab = (t.measure.str.replace("Consents ", "").str.replace(" months", " m").str.replace(" per 1,000 dwellings", "/1,000 dw.")
           + t.block.map({"E1 lag structure": "", "E2a size control": " | size-adj.",
                          "E2b per 1,000 dwellings": "", "E3 confirmed exit": " | confirmed exit"}))
    y = np.arange(len(t))[::-1]
    col = t.block.map({"E1 lag structure": "#2F5D8A", "E2a size control": "#6C8EAD",
                       "E2b per 1,000 dwellings": "#3E8E7E", "E3 confirmed exit": "#B23A48"})
    ax[1].errorbar(t.rho, y, xerr=[t.rho - t.lo, t.hi - t.rho], fmt="none", ecolor="#888", lw=1.4, capsize=3)
    ax[1].scatter(t.rho, y, c=col, s=40, zorder=3)
    ax[1].axvline(0, color="#444", lw=.8, ls=":")
    ax[1].set_yticks(y); ax[1].set_yticklabels(lab, fontsize=9.5)
    ax[1].set_xlabel("Within-transition Spearman ρ (95% TA-cluster CI)")
    ax[1].set_title("(b) Pipeline stage, scale and outcome checks")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out() / f"F7_construction_area_level.{ext}", dpi=300, bbox_inches="tight")
    log(f"\nFIGURE: {out() / 'F7_construction_area_level.pdf'}")


# ============================================================================ main
def main():
    t0 = time.time()
    log("R6 construction extension | " + time.strftime("%Y-%m-%d %H:%M") + f" | pandas {pd.__version__}, numpy {np.__version__}")
    log("All analyses are post hoc and exploratory; the frozen model and T10 results are unchanged.")
    pan = load_panel()
    snaps = snapshots()
    origins = {k + 1: pd.Period(s.name, freq="M") for k, s in enumerate(snaps[:-1])}
    names = ta_name_map()
    cf = find_file("consents_file", [r"consent"])
    if cf is None:
        raise SystemExit("No Stats NZ consent file found under 01_raw/; set CONFIG['consents_file']")
    raw = parse_consents(cf, names)
    W, Wh, ref, exact = align(pan, raw, origins)
    if Wh is None:
        log("multi-unit series: no 'houses' category identified -> multi_share_0_12 not recomputed")
    stock = load_dwellings(names)
    pan = confirmed(pan)
    cells = build_cells(pan, W, Wh, ref, stock)
    shared = cells[cells.n >= 10][["t", "ta", "n", "rate", "conf_rate", "c_0_3", "c_0_12", "c_12_24", "c_24_36"] +
                                  [c for c in cells if c.endswith("_per1000")]]
    shared.rename(columns={"t": "transition", "ta": "ta2025_code", "n": "n_listings"}).to_csv(
        out() / "R6_ta_transition_construction_cells.csv", index=False)
    a = e1_e2_e3(cells, stock)
    dev = load_dev()
    e4(pan, cells, dev)
    e5(cells)
    try:
        figure(cells, a)
    except Exception as e:
        log(f"FIGURE failed: {e}")
    S = Path(CONFIG["shareable_dir"])
    if S.exists():
        for f in out().glob("*.csv"):
            if f.name != "E5_area_forecasts.csv":
                (S / f"R6_{f.name}" if not f.name.startswith("R6_") else S / f.name).write_bytes(f.read_bytes())
        log(f"copied R6 tables to {S} (aggregates only; build_supplementary.py will policy-scan them)")
    log(f"\nE0 exact reproduction: {exact} | runtime {time.time() - t0:.0f} s | DONE")
    (out() / "R6_SUMMARY.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n>>> Send me: {out() / 'R6_SUMMARY.txt'} and {out() / 'F7_construction_area_level.png'}")


if __name__ == "__main__":
    main()
