#!/usr/bin/env python
"""
Revision add-ons for the MDPI Data manuscript (frozen v1 results are NOT changed).

Computes, from files that already exist in the project:
  R1  paired feature-family comparisons (per-window wins + TA-cluster paired bootstrap CIs)
  R2  label robustness: "confirmed disappearance" (absent at t+1 AND t+2) for T4-T9
  R3  construction heterogeneity: A2-vs-A1 and A6-vs-A7 gains by TA consent-intensity tercile
  R4  TA-level association between construction context and disappearance rates
  R5  a SHAREABLE data package that respects Inside Airbnb's "do not republish the data" policy:
        - no listing-level rows, no listing/host ids, no coordinates, no text
        - TA x transition aggregates (cells with < MIN_CELL listings suppressed)
        - TA x transition construction / rental context layer (Stats NZ, MBIE: CC BY)
        - metrics, ablation and bootstrap tables, frozen model spec
        - a SYNTHETIC listing-level sample (same schema, no real records) so code can be run
        - a rebuild kit: snapshot manifest + instructions to rebuild the panel from Inside Airbnb
  and writes LaTeX files used by main.tex: revision_macros.tex, tab_family_paired.tex,
  tab_confirmed_label.tex, tab_construction_strata.tex.

Run from the project root (folder containing 01_raw/):  python 10_revision/revision_addons.py
or step by step in 10_revision/revision_addons.ipynb.  Fill CONFIG first (see discover()).
"""
import hashlib, json, re, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ============================================================================ CONFIG
CONFIG = {
    "airbnb_dir": "01_raw/airbnb",
    "out_dir": "10_revision_out",
    # --- development predictions of the FROZEN study (T4-T9, all feature sets) ---------------
    # Option A: one long file with a column identifying the specification
    "dev_pred_path": None,            # e.g. "05_results/block_I_B/dev_predictions_all.parquet"
    "dev_spec_col": "feature_set",    # values like "A0".."A10" (map below if names differ)
    "dev_model_col": "model",         # set None if the file contains XGBoost only
    "dev_model_value": "xgboost",
    # Option B: one file per specification  {"A1": "path", "A2": "path", ...}
    "dev_pred_files": {},
    "spec_name_map": {},              # e.g. {"xgb_A5_structured_full": "A5"}
    # --- frozen T10 predictions ------------------------------------------------------------------
    "t10_pred_path": None,            # e.g. "05_results/block_J/T10_predictions.parquet"
    # --- integrated panel / feature table (one row per listing-origin) --------------------------
    "panel_path": None,               # e.g. "03_processed/panel_features_T1_T10.parquet"
    # --- column names (shared by the files above; edit if different) --------------------------
    "id_col": "listing_id",
    "t_col": "transition",            # 1..10 or "T1".."T10"
    "y_col": "y_disappear",           # 1 = disappeared at next snapshot
    "p_col": "p_raw",                 # raw predicted probability
    "ta_col": "ta_code",
    "construction_cols": [],          # the 4 construction predictors, e.g. ["cons_log_units_3m", ...]
    "construction_level_col": None,   # one of them used for terciles, e.g. 12-month log units
    "rental_cols": [],                # the 2 rental predictors
    "bootstrap_reps": 400,
    "seed": 2026,
    "min_cell": 10,                   # suppress TA x transition cells with fewer listings
}
PAIRS = [  # (baseline, augmented, family label, type)
    ("A0", "A1", "Rental market", "add"), ("A1", "A2", "Construction", "add"),
    ("A2", "A3", "Seasonality", "add"), ("A3", "A4", "Calendar", "add"),
    ("A4", "A5", "Review history", "add"), ("A5", "A6", "Review semantics", "add"),
    ("A7", "A6", "Construction", "remove"), ("A8", "A6", "Seasonality", "remove"),
    ("A9", "A6", "Calendar", "remove"), ("A10", "A6", "All review information", "remove"),
]
DEV_WINDOWS = [4, 5, 6, 7, 8, 9]
FORBIDDEN = re.compile(r"(^id$|listing_id|host_id|host_name|reviewer|comment|name$|url|picture|latitude|longitude|^lat$|^lon|description|neighborhood_overview|license)", re.I)


# ============================================================================ helpers
def out():
    o = Path(CONFIG["out_dir"]); o.mkdir(parents=True, exist_ok=True); return o


def log(m):
    print(m, flush=True)
    with open(out() / "revision_log.txt", "a", encoding="utf-8") as f:
        f.write(m + "\n")


def read_any(p, columns=None):
    p = Path(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, usecols=columns, low_memory=False)


def tnum(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")


def discover(root="."):
    """Print candidate result/feature files and their columns, to help fill CONFIG."""
    rows = []
    for d in ("03_processed", "05_results", "02_intermediate"):
        for p in sorted(Path(root, d).rglob("*")):
            if p.suffix.lower() not in (".parquet", ".csv") or p.stat().st_size < 1000:
                continue
            try:
                cols = list(pd.read_parquet(p).columns) if p.suffix == ".parquet" else list(pd.read_csv(p, nrows=0).columns)
            except Exception as e:
                cols = [f"<unreadable: {e}>"]
            rows.append({"file": p.as_posix(), "MB": round(p.stat().st_size / 1e6, 1), "n_cols": len(cols),
                         "columns": ", ".join(map(str, cols[:25])) + (" ..." if len(cols) > 25 else "")})
    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 200)
    return df


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def fmt(x, d=4):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


# ============================================================================ loading
def load_dev():
    C = CONFIG
    cols = None
    if C["dev_pred_files"]:
        parts = []
        for spec, p in C["dev_pred_files"].items():
            d = read_any(p); d["spec"] = spec; parts.append(d)
        dev = pd.concat(parts, ignore_index=True)
    elif C["dev_pred_path"]:
        dev = read_any(C["dev_pred_path"])
        if C["dev_model_col"] and C["dev_model_col"] in dev:
            dev = dev[dev[C["dev_model_col"]].astype(str).str.lower().str.contains(C["dev_model_value"].lower())]
        dev["spec"] = dev[C["dev_spec_col"]].astype(str).map(lambda s: C["spec_name_map"].get(s, s))
        dev["spec"] = dev["spec"].str.extract(r"(A\d+)")[0].fillna(dev["spec"])
    else:
        raise SystemExit("Set CONFIG['dev_pred_path'] or CONFIG['dev_pred_files'] (use discover()).")
    dev = dev.rename(columns={C["id_col"]: "id", C["t_col"]: "t", C["y_col"]: "y", C["p_col"]: "p"})
    dev["t"] = tnum(dev["t"]); dev["id"] = dev["id"].astype(str)
    dev = dev[dev["t"].isin(DEV_WINDOWS)][["spec", "t", "id", "y", "p"]]
    log(f"dev predictions: {len(dev):,} rows; specs {sorted(dev.spec.unique())}")
    return dev


def load_panel():
    C = CONFIG
    if not C["panel_path"]:
        raise SystemExit("Set CONFIG['panel_path'] (use discover()).")
    want = [C["id_col"], C["t_col"], C["ta_col"], C["y_col"]] + C["construction_cols"] + C["rental_cols"]
    pan = read_any(C["panel_path"])
    miss = [c for c in want if c not in pan.columns]
    if miss:
        raise SystemExit(f"panel is missing columns {miss}; edit CONFIG column names")
    pan = pan[want].rename(columns={C["id_col"]: "id", C["t_col"]: "t", C["ta_col"]: "ta", C["y_col"]: "y"})
    pan["t"] = tnum(pan["t"]); pan["id"] = pan["id"].astype(str); pan["ta"] = pan["ta"].astype(str)
    log(f"panel: {len(pan):,} rows, transitions {sorted(pan.t.dropna().unique().tolist())}")
    return pan


# ============================================================================ R1
def family_paired(dev, pan):
    C = CONFIG
    rng = np.random.default_rng(C["seed"])
    ta = pan[["id", "t", "ta"]]
    d = dev.merge(ta, on=["id", "t"], how="left")
    if d["ta"].isna().any():
        log(f"WARNING: {d.ta.isna().sum():,} dev rows without TA; they are kept as one 'unknown' cluster")
        d["ta"] = d["ta"].fillna("unknown")
    tas = np.array(sorted(d["ta"].unique()))
    draws = [rng.choice(tas, size=len(tas), replace=True) for _ in range(C["bootstrap_reps"])]
    rows = []
    for base, aug, fam, kind in PAIRS:
        if base not in set(d.spec) or aug not in set(d.spec):
            log(f"skip {base} vs {aug}: specification not found"); continue
        a = d[d.spec == base].set_index(["t", "id"])
        b = d[d.spec == aug].set_index(["t", "id"])
        j = a[["y", "p", "ta"]].join(b[["p"]], rsuffix="_b", how="inner").reset_index()
        per = []
        for w, g in j.groupby("t"):
            r0, p0 = metrics(g.y.values, g.p.values); r1, p1 = metrics(g.y.values, g.p_b.values)
            per.append({"t": w, "droc": r1 - r0, "dpr": p1 - p0})
        per = pd.DataFrame(per)
        # paired TA-cluster bootstrap: one TA draw per replicate, applied to every window
        groups = {w: g.groupby("ta").indices for w, g in j.groupby("t")}
        arrs = {w: (g.y.values, g.p.values, g.p_b.values) for w, g in j.groupby("t")}
        bs_roc, bs_pr = [], []
        for dr in draws:
            dr_roc, dr_pr = [], []
            for w in arrs:
                idx = np.concatenate([groups[w][k] for k in dr if k in groups[w]] or [np.array([], int)])
                if len(idx) == 0:
                    continue
                y, p0, p1 = (x[idx] for x in arrs[w])
                r0, q0 = metrics(y, p0); r1, q1 = metrics(y, p1)
                dr_roc.append(r1 - r0); dr_pr.append(q1 - q0)
            bs_roc.append(np.nanmean(dr_roc)); bs_pr.append(np.nanmean(dr_pr))
        sign = 1   # both pair types are defined as (spec with the family) - (spec without it)
        rows.append({"family": fam, "comparison": f"{aug} vs {base}" if kind == "add" else f"{aug} vs {base} (removed)",
                     "type": kind, "macro_droc": sign * per.droc.mean(), "macro_dpr": sign * per.dpr.mean(),
                     "wins_roc": int((per.droc > 0).sum()), "wins_pr": int((per.dpr > 0).sum()), "n_windows": len(per),
                     "droc_lo": np.nanpercentile(bs_roc, 2.5), "droc_hi": np.nanpercentile(bs_roc, 97.5),
                     "dpr_lo": np.nanpercentile(bs_pr, 2.5), "dpr_hi": np.nanpercentile(bs_pr, 97.5)})
        log(f"R1 {fam:24s} {aug} vs {base}: dROC {per.droc.mean():+.4f} [{rows[-1]['droc_lo']:+.4f},{rows[-1]['droc_hi']:+.4f}] "
            f"wins {rows[-1]['wins_roc']}/{len(per)}; dPR {per.dpr.mean():+.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(out() / "R1_family_paired.csv", index=False)
    return res


# ============================================================================ R2
def listing_ids_by_snapshot():
    snaps = sorted(p for p in Path(CONFIG["airbnb_dir"]).iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    ids = []
    for s in snaps:
        f = sorted(s.glob("listings.csv*"))[0]
        ids.append(set(pd.read_csv(f, usecols=["id"], dtype=str)["id"].str.replace(r"\.0$", "", regex=True)))
    log(f"R2: read ids of {len(snaps)} snapshots")
    return ids


def confirmed_label(dev, spec="A5"):
    ids = listing_ids_by_snapshot()
    d = dev[dev.spec == spec].copy()
    d["t2_present"] = [i in ids[t + 1] if t + 1 < len(ids) else np.nan for i, t in zip(d.id, d.t.astype(int))]
    d["t1_present_raw"] = [i in ids[t] for i, t in zip(d.id, d.t.astype(int))]
    agree = (d.y.astype(int) == (~d.t1_present_raw).astype(int)).mean()
    log(f"R2: stored label agrees with raw next-snapshot absence for {100*agree:.3f}% of rows")
    d["y_conf"] = ((d.y == 1) & (d.t2_present == False)).astype(int)       # absent at t+1 and t+2
    d["temp"] = (d.y == 1) & (d.t2_present == True)                      # returned at t+2
    rows = []
    for w, g in d.groupby("t"):
        r0, p0 = metrics(g.y.values, g.p.values)
        r1, p1 = metrics(g.y_conf.values, g.p.values)
        h = g[~g.temp]
        r2, p2 = metrics(h.y.values, h.p.values)
        rows.append({"t": int(w), "n": len(g), "events": int(g.y.sum()), "confirmed": int(g.y_conf.sum()),
                     "returned_t2": int(g.temp.sum()), "roc_orig": r0, "pr_orig": p0, "roc_conf": r1, "pr_conf": p1,
                     "roc_excl_returned": r2, "pr_excl_returned": p2,
                     "mean_p_returned": g.loc[g.temp, "p"].mean(), "mean_p_confirmed": g.loc[g.y_conf == 1, "p"].mean(),
                     "mean_p_persist": g.loc[g.y == 0, "p"].mean()})
    r = pd.DataFrame(rows)
    r.to_csv(out() / "R2_confirmed_label.csv", index=False)
    log("R2:\n" + r.round(4).to_string(index=False))
    return r


# ============================================================================ R3
def construction_strata(dev, pan):
    C = CONFIG
    lev = C["construction_level_col"]
    if not lev:
        log("R3 skipped: set CONFIG['construction_level_col']"); return None
    tl = pan.groupby(["t", "ta"])[lev].first().reset_index()
    # terciles of TA consent intensity, computed within each transition across TAs (not listing-weighted)
    tl["stratum"] = tl.groupby("t")[lev].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["Low", "Middle", "High"]))
    d = dev.merge(pan[["id", "t", "ta"]], on=["id", "t"], how="left").merge(tl[["t", "ta", "stratum"]], on=["t", "ta"], how="left")
    rows = []
    for base, aug in (("A1", "A2"), ("A7", "A6")):
        if base not in set(d.spec) or aug not in set(d.spec):
            continue
        a = d[d.spec == base].set_index(["t", "id"]); b = d[d.spec == aug].set_index(["t", "id"])
        j = a[["y", "p", "stratum"]].join(b[["p"]], rsuffix="_b", how="inner").reset_index()
        for st, g in j.groupby("stratum", observed=True):
            per = []
            for w, h in g.groupby("t"):
                r0, p0 = metrics(h.y.values, h.p.values); r1, p1 = metrics(h.y.values, h.p_b.values)
                per.append((r1 - r0, p1 - p0, len(h), h.y.mean()))
            per = np.array(per, dtype=float)
            rows.append({"comparison": f"{aug} vs {base}", "stratum": st, "n_rows": int(per[:, 2].sum()),
                         "event_rate": float(np.average(per[:, 3], weights=per[:, 2])),
                         "macro_droc": np.nanmean(per[:, 0]), "macro_dpr": np.nanmean(per[:, 1]),
                         "wins_roc": int((per[:, 0] > 0).sum()), "n_windows": len(per)})
    r = pd.DataFrame(rows)
    r.to_csv(out() / "R3_construction_strata.csv", index=False)
    log("R3:\n" + r.round(4).to_string(index=False))
    return r


# ============================================================================ R4
def ta_association(pan):
    C = CONFIG
    cols = C["construction_cols"]
    if not cols:
        log("R4 skipped: set CONFIG['construction_cols']"); return None
    cell = pan[pan.t.between(1, 9)].groupby(["t", "ta"]).agg(n=("y", "size"), rate=("y", "mean"),
                                                         **{c: (c, "first") for c in cols}).reset_index()
    cell = cell[cell.n >= 30]
    rng = np.random.default_rng(C["seed"])
    tas = cell.ta.unique()
    rows = []
    for c in cols:
        z = cell.dropna(subset=[c]).copy()
        # within-transition ranks remove period effects; Spearman on demeaned ranks
        z["rx"] = z.groupby("t")[c].rank(pct=True); z["ry"] = z.groupby("t")["rate"].rank(pct=True)
        z["rx"] -= z.groupby("t")["rx"].transform("mean"); z["ry"] -= z.groupby("t")["ry"].transform("mean")
        rho = np.corrcoef(z.rx, z.ry)[0, 1]
        bs = []
        g = {k: v for k, v in z.groupby("ta")}
        for _ in range(C["bootstrap_reps"]):
            s = pd.concat([g[k] for k in rng.choice(tas, len(tas)) if k in g])
            bs.append(np.corrcoef(s.rx, s.ry)[0, 1] if len(s) > 3 else np.nan)
        rows.append({"construction_feature": c, "cells": len(z), "within_transition_spearman": rho,
                     "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5)})
    r = pd.DataFrame(rows)
    r.to_csv(out() / "R4_ta_association.csv", index=False)
    log("R4:\n" + r.round(3).to_string(index=False))
    return r


# ============================================================================ R5 shareable package
def sha(p):
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest(); return h


def shareable_package(dev, pan):
    C = CONFIG
    S = out() / "shareable_data"; S.mkdir(exist_ok=True)
    k = C["min_cell"]
    # 1) TA x transition aggregates (no listing rows)
    agg = pan.groupby(["t", "ta"]).agg(n_listings=("y", "size"), disappear_rate=("y", "mean")).reset_index()
    if C["t10_pred_path"]:
        t10 = read_any(C["t10_pred_path"]).rename(columns={C["id_col"]: "id", C["p_col"]: "p"})
        t10["id"] = t10["id"].astype(str)
        t10 = t10.merge(pan[pan.t == 10][["id", "ta"]], on="id", how="left")
        m = t10.groupby("ta")["p"].mean().rename("t10_mean_predicted_risk").reset_index(); m["t"] = 10
        agg = agg.merge(m, on=["t", "ta"], how="left")
    a5 = dev[dev.spec == "A5"].merge(pan[["id", "t", "ta"]], on=["id", "t"], how="left")
    md = a5.groupby(["t", "ta"])["p"].mean().rename("dev_mean_predicted_risk_A5").reset_index()
    agg = agg.merge(md, on=["t", "ta"], how="left")
    small = agg.n_listings < k
    agg.loc[small, [c for c in agg.columns if c not in ("t", "ta")]] = np.nan
    agg["suppressed_lt_min_cell"] = small
    agg.rename(columns={"t": "transition", "ta": "ta2025_code"}).to_csv(S / "ta_transition_aggregates.csv", index=False)
    # 2) contextual layer (Stats NZ consents, MBIE bonds): TA x transition, CC BY sources
    ctx_cols = C["construction_cols"] + C["rental_cols"]
    if ctx_cols:
        ctx = pan.groupby(["t", "ta"])[ctx_cols].first().reset_index()
        ctx.rename(columns={"t": "transition", "ta": "ta2025_code"}).to_csv(S / "ta_transition_context_layer.csv", index=False)
    # 3) metrics tables (copy R1-R4 + per-window dev metrics)
    pw = [{"spec": s, "transition": int(w), **dict(zip(("roc_auc", "pr_auc"), metrics(g.y.values, g.p.values))),
           "n": len(g), "events": int(g.y.sum())} for (s, w), g in dev.groupby(["spec", "t"])]
    pd.DataFrame(pw).to_csv(S / "dev_metrics_by_window.csv", index=False)
    for f in ("R1_family_paired.csv", "R2_confirmed_label.csv", "R3_construction_strata.csv", "R4_ta_association.csv"):
        if (out() / f).exists():
            shutil.copy2(out() / f, S / f)
    # 4) synthetic listing-level sample: columns resampled independently (breaks every real record)
    rng = np.random.default_rng(C["seed"])
    n = 5000
    full = read_any(C["panel_path"])
    syn_src = full[[c for c in full.columns if not FORBIDDEN.search(str(c)) and c != C["id_col"]]]
    syn = pd.DataFrame({"synthetic_id": [f"S{i:05d}" for i in range(n)]})
    for c in syn_src.columns:
        syn[c] = syn_src[c].sample(n=n, replace=True, random_state=int(rng.integers(1e9))).values
    syn.to_csv(S / "SYNTHETIC_listing_origin_sample.csv", index=False)
    # 5) privacy / policy scan of everything in the package
    viol = []
    for f in S.glob("*.csv"):
        cols = pd.read_csv(f, nrows=0).columns
        bad = [c for c in cols if FORBIDDEN.search(str(c)) and c != "synthetic_id"]
        if bad:
            viol.append((f.name, bad))
    if viol:
        raise SystemExit(f"POLICY CHECK FAILED, forbidden columns present: {viol}")
    # 6) README + checksums
    (S / "DATA_README.md").write_text(DATA_README.format(k=k), encoding="utf-8")
    man = [{"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f)} for f in sorted(S.iterdir()) if f.is_file() and f.name != "CHECKSUMS.csv"]
    pd.DataFrame(man).to_csv(S / "CHECKSUMS.csv", index=False)
    log(f"R5: shareable package with {len(man)} files written to {S} (policy scan passed)")


DATA_README = """# Shareable data products (derived; no Inside Airbnb records republished)

Inside Airbnb asks users not to republish its data. This package therefore contains **no listing-level
records, no listing or host identifiers, no coordinates and no review text**. It contains:

| File | Content | Source licence |
|---|---|---|
| ta_transition_aggregates.csv | Listings, observed next-snapshot disappearance rate and mean predicted risk per TA x transition; cells with < {k} listings suppressed | derived aggregates (cite Inside Airbnb) |
| ta_transition_context_layer.csv | Release-aware building-consent and rental-bond features per TA x transition, exactly as used by the model | Stats NZ (CC BY 4.0), MBIE (CC BY 3.0 NZ) |
| dev_metrics_by_window.csv | ROC-AUC/PR-AUC per specification and development window | derived |
| R1-R4 *.csv | Paired family comparisons, confirmed-label robustness, construction strata, TA-level association | derived |
| SYNTHETIC_listing_origin_sample.csv | 5,000 synthetic rows with the panel schema; every column resampled independently, so no row is a real listing. For running the code only; results on it are meaningless | synthetic |
| CHECKSUMS.csv | SHA-256 of every file | |

To reproduce listing-level results, obtain the Inside Airbnb New Zealand snapshots listed in the snapshot
manifest (current files from insideairbnb.com/get-the-data; older ones by archived-data request), verify their
SHA-256 against the manifest, and run the published code. Attribution: Inside Airbnb (insideairbnb.com).
"""


# ============================================================================ LaTeX outputs
def write_latex(r1, r2, r3, r4):
    o = out()
    M = ["% AUTO-GENERATED by 10_revision/revision_addons.py -- do not edit"]
    def mac(name, val):
        M.append(f"\\newcommand{{\\Rev{name}}}{{{val}}}")
    if r1 is not None and len(r1):
        L = [r"\begin{table}[H]", r"\begin{adjustwidth}{-\extralength}{0cm}", r"\centering\footnotesize",
             r"\caption{Paired feature-family comparisons on development windows T4--T9 (XGBoost, fixed hyperparameters). $\Delta$ = specification with the family minus without it; macro average over windows; wins = windows with a positive difference; 95\% intervals from a paired Territorial-Authority cluster bootstrap.}",
             r"\label{tab:family_paired}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Family & Comparison & $\Delta$ROC-AUC (95\% CI) & Wins & $\Delta$PR-AUC (95\% CI) & Wins \\", r"\midrule"]
        for _, x in r1.iterrows():
            L.append(f"{x.family} & {x.comparison} & {x.macro_droc:+.4f} ({x.droc_lo:+.4f}, {x.droc_hi:+.4f}) & {x.wins_roc}/{x.n_windows} & "
                     f"{x.macro_dpr:+.4f} ({x.dpr_lo:+.4f}, {x.dpr_hi:+.4f}) & {x.wins_pr}/{x.n_windows} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
        (o / "tab_family_paired.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
        c = r1[(r1.family == "Construction") & (r1.type == "add")]
        if len(c):
            c = c.iloc[0]
            mac("ConsDroc", f"{c.macro_droc:+.4f}"); mac("ConsDrocLo", f"{c.droc_lo:+.4f}"); mac("ConsDrocHi", f"{c.droc_hi:+.4f}")
            mac("ConsDpr", f"{c.macro_dpr:+.4f}"); mac("ConsDprLo", f"{c.dpr_lo:+.4f}"); mac("ConsDprHi", f"{c.dpr_hi:+.4f}")
            mac("ConsWins", f"{c.wins_roc}"); mac("ConsWinsPr", f"{c.wins_pr}"); mac("NWin", f"{c.n_windows}")
        c = r1[(r1.family == "Calendar") & (r1.type == "add")]
        if len(c):
            c = c.iloc[0]; mac("CalDroc", f"{c.macro_droc:+.4f}"); mac("CalDrocLo", f"{c.droc_lo:+.4f}"); mac("CalDrocHi", f"{c.droc_hi:+.4f}"); mac("CalWins", f"{c.wins_roc}")
    if r2 is not None and len(r2):
        L = [r"\begin{table}[H]", r"\begin{adjustwidth}{-\extralength}{0cm}", r"\centering\footnotesize",
             r"\caption{Label robustness of the A5 development predictions. ``Confirmed'' disappearance requires absence at both $t{+}1$ and $t{+}2$; ``excluding returned'' drops disappeared listings observed again at $t{+}2$.}",
             r"\label{tab:confirmed_label}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Window & Events & Confirmed & Returned & ROC (orig.) & ROC (confirmed) & PR (orig.) & PR (confirmed) \\", r"\midrule"]
        for _, x in r2.iterrows():
            L.append(f"T{int(x.t)} & {int(x.events):,} & {int(x.confirmed):,} & {int(x.returned_t2):,} & {fmt(x.roc_orig)} & {fmt(x.roc_conf)} & {fmt(x.pr_orig)} & {fmt(x.pr_conf)} \\\\")
        L.append(r"\midrule")
        L.append(f"Macro & & & & {fmt(r2.roc_orig.mean())} & {fmt(r2.roc_conf.mean())} & {fmt(r2.pr_orig.mean())} & {fmt(r2.pr_conf.mean())} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
        (o / "tab_confirmed_label.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
        mac("ConfRocOrig", fmt(r2.roc_orig.mean())); mac("ConfRoc", fmt(r2.roc_conf.mean()))
        mac("ConfPrOrig", fmt(r2.pr_orig.mean())); mac("ConfPr", fmt(r2.pr_conf.mean()))
        mac("ExclRoc", fmt(r2.roc_excl_returned.mean()))
        mac("ReturnedShare", f"{100 * r2.returned_t2.sum() / r2.events.sum():.2f}")
        mac("MeanPReturned", fmt(r2.mean_p_returned.mean())); mac("MeanPConfirmed", fmt(r2.mean_p_confirmed.mean()))
        mac("MeanPPersist", fmt(r2.mean_p_persist.mean()))
    if r3 is not None and len(r3):
        L = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{Development gain from construction context by tercile of Territorial-Authority consent intensity (terciles formed across TAs within each transition).}",
             r"\label{tab:construction_strata}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Comparison & Consent tercile & Rows & Event rate (\%) & $\Delta$ROC-AUC & $\Delta$PR-AUC & Wins \\", r"\midrule"]
        for _, x in r3.iterrows():
            L.append(f"{x.comparison} & {x.stratum} & {int(x.n_rows):,} & {100 * x.event_rate:.2f} & {x.macro_droc:+.4f} & {x.macro_dpr:+.4f} & {x.wins_roc}/{x.n_windows} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (o / "tab_construction_strata.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
        h = r3[(r3.comparison == "A2 vs A1") & (r3.stratum == "High")]
        lo = r3[(r3.comparison == "A2 vs A1") & (r3.stratum == "Low")]
        if len(h) and len(lo):
            mac("ConsHighDroc", f"{h.iloc[0].macro_droc:+.4f}"); mac("ConsLowDroc", f"{lo.iloc[0].macro_droc:+.4f}")
    if r4 is not None and len(r4):
        best = r4.iloc[r4.within_transition_spearman.abs().argmax()]
        mac("AssocFeature", str(best.construction_feature).replace("_", r"\_"))
        mac("AssocRho", f"{best.within_transition_spearman:+.3f}"); mac("AssocLo", f"{best.ci_lo:+.3f}"); mac("AssocHi", f"{best.ci_hi:+.3f}")
        mac("AssocCells", f"{int(best.cells):,}")
        parts = []
        for _, x in r4.iterrows():
            nm = str(x.construction_feature).replace("_", r"\_")
            parts.append(f"{nm}: {x.within_transition_spearman:+.3f} ({x.ci_lo:+.3f}, {x.ci_hi:+.3f})")
        rng_txt = ", ".join(parts)
        mac("AssocAll", rng_txt)
    (o / "revision_macros.tex").write_text("\n".join(M) + "\n", encoding="utf-8")
    log(f"LaTeX: revision_macros.tex + tables written to {o}")


# ============================================================================ main
def main():
    dev = load_dev(); pan = load_panel()
    r1 = family_paired(dev, pan)
    r2 = confirmed_label(dev, "A5")
    r3 = construction_strata(dev, pan)
    r4 = ta_association(pan)
    write_latex(r1, r2, r3, r4)
    shareable_package(dev, pan)
    log("DONE")


if __name__ == "__main__":
    main()
