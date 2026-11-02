#!/usr/bin/env python
"""
Reviewer 1 analyses for the MDPI Data manuscript data-4668825 (post hoc; the frozen model is NOT changed).

  RV1  Comment 1: refit without the origin-month sine/cosine terms (and without month + horizon terms)
  RV2  Comment 2: rule-based benchmarks from calendar availability and review recency
  RV3  Comment 3: development results of logistic regression and the prevalence-only baseline
  RV4  Comment 4: how much of the area-level context is absorbed by the TA code
  RV5  Comment 5: construction consents normalised by dwelling stock (2023 Census)
  RV6  Comment 6: window-by-window comparison of the stored (frozen) and re-fitted development predictions
  RV7  Comment 7: role of Platt-calibrated probabilities in the screening workflow

Every refitted model is compared ONLY with the 53-predictor model refitted by the same code (RV1 "A5_ref"),
so that implementation differences do not enter any comparison.

Run from the project root (the folder that contains 01_raw/):
    python 10_revision/reviewer_R1_analyses.py            # all analyses
    python 10_revision/reviewer_R1_analyses.py --only RV3 RV6 RV7   # fast parts that need no refitting
Outputs: 10_revision_out/RV_reviewer1/  ->  send back RV_SUMMARY.txt
"""
import argparse, importlib.util, json, platform, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================ CONFIG
CONFIG = {
    "out_dir": "10_revision_out/RV_reviewer1",
    # stored development predictions of the frozen study (all models / feature sets, T4-T9)
    "dev_pred_path": "05_results/development_modelling_IB/predictions/IB_all_development_predictions.parquet",
    "dev_model_col": "model", "dev_spec_col": "feature_set",
    # frozen T10 predictions (raw probabilities). None = search 05_results for a T10 prediction file
    "t10_pred_path": None,
    # listing-origin feature table with ALL 53 A5 predictors for T1-T10. None = search the project
    "features_path": None,
    # the 53 A5 predictors. None = read from a frozen-spec json/csv found in 05_results, else set by hand
    "a5_features": None,
    # file that lists the 53 frozen A5 predictors (first existing file is used)
    "a5_features_files": ["05_results/FINAL_PUBLICATION_PACKAGE/01_frozen_53_predictor_evidence.csv",
                          "05_results/manuscript_final_L_to_O/tables/M_SHAP_53_frozen_features.csv",
                          "05_results/FINAL_PUBLICATION_PACKAGE/08_frozen_spec.json",
                          "05_results/development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json"],
    # label: 1 = disappeared. If y_col is absent, y = 1 - persistence_col
    "persistence_col": "next_period_persistence",
    "categorical": ["ta_code", "room_type", "property_type_group"],   # edit to the categorical A5 predictors
    # column names
    "id_col": "listing_id", "t_col": "transition", "y_col": "y_disappear", "p_col": "p_disappear",
    "ta_col": "ta_code",
    "calendar_rule_col": "open_ratio_w4",
    "recency_col": None,          # None = auto-detect (days since last review)
    "month_cols": None,           # None = auto-detect sine/cosine of the origin month
    "horizon_cols": None,         # None = auto-detect the two horizon-dependent predictors
    "construction_count_cols": ["construction_dwelling_3m_log", "construction_dwelling_12m_log"],
    "construction_cols": ["construction_dwelling_3m_log", "construction_dwelling_12m_log",
                          "construction_momentum_3m_vs_prior9m", "construction_multiunit_share_12m"],
    "rental_cols": ["rental_geomean_rent_log", "rental_geomean_rent_change_12m_log"],
    # 2023 Census dwellings by TA (CSV). Either columns ta_code,dwellings, or the Figure.NZ file used by R6.
    "dwellings_file": None,
    "construction_extension_py": "10_revision/construction_extension.py",   # reused to read the Census file
    # frozen XGBoost specification (manuscript Section 3.6)
    "xgb": dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=1.0, tree_method="hist"),
    "frozen_threshold": 0.02679064,
    "topk": [0.01, 0.05, 0.10, 0.20, 0.30],
    "bootstrap_reps": 400, "seed": 2026,
}
DEV = [4, 5, 6, 7, 8, 9]
TEST = 10

# Optional hook: if you have the ORIGINAL fitting function of the frozen pipeline, assign it here, e.g.
#   from my_pipeline import fit_predict_xgb ; FIT_FN = fit_predict_xgb
# It must accept (train_df, test_df, feature_list) and return raw probabilities for test_df.
FIT_FN = None


# ============================================================================ helpers
def out():
    o = Path(CONFIG["out_dir"]); o.mkdir(parents=True, exist_ok=True); return o


def log(m=""):
    print(m, flush=True)
    with open(out() / "RV_SUMMARY.txt", "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")


def read_any(p, columns=None):
    p = Path(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, usecols=columns, low_memory=False)


def cols_of(p):
    p = Path(p)
    try:
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(p).schema_arrow.names)
        return list(pd.read_csv(p, nrows=0).columns)
    except Exception:
        return []


def tnum(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64").values


def norm_id(s):
    return pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).values


def met(y, p):
    y = np.asarray(y); p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return dict(roc=np.nan, pr=np.nan, brier=np.nan)
    return dict(roc=roc_auc_score(y, p), pr=average_precision_score(y, p),
                brier=brier_score_loss(y, np.clip(p, 0, 1)) if np.nanmax(p) <= 1 and np.nanmin(p) >= 0 else np.nan)


def topk_recall(y, p, q):
    y = np.asarray(y); k = int(np.ceil(q * len(y)))
    idx = np.argsort(-np.asarray(p, float), kind="stable")[:k]
    return y[idx].sum() / max(y.sum(), 1), y[idx].sum(), k


def f4(x):
    return "--" if pd.isna(x) else f"{x:.4f}"


# ============================================================================ discovery
def find(patterns, must_cols=(), roots=("05_results", "03_processed", "02_intermediate", "10_revision_out")):
    hits = []
    for r in roots:
        if not Path(r).exists():
            continue
        for p in Path(r).rglob("*"):
            if p.suffix.lower() not in (".parquet", ".csv") or "embedding" in p.as_posix().lower():
                continue
            if patterns and not any(re.search(pt, p.as_posix(), re.I) for pt in patterns):
                continue
            c = cols_of(p)
            if all(m in c for m in must_cols):
                hits.append((p, len(c), p.stat().st_size))
    return sorted(hits, key=lambda h: (-h[1], -h[2]))


def resolve_config():
    C = CONFIG
    if not C["features_path"]:
        h = find([], must_cols=(C["id_col"], C["t_col"], C["calendar_rule_col"], C["construction_cols"][0]))
        if not h:
            raise SystemExit("Feature table not found: set CONFIG['features_path'] to the parquet with all 53 A5 "
                             "predictors (it must contain open_ratio_w4 and the construction columns).")
        C["features_path"] = h[0][0].as_posix()
    log(f"features table: {C['features_path']}")
    allc = cols_of(C["features_path"])
    if not C["a5_features"]:
        for f in C["a5_features_files"]:
            if not Path(f).exists():
                continue
            txt = Path(f).read_text(encoding="utf-8", errors="ignore")
            found = [c for c in allc if re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", txt)]
            found = [c for c in found if c not in (C["id_col"], C["t_col"], C["y_col"], C["p_col"], C["persistence_col"],
                                                   "evaluation_role", "snapshot_date", "anchor_date", "future_date")]
            log(f"{f}: {len(found)} feature-table columns named")
            if len(found) == 53:
                C["a5_features"] = found
                log(f"A5 predictor list read from {f}")
                break
    if not C["a5_features"]:
        for p in sorted(Path("05_results").rglob("*")) if Path("05_results").exists() else []:
            if p.suffix.lower() not in (".json", ".csv", ".txt") or p.stat().st_size > 5e6:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if C["calendar_rule_col"] not in txt or "A5" not in txt:
                continue
            names = [c for c in allc if re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", txt)]
            names = [c for c in names if c not in (C["id_col"], C["t_col"], C["y_col"], C["p_col"])]
            if len(names) == 53:
                C["a5_features"] = names
                log(f"A5 predictor list read from {p.as_posix()}")
                break
    log("A5 predictors (" + str(len(C["a5_features"] or [])) + "): " + ", ".join(C["a5_features"] or []))
    if not C["a5_features"] or len(C["a5_features"]) != 53:
        raise SystemExit("Could not identify the 53 A5 predictors automatically. Set CONFIG['a5_features'] to the "
                         "list of 53 column names (from the frozen specification file).")
    A5 = C["a5_features"]
    if not C["month_cols"]:
        C["month_cols"] = [c for c in A5 if re.search(r"month", c, re.I) and re.search(r"sin|cos", c, re.I)]
    if not C["horizon_cols"]:
        C["horizon_cols"] = [c for c in A5 if re.search(r"horizon|days_to_next|next_snapshot", c, re.I)]
    if not C["recency_col"]:
        r = [c for c in A5 if re.search(r"(days_)?since_last_review|last_review.*(days|age)|review_recency", c, re.I)]
        C["recency_col"] = r[0] if r else None
    C["categorical"] = [c for c in C["categorical"] if c in A5] or \
        [c for c in A5 if c == C["ta_col"] or re.search(r"room_type|property_type", c)]
    for k in ("month_cols", "horizon_cols"):
        if len(C[k]) != 2:
            raise SystemExit(f"Expected 2 {k}, found {C[k]}; set CONFIG['{k}'] by hand.")
    log(f"month terms: {C['month_cols']} | horizon terms: {C['horizon_cols']} | recency: {C['recency_col']} "
        f"| categorical: {C['categorical']}")
    if not C["t10_pred_path"]:
        h = find([r"T10|t10|final|holdout|block_J"], must_cols=(C["id_col"],))
        h = [x for x in h if any(re.search(r"^p_|prob|pred", c, re.I) for c in cols_of(x[0]))]
        C["t10_pred_path"] = h[0][0].as_posix() if h else None
    log(f"frozen T10 predictions: {C['t10_pred_path']}")


# ============================================================================ loading
def load_features():
    C = CONFIG
    have = cols_of(C["features_path"])
    want = list(dict.fromkeys([C["id_col"], C["t_col"], C["y_col"], C["persistence_col"]] + C["a5_features"]
                              + C["construction_cols"] + C["rental_cols"]))
    X = read_any(C["features_path"], columns=[c for c in want if c in have])
    if C["y_col"] not in X:
        if C["persistence_col"] not in X:
            raise SystemExit(f"Neither {C['y_col']} nor {C['persistence_col']} found in the feature table")
        X[C["y_col"]] = 1 - pd.to_numeric(X[C["persistence_col"]], errors="coerce")
        log(f"label: y = 1 - {C['persistence_col']}")
    X = X.rename(columns={C["id_col"]: "id", C["t_col"]: "t", C["y_col"]: "y"})
    # every non-numeric A5 predictor is treated as categorical (one-hot)
    for c in C["a5_features"]:
        if c in X and not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]) \
                and c not in C["categorical"]:
            C["categorical"].append(c)
    log(f"categorical predictors: {C['categorical']}")
    X["t"] = tnum(X["t"]); X["id"] = norm_id(X["id"])
    X["ta"] = X[C["ta_col"]].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
    for c in C["categorical"]:
        X[c] = X[c].astype(str)
    log(f"features: {len(X):,} listing-origins, T{X.t.min()}-T{X.t.max()}, disappearances {int(X.y.sum()):,}")
    return X


def load_dev_all():
    C = CONFIG
    p = Path(C["dev_pred_path"])
    if not p.exists():
        log(f"stored development predictions not found: {p}"); return None
    d = read_any(p)
    d = d.rename(columns={C["id_col"]: "id", C["t_col"]: "t", C["y_col"]: "y", C["p_col"]: "p"})
    d["t"] = tnum(d["t"]); d["id"] = norm_id(d["id"])
    d["model"] = d[C["dev_model_col"]].astype(str) if C["dev_model_col"] in d else "xgb"
    d["spec"] = d[C["dev_spec_col"]].astype(str) if C["dev_spec_col"] in d else "?"
    return d[d.t.isin(DEV)]


def frozen_dev_a5(d):
    m = d.model.str.lower().str.contains("xgb") & d.spec.str.extract(r"(A\d+)")[0].eq("A5")
    s = d[m]
    if s.spec.nunique() > 1:   # keep the structured A5 specification
        s = s[s.spec == s.spec.value_counts().index[0]]
    return s[["t", "id", "y", "p"]]


def load_t10():
    C = CONFIG
    if not C["t10_pred_path"]:
        return None
    d = read_any(C["t10_pred_path"])
    pc = C["p_col"] if C["p_col"] in d else next(c for c in d.columns if re.search(r"raw|^p_|prob", c, re.I))
    d = d.rename(columns={C["id_col"]: "id", pc: "p"})
    if C["y_col"] in d:
        d = d.rename(columns={C["y_col"]: "y"})
    if C["t_col"] in d:
        d["t"] = tnum(d[C["t_col"]]); d = d[d.t == TEST]
    d["id"] = norm_id(d["id"])
    log(f"frozen T10 predictions: {len(d):,} rows, probability column '{pc}'")
    return d


# ============================================================================ model fitting
def _prep(tr, te, cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    cat = [c for c in cols if c in CONFIG["categorical"]]
    num = [c for c in cols if c not in cat]
    ct = ColumnTransformer([
        ("n", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), num),
        ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    f = lambda d: d[cols].assign(**{c: pd.to_numeric(d[c], errors="coerce") for c in num})
    return ct.fit_transform(f(tr)), ct.transform(f(te))


def fit_predict(tr, te, cols, kind="xgb"):
    if FIT_FN is not None and kind == "xgb":
        return np.asarray(FIT_FN(tr, te, cols))
    A, B = _prep(tr, te, cols)
    if kind == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(**CONFIG["xgb"], random_state=CONFIG["seed"], n_jobs=-1, eval_metric="logloss")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
        m = LogisticRegression(max_iter=3000)
    m.fit(A, tr["y"].values)
    return m.predict_proba(B)[:, 1]


def rolling(X, cols, name, kind="xgb", with_t10=True):
    """Expanding-origin development T4-T9 (+ optional post hoc T10). Returns long predictions.
    Predictions are cached in 10_revision_out/RV_cache so that both reviewer scripts reuse the same refits."""
    cache = Path("10_revision_out/RV_cache"); cache.mkdir(parents=True, exist_ok=True)
    impl = "origfit" if (FIT_FN is not None and kind == "xgb") else "rvfit"
    key = cache / f"{name}__{kind}__{impl}__{len(cols)}.parquet"
    if key.exists():
        r = pd.read_parquet(key)
        if set(r.t.unique()) >= set(DEV + ([TEST] if with_t10 else [])):
            log(f"  loaded {name:28s} from cache ({key.name})"); return r
    t0 = time.time(); rows = []
    for w in DEV + ([TEST] if with_t10 else []):
        tr, te = X[X.t < w], X[X.t == w]
        rows.append(pd.DataFrame({"spec": name, "t": w, "id": te.id.values, "y": te.y.values,
                                  "ta": te.ta.values, "p": fit_predict(tr, te, cols, kind)}))
    r = pd.concat(rows, ignore_index=True)
    r.to_parquet(key, index=False)
    log(f"  fitted {name:28s} ({len(cols)} predictors, {kind}) in {time.time() - t0:.0f}s")
    return r


def summarise(P, ref="A5_ref"):
    rows = []
    for s, g in P.groupby("spec", sort=False):
        dv = [met(h.y, h.p) for _, h in g[g.t.isin(DEV)].groupby("t")]
        row = {"spec": s, "dev_roc": np.nanmean([m["roc"] for m in dv]), "dev_pr": np.nanmean([m["pr"] for m in dv]),
               "dev_brier": np.nanmean([m["brier"] for m in dv])}
        h = g[g.t == TEST]
        if len(h):
            m = met(h.y, h.p); row.update(t10_roc=m["roc"], t10_pr=m["pr"], t10_brier=m["brier"],
                                          t10_recall_top10=topk_recall(h.y.values, h.p.values, 0.10)[0])
        rows.append(row)
    S = pd.DataFrame(rows)
    if ref in set(P.spec):
        S = S.merge(paired_bootstrap(P, ref), on="spec", how="left")
    return S


def paired_bootstrap(P, ref):
    """Macro dev difference (spec - ref) with a paired TA-cluster bootstrap (same TA draw in every window)."""
    rng = np.random.default_rng(CONFIG["seed"])
    D = P[P.t.isin(DEV)]
    base = D[D.spec == ref].set_index(["t", "id"])[["y", "ta", "p"]]
    tas = np.array(sorted(D.ta.unique()))
    draws = [rng.choice(tas, len(tas)) for _ in range(CONFIG["bootstrap_reps"])]
    rows = []
    for s in D.spec.unique():
        if s == ref:
            continue
        j = base.join(D[D.spec == s].set_index(["t", "id"])[["p"]], rsuffix="_s", how="inner").reset_index()
        per = {w: g for w, g in j.groupby("t")}
        point = [(met(g.y, g.p_s)["roc"] - met(g.y, g.p)["roc"], met(g.y, g.p_s)["pr"] - met(g.y, g.p)["pr"])
                 for g in per.values()]
        idx = {w: g.groupby("ta").indices for w, g in per.items()}
        bs = []
        for dr in draws:
            dd = []
            for w, g in per.items():
                ii = np.concatenate([idx[w][k] for k in dr if k in idx[w]] or [np.array([], int)])
                if len(ii) and g.y.values[ii].min() != g.y.values[ii].max():
                    a, b = met(g.y.values[ii], g.p_s.values[ii]), met(g.y.values[ii], g.p.values[ii])
                    dd.append((a["roc"] - b["roc"], a["pr"] - b["pr"]))
            if dd:
                bs.append(np.nanmean(dd, axis=0))
        bs = np.array(bs)
        rows.append({"spec": s, "d_roc": np.mean([x[0] for x in point]), "d_roc_lo": np.percentile(bs[:, 0], 2.5),
                     "d_roc_hi": np.percentile(bs[:, 0], 97.5), "wins_roc": int(sum(x[0] > 0 for x in point)),
                     "d_pr": np.mean([x[1] for x in point]), "d_pr_lo": np.percentile(bs[:, 1], 2.5),
                     "d_pr_hi": np.percentile(bs[:, 1], 97.5), "wins_pr": int(sum(x[1] > 0 for x in point))})
    return pd.DataFrame(rows)


def show(S, title):
    log(f"\n{title}")
    keep = [c for c in ["spec", "dev_roc", "dev_pr", "dev_brier", "d_roc", "d_roc_lo", "d_roc_hi", "wins_roc",
                        "d_pr", "d_pr_lo", "d_pr_hi", "wins_pr", "t10_roc", "t10_pr", "t10_recall_top10"] if c in S]
    log(S[keep].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ============================================================================ RV1 + RV4 + RV5 (refits)
_REFIT_CACHE = {}


def refits(X, which):
    C = CONFIG
    A5 = C["a5_features"]
    drop = lambda *cs: [c for c in A5 if c not in set(sum(map(list, cs), []))]
    specs = {"A5_ref": A5}
    if "RV1" in which:
        specs["A5_no_month"] = drop(C["month_cols"])
        specs["A5_horizon_free"] = drop(C["horizon_cols"])
        specs["A5_no_month_no_horizon"] = drop(C["month_cols"], C["horizon_cols"])
    if "RV4" in which:
        specs["A5_no_TA"] = drop([C["ta_col"]])
        specs["A5_no_construction"] = drop(C["construction_cols"])
        specs["A5_no_TA_no_construction"] = drop([C["ta_col"]], C["construction_cols"])
        specs["A5_no_TA_no_context"] = drop([C["ta_col"]], C["construction_cols"], C["rental_cols"])
    if "RV5" in which and "cons_norm_3m" in X:
        specs["A5_construction_per_dwelling"] = [
            {C["construction_count_cols"][0]: "cons_norm_3m", C["construction_count_cols"][1]: "cons_norm_12m"}.get(c, c)
            for c in A5]
        specs["A5_no_construction"] = drop(C["construction_cols"])
    P = []
    for name, cols in specs.items():
        if name not in _REFIT_CACHE:
            _REFIT_CACHE[name] = rolling(X, cols, name)
        P.append(_REFIT_CACHE[name])
    return pd.concat(P, ignore_index=True)


def rv1(X):
    log("\n" + "=" * 100 + "\nRV1  Comment 1: origin-month sine/cosine terms")
    P = refits(X, ["RV1"])
    S = summarise(P); S.to_csv(out() / "RV1_no_month.csv", index=False)
    show(S, "Same implementation; differences are spec - A5_ref (dev macro T4-T9, paired TA bootstrap). "
            "T10 columns are post hoc.")
    T = X[X.t <= 9].groupby("t").agg(month_sin=(CONFIG["month_cols"][0], "nunique"),
                                     month_cos=(CONFIG["month_cols"][1], "nunique"), rate=("y", "mean"))
    log("distinct month-term values per transition (1 = the terms only index the transition):\n" + T.to_string())
    return S


def rv4(X):
    log("\n" + "=" * 100 + "\nRV4  Comment 4: area-level context versus the TA code")
    C = CONFIG
    ctx = C["construction_cols"] + C["rental_cols"]
    cell = X.groupby(["t", "ta"])[ctx].first().reset_index()
    rows = []
    for c in ctx:
        z = cell.dropna(subset=[c])
        tot = z[c].var(ddof=0)
        within_ta = (z[c] - z.groupby("ta")[c].transform("mean")).var(ddof=0)
        within_t = (z[c] - z.groupby("t")[c].transform("mean")).var(ddof=0)
        rows.append({"feature": c, "cells": len(z), "share_between_TA": 1 - within_ta / tot,
                     "share_within_TA_over_time": within_ta / tot, "share_between_transitions": 1 - within_t / tot})
    V = pd.DataFrame(rows); V.to_csv(out() / "RV4_variance_by_TA.csv", index=False)
    log("Share of the variance of each area-level predictor (TA x transition cells, T1-T10) that lies between TAs:")
    log(V.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    P = refits(X, ["RV4"])
    S = summarise(P); S.to_csv(out() / "RV4_TA_code.csv", index=False)
    show(S, "Refits with and without the TA code (differences are spec - A5_ref)")
    # construction gain with and without the TA code, computed directly
    for a, b, lab in (("A5_ref", "A5_no_construction", "construction gain WITH TA code"),
                      ("A5_no_TA", "A5_no_TA_no_construction", "construction gain WITHOUT TA code")):
        Q = P[P.spec.isin([a, b])].copy(); Q["spec"] = Q.spec.map({a: "with", b: "without"})
        g = paired_bootstrap(Q, "without")          # (with construction) - (without construction)
        if len(g):
            r = g.iloc[0]
            log(f"  {lab:36s}: dROC {r.d_roc:+.4f} ({r.d_roc_lo:+.4f}, {r.d_roc_hi:+.4f}), wins {r.wins_roc}/6; "
                f"dPR {r.d_pr:+.4f} ({r.d_pr_lo:+.4f}, {r.d_pr_hi:+.4f})")
    return S


def load_dwellings():
    C = CONFIG
    p = C["dwellings_file"]
    if p and Path(p).exists():
        d = pd.read_csv(p)
        if {"ta_code", "dwellings"} <= set(d.columns):
            return pd.Series(d.dwellings.values, index=d.ta_code.astype(str).str.zfill(3), name="dwellings")
    ce_path = Path(C["construction_extension_py"])
    if ce_path.exists():
        spec = importlib.util.spec_from_file_location("ce", ce_path)
        ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
        ce.CONFIG["dwellings_file"] = p
        ce.CONFIG["out_dir"] = str(out() / "_ce_tmp")
        try:
            s = ce.load_dwellings(ce.ta_name_map())
            if s is not None:
                return s.rename("dwellings")
        except SystemExit as e:
            log(f"Census dwellings could not be read: {e}")
    return None


def rv5(X):
    log("\n" + "=" * 100 + "\nRV5  Comment 5: consents normalised by dwelling stock")
    C = CONFIG
    st = load_dwellings()
    if st is None:
        log("SKIPPED: no 2023 Census dwelling file. Download 'Census - Dwellings by type and territorial authority "
            "2013, 2018 and 2023' (Figure.NZ, CSV; source Stats NZ), save it in 01_raw/census/ and set "
            "CONFIG['dwellings_file'].")
        return None
    st.index = st.index.astype(str).str.zfill(3)
    X["dwellings"] = X.ta.map(st)
    log(f"dwelling stock matched for {X.dwellings.notna().mean():.2%} of listing-origins "
        f"({X.loc[X.dwellings.isna(), 'ta'].unique().tolist()} unmatched)")
    c3, c12 = C["construction_count_cols"]
    X["cons_norm_3m"] = np.log1p(1000 * np.expm1(X[c3]) / X.dwellings)       # consents per 1,000 dwellings
    X["cons_norm_12m"] = np.log1p(1000 * np.expm1(X[c12]) / X.dwellings)
    # area level: within-transition rank association with the disappearance rate (cells >= 30 listings)
    cell = X[X.t <= 9].groupby(["t", "ta"]).agg(n=("y", "size"), rate=("y", "mean"), c12=(c12, "first"),
                                                 n12=("cons_norm_12m", "first"), dw=("dwellings", "first")).reset_index()
    cell = cell[cell.n >= 30].dropna()
    rng = np.random.default_rng(C["seed"])

    def wr(z, x):
        r = z.groupby("t")[x].rank(pct=True); return r - r.groupby(z.t).transform("mean")

    def rho(z, x):
        return np.corrcoef(wr(z, x), wr(z, "rate"))[0, 1]

    tas = cell.ta.unique(); g = {k: v for k, v in cell.groupby("ta")}
    rows = []
    for x, lab in (("c12", "consents 12 m (count)"), ("n12", "consents 12 m per 1,000 dwellings")):
        bs = [rho(pd.concat([g[k] for k in rng.choice(tas, len(tas))]), x) for _ in range(C["bootstrap_reps"])]
        rows.append({"measure": lab, "cells": len(cell), "tas": len(tas), "rho": rho(cell, x),
                     "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5)})
    A = pd.DataFrame(rows); A.to_csv(out() / "RV5_area_association.csv", index=False)
    log("Area level, within-transition Spearman with TA disappearance rate (T1-T9):")
    log(A.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    P = refits(X, ["RV5"])
    S = summarise(P); S.to_csv(out() / "RV5_listing_level.csv", index=False)
    show(S, "Listing level: the two count predictors replaced by per-1,000-dwelling rates (spec - A5_ref)")
    return S


# ============================================================================ RV2 rule-based benchmarks
def rv2(X, dev_frozen, t10_frozen):
    log("\n" + "=" * 100 + "\nRV2  Comment 2: rule-based benchmarks (calendar availability, review recency)")
    C = CONFIG
    cal, rec = C["calendar_rule_col"], C["recency_col"]
    rows = []
    for w in DEV + [TEST]:
        tr, te = X[X.t < w], X[X.t == w]
        sc = {}
        # direction fixed on the training windows only (no information from window w)
        s_cal = -1 if met(tr.y, tr[cal].fillna(tr[cal].median()))["roc"] < 0.5 else 1
        sc["R1 calendar (open share, week 4)"] = s_cal * te[cal].fillna(tr[cal].median())
        if rec:
            fill = tr[rec].max()          # never reviewed = longest time since a review
            s_rec = -1 if met(tr.y, tr[rec].fillna(fill))["roc"] < 0.5 else 1
            sc["R2 review recency"] = s_rec * te[rec].fillna(fill)
            sc["R3 calendar + recency (mean rank)"] = (sc["R1 calendar (open share, week 4)"].rank(pct=True)
                                                       + sc["R2 review recency"].rank(pct=True)) / 2
            sc["R4 logistic, 2 predictors"] = fit_predict(tr, te, [cal, rec], kind="logit")
        for k, v in sc.items():
            m = met(te.y, v)
            r10, ev, kk = topk_recall(te.y.values, np.asarray(v, float), 0.10)
            rows.append({"benchmark": k, "t": w, "roc": m["roc"], "pr": m["pr"], "recall_top10": r10, "events_top10": ev})
    B = pd.DataFrame(rows)
    if dev_frozen is not None:
        for w, g in dev_frozen.groupby("t"):
            m = met(g.y, g.p); r10, ev, _ = topk_recall(g.y.values, g.p.values, 0.10)
            rows.append({"benchmark": "Frozen A5 XGBoost", "t": w, "roc": m["roc"], "pr": m["pr"],
                         "recall_top10": r10, "events_top10": ev})
    if t10_frozen is not None and "y" in t10_frozen:
        m = met(t10_frozen.y, t10_frozen.p); r10, ev, _ = topk_recall(t10_frozen.y.values, t10_frozen.p.values, 0.10)
        rows.append({"benchmark": "Frozen A5 XGBoost", "t": TEST, "roc": m["roc"], "pr": m["pr"],
                     "recall_top10": r10, "events_top10": ev})
    B = pd.DataFrame(rows); B.to_csv(out() / "RV2_rule_benchmarks_by_window.csv", index=False)
    S = (B.assign(part=np.where(B.t == TEST, "T10", "dev"))
          .groupby(["benchmark", "part"])[["roc", "pr", "recall_top10"]].mean().unstack("part"))
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.reset_index(); S.to_csv(out() / "RV2_rule_benchmarks.csv", index=False)
    log("dev = macro mean T4-T9; T10 = locked holdout (rules involve no fitting except R4):")
    log(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return S


# ============================================================================ RV3 stored LR / prevalence
def rv3(dev_all, X):
    log("\n" + "=" * 100 + "\nRV3  Comment 3: logistic regression and prevalence-only baseline (development)")
    rows = []
    if dev_all is not None:
        log("models x feature sets in the stored development predictions:\n"
            + dev_all.groupby(["model", "spec"]).size().to_string())
        for (mo, sp), g in dev_all.groupby(["model", "spec"]):
            per = [met(h.y, h.p) for _, h in g.groupby("t")]
            rows.append({"source": "stored", "model": mo, "spec": sp, "windows": len(per),
                         "dev_roc": np.nanmean([m["roc"] for m in per]), "dev_pr": np.nanmean([m["pr"] for m in per]),
                         "dev_brier": np.nanmean([m["brier"] for m in per])})
    # prevalence-only baseline, computed exactly: constant = training prevalence of each expanding window
    per = []
    for w in DEV:
        tr, te = X[X.t < w], X[X.t == w]
        p0 = tr.y.mean()
        per.append({"roc": 0.5, "pr": te.y.mean(), "brier": brier_score_loss(te.y, np.full(len(te), p0))})
    rows.append({"source": "computed", "model": "prevalence-only", "spec": "none", "windows": 6,
                 "dev_roc": 0.5, "dev_pr": np.mean([m["pr"] for m in per]), "dev_brier": np.mean([m["brier"] for m in per])})
    # prevalence-only on the locked T10 transition (constant = T1-T9 prevalence)
    tr, te = X[X.t < TEST], X[X.t == TEST]
    p0 = tr.y.mean()
    log(f"prevalence-only on T10: constant {p0:.5f} (T1-T9 rate), Brier {brier_score_loss(te.y, np.full(len(te), p0)):.6f}, "
        f"PR-AUC {te.y.mean():.5f} (= T10 event rate)")
    # logistic regression: refit with the same expanding windows if it is not among the stored predictions
    stored_lr = dev_all is not None and dev_all.model.str.lower().str.contains("log").any()
    orig = Path("05_results/development_modelling/primary_development_metrics_summary.csv")
    if not stored_lr and orig.exists():
        M = pd.read_csv(orig)
        if M.astype(str).apply(lambda c: c.str.contains("LOGISTIC", case=False)).any().any():
            log(f"original logistic-regression results found in {orig.as_posix()} (used; no refit):")
            log(M.to_string(index=False))
            stored_lr = True
    if not stored_lr:
        log("logistic regression is NOT among the stored development predictions -> refitted now "
            "(53 A5 predictors, same expanding windows T4-T9, training-fold medians + missing indicators, "
            "one-hot categoricals, training-fold scaling)")
        P = rolling(X, CONFIG["a5_features"], "LR_A5", kind="logit", with_t10=False)
        per = [met(h.y, h.p) for _, h in P.groupby("t")]
        for (w, h), m in zip(P.groupby("t"), per):
            log(f"  LR T{w}: ROC {m['roc']:.4f}  PR {m['pr']:.4f}  Brier {m['brier']:.5f}")
        rows.append({"source": "refitted", "model": "logistic regression", "spec": "A5 (53)", "windows": len(per),
                     "dev_roc": np.nanmean([m["roc"] for m in per]), "dev_pr": np.nanmean([m["pr"] for m in per]),
                     "dev_brier": np.nanmean([m["brier"] for m in per])})
    R = pd.DataFrame(rows); R.to_csv(out() / "RV3_all_models_dev.csv", index=False)
    log(R.sort_values("dev_pr", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    log("(prevalence-only: ROC-AUC is 0.5 by construction and PR-AUC equals the event rate of each window)")
    return R


# ============================================================================ RV6 discrepancy
def rv6(X, dev_frozen, t10_frozen):
    log("\n" + "=" * 100 + "\nRV6  Comment 6: stored (frozen) versus re-fitted development predictions")
    import sklearn, xgboost
    log(f"software now: python {platform.python_version()}, xgboost {xgboost.__version__}, "
        f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}")
    for p in Path("05_results").rglob("*") if Path("05_results").exists() else []:
        if p.suffix.lower() in (".json", ".txt", ".yaml", ".yml") and re.search(r"version|environment|requirements|spec|frozen", p.name, re.I):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            v = re.findall(r"(xgboost|scikit-learn|sklearn|pandas|numpy)[^\n]{0,30}?(\d+\.\d+(?:\.\d+)?)", txt, re.I)
            if v:
                log(f"versions recorded in {p.as_posix()}: {sorted(set(v))}")
    ref = _REFIT_CACHE.get("A5_ref")
    if ref is None:
        ref = _REFIT_CACHE["A5_ref"] = rolling(X, CONFIG["a5_features"], "A5_ref")
    rows = []
    if dev_frozen is not None:
        j = dev_frozen.merge(ref[ref.t.isin(DEV)][["t", "id", "y", "p"]], on=["t", "id"], suffixes=("_frozen", "_refit"))
        log(f"matched rows {len(j):,} of {len(dev_frozen):,} stored; labels identical: {(j.y_frozen == j.y_refit).mean():.4%}")
        for w, g in j.groupby("t"):
            a, b = met(g.y_frozen, g.p_frozen), met(g.y_frozen, g.p_refit)
            k = int(np.ceil(0.1 * len(g)))
            top_a = set(g.nlargest(k, "p_frozen").id); top_b = set(g.nlargest(k, "p_refit").id)
            rows.append({"t": w, "n": len(g), "roc_frozen": a["roc"], "roc_refit": b["roc"], "pr_frozen": a["pr"],
                         "pr_refit": b["pr"], "spearman_p": g.p_frozen.rank().corr(g.p_refit.rank()),
                         "max_abs_diff_p": (g.p_frozen - g.p_refit).abs().max(),
                         "top10_overlap": len(top_a & top_b) / k})
        # are the stored development rows the same origins as the panel?
        n_panel = X[X.t.isin(DEV)].groupby("t").size()
        n_stored = dev_frozen.groupby("t").size()
        log("rows per window, panel vs stored:\n" + pd.DataFrame({"panel": n_panel, "stored": n_stored}).to_string())
    if t10_frozen is not None:
        t = ref[ref.t == TEST].merge(t10_frozen[["id", "p"]], on="id", suffixes=("_refit", "_frozen"))
        rows.append({"t": TEST, "n": len(t), "spearman_p": t.p_frozen.rank().corr(t.p_refit.rank()),
                     "max_abs_diff_p": (t.p_frozen - t.p_refit).abs().max()})
    R = pd.DataFrame(rows); R.to_csv(out() / "RV6_frozen_vs_refit_by_window.csv", index=False)
    log(R.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    log("Interpretation guide: spearman ~1 and max diff ~0 -> identical; same ROC but lower spearman -> different "
        "random streams/encoding; different n -> different rows; T10 exact but dev not -> the stored development "
        "file was produced by an earlier run (check its timestamp and the code version that wrote it).")
    for p in [Path(CONFIG["dev_pred_path"])] + ([Path(CONFIG["t10_pred_path"])] if CONFIG["t10_pred_path"] else []):
        if p.exists():
            log(f"file time: {p.as_posix()}  modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}")
    return R


# ============================================================================ RV7 Platt in the workflow
def rv7(dev_frozen, t10_frozen):
    log("\n" + "=" * 100 + "\nRV7  Comment 7: role of the Platt-calibrated probabilities")
    if dev_frozen is None or t10_frozen is None or "y" not in t10_frozen:
        log("SKIPPED: needs stored development A5 predictions and frozen T10 predictions with labels"); return None
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lg(dev_frozen.p.values), dev_frozen.y.values)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    T = t10_frozen.copy(); T["pc"] = m.predict_proba(lg(T.p.values))[:, 1]
    rho = T.p.rank().corr(T.pc.rank())
    thr = CONFIG["frozen_threshold"]
    thr_c = float(m.predict_proba(lg(np.array([thr])))[0, 1])
    same_flags = ((T.p >= thr) == (T.pc >= thr_c)).mean()
    log(f"Platt mapping fitted on pooled development predictions: logit(p_cal) = {b:.4f} + {a:.4f} * logit(p_raw)")
    log(f"slope > 0 -> monotone: rank correlation raw vs calibrated on T10 = {rho:.6f}; "
        f"frozen threshold {thr} on the raw scale = {thr_c:.6f} on the calibrated scale; "
        f"identical flags for {same_flags:.4%} of T10 listings")
    rows = []
    for q in CONFIG["topk"]:
        k = int(np.ceil(q * len(T)))
        top = T.sort_values("p", ascending=False, kind="stable").head(k)
        rows.append({"share": q, "listings": k, "observed_events": int(top.y.sum()),
                     "expected_events_calibrated": top.pc.sum(), "expected_events_raw": top.p.sum(),
                     "mean_calibrated_risk": top.pc.mean(), "observed_rate": top.y.mean()})
    rows.append({"share": 1.0, "listings": len(T), "observed_events": int(T.y.sum()),
                 "expected_events_calibrated": T.pc.sum(), "expected_events_raw": T.p.sum(),
                 "mean_calibrated_risk": T.pc.mean(), "observed_rate": T.y.mean()})
    flagged = T[T.p >= thr]
    rows.append({"share": "at frozen threshold", "listings": len(flagged), "observed_events": int(flagged.y.sum()),
                 "expected_events_calibrated": flagged.pc.sum(), "expected_events_raw": flagged.p.sum(),
                 "mean_calibrated_risk": flagged.pc.mean(), "observed_rate": flagged.y.mean()})
    R = pd.DataFrame(rows); R.to_csv(out() / "RV7_calibrated_expected_events.csv", index=False)
    log("Expected (sum of probabilities) versus observed disappearances on T10:")
    log(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return R


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["RV1", "RV2", "RV3", "RV4", "RV5", "RV6", "RV7"])
    which = [w.upper() for w in ap.parse_known_args()[0].only]
    (out() / "RV_SUMMARY.txt").write_text("", encoding="utf-8")
    log(f"Reviewer 1 analyses  {time.strftime('%Y-%m-%d %H:%M')}  parts: {which}")
    resolve_config()
    json.dump({k: v for k, v in CONFIG.items()}, open(out() / "RV_config_used.json", "w"), indent=1, default=str)
    X = load_features()
    dev_all = load_dev_all()
    dev_frozen = frozen_dev_a5(dev_all) if dev_all is not None else None
    t10 = load_t10()
    if t10 is not None and "y" not in t10:
        t10 = t10.merge(X[X.t == TEST][["id", "y"]], on="id", how="left")
    if "RV3" in which: rv3(dev_all, X)
    if "RV7" in which: rv7(dev_frozen, t10)
    if "RV2" in which: rv2(X, dev_frozen, t10)
    if "RV1" in which: rv1(X)
    if "RV6" in which: rv6(X, dev_frozen, t10)
    if "RV4" in which: rv4(X)
    if "RV5" in which: rv5(X)
    log("\nDone. Send back: " + (out() / "RV_SUMMARY.txt").as_posix())


if __name__ == "__main__":
    main()
