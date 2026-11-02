#!/usr/bin/env python
"""
Reviewer 2 analyses for manuscript data-4668825 (post hoc; the frozen model is NOT changed).
Reuses the loaders, refits and cache of reviewer_R1_analyses.py (keep both files in the same folder).

  Q1/S1  confirmed-disappearance label: prevalence, ROC-AUC, PR-AUC, top-10% recall; model trained on it
  Q3     information age within an origin (cutoff minus each listing's own last_scraped)
  Q4     rental family under release-timing uncertainty (refit without it; verified windows only)
  Q5     alternative thresholds: max-F1, fixed review capacity, cost-based on calibrated probabilities
  Q6     (optional, --tune) small hyperparameter grid for A5 and A6
  Q7/S2  transition-level development performance and temporal variability
  Q8     between-TA versus within-TA construction effects (Mundlak logistic, TA-clustered SE)
  Q9     model selection under alternative averaging criteria
  Q10    secondary price-rank outcome: noise, mean reversion and horizon
  S3     screening workload: false positives per review capacity

Run from the project root:
    python 10_revision/reviewer_R2_analyses.py                 # all except tuning
    python 10_revision/reviewer_R2_analyses.py --only Q5 Q7 Q9 S3   # fast parts (no refitting)
    python 10_revision/reviewer_R2_analyses.py --only Q6 --tune     # hyperparameter grid (long)
Outputs: 10_revision_out/RV_reviewer2/  ->  send back RV2_SUMMARY.txt
"""
import argparse, importlib.util, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r1", HERE / "reviewer_R1_analyses.py")
r1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1)
C = r1.CONFIG
C["out_dir"] = "10_revision_out/RV_reviewer2"
C.update({
    "airbnb_dir": "01_raw/airbnb",
    "secondary_y_col": "price_resilience_delta",   # within-peer price-rank change (secondary outcome)
    "semantic_cols_regex": r"^(sem|emb|semantic)_?\d+",   # A6 review-embedding columns (Q6 only)
    "semantic_path": None,          # parquet with id, transition and the 384 embedding columns, if separate
})
DEV, TEST = r1.DEV, r1.TEST
met, topk, log, out = r1.met, r1.topk_recall, None, r1.out


def log(m=""):
    print(m, flush=True)
    with open(out() / "RV2_SUMMARY.txt", "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")


r1.log = log
fmt = lambda v: f"{v:.4f}"


def snapshots():
    d = Path(C["airbnb_dir"])
    return sorted(p for p in d.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))


def snapshot_listing_info():
    """id and last_scraped of every snapshot (raw files are read, never copied)."""
    info = []
    for k, s in enumerate(snapshots(), start=1):
        f = sorted(s.glob("listings.csv*"))[0]
        d = pd.read_csv(f, usecols=["id", "last_scraped"], dtype=str)
        d["id"] = d.id.str.replace(r"\.0$", "", regex=True)
        d["last_scraped"] = pd.to_datetime(d.last_scraped, errors="coerce")
        d["snap"] = k
        info.append(d)
    log(f"read id/last_scraped from {len(info)} snapshots")
    return info


# ============================================================================ Q1 / S1
def q1(X, dev_frozen, info):
    log("\n" + "=" * 100 + "\nQ1/S1  Original versus confirmed disappearance (absent at t+1 AND t+2)")
    ids = [set(d.id) for d in info]
    X = X.copy()
    X["present_t2"] = [i in ids[t + 1] if t + 1 < len(ids) else np.nan for i, t in zip(X.id, X.t.astype(int))]
    X["y_conf"] = np.where(X.t <= 9, ((X.y == 1) & (X.present_t2 == False)).astype(float), np.nan)
    rows = []
    d = dev_frozen.merge(X[["t", "id", "y_conf"]], on=["t", "id"], how="left")
    for w, g in d.groupby("t"):
        a, b = met(g.y, g.p), met(g.y_conf, g.p)
        rows.append({"t": w, "n": len(g), "prev_orig": g.y.mean(), "prev_conf": g.y_conf.mean(),
                     "roc_orig": a["roc"], "roc_conf": b["roc"], "pr_orig": a["pr"], "pr_conf": b["pr"],
                     "rec10_orig": topk(g.y.values, g.p.values, .1)[0],
                     "rec10_conf": topk(g.y_conf.values, g.p.values, .1)[0]})
    R = pd.DataFrame(rows)
    R.loc[len(R)] = {"t": "macro", **R.drop(columns="t").mean().to_dict()}
    log("Frozen A5 predictions re-scored against each label (development):")
    log(R.to_string(index=False, float_format=fmt))
    # model TRAINED on the confirmed label. The confirmed label of origin w-1 needs snapshot w+1, which is
    # collected after the origin-w cutoff, so training must stop at origin w-2 (one extra snapshot of delay).
    A5 = C["a5_features"]; rows = []
    Xc = X[X.t <= 9].copy(); Xc["y"] = Xc["y_conf"]
    for w in DEV:
        tr, te = Xc[Xc.t <= w - 2], Xc[Xc.t == w]
        p = r1.fit_predict(tr, te, A5)
        m = met(te.y, p)
        rows.append({"t": w, "roc_conf_trained": m["roc"], "pr_conf_trained": m["pr"],
                     "rec10_conf_trained": topk(te.y.values, p, .1)[0]})
    T = pd.DataFrame(rows)
    ref = r1.rolling(X, A5, "A5_ref")
    ref = ref[ref.t.isin(DEV)].merge(X[["t", "id", "y_conf"]], on=["t", "id"])
    T["roc_orig_trained_on_conf"] = [met(g.y_conf, g.p)["roc"] for _, g in ref.groupby("t")]
    T["pr_orig_trained_on_conf"] = [met(g.y_conf, g.p)["pr"] for _, g in ref.groupby("t")]
    T.loc[len(T)] = {"t": "macro", **T.drop(columns="t").mean().to_dict()}
    log("Same code: model trained on the confirmed label (origins <= w-2) versus model trained on the original "
        "label (origins < w), both scored against the confirmed label:")
    log(T.to_string(index=False, float_format=fmt))
    R.to_csv(out() / "Q1_confirmed_rescored.csv", index=False); T.to_csv(out() / "Q1_confirmed_trained.csv", index=False)


# ============================================================================ Q3
def q3(X, dev_frozen, t10, info):
    log("\n" + "=" * 100 + "\nQ3  Information age within an origin")
    ls = pd.concat(info)
    cut = ls.groupby("snap").last_scraped.max().rename("cutoff")
    ls = ls.join(cut, on="snap")
    ls["info_age_days"] = (ls.cutoff - ls.last_scraped).dt.days
    ls = ls.rename(columns={"snap": "t"})[["t", "id", "info_age_days"]].drop_duplicates(["t", "id"])
    Xa = X.merge(ls, on=["t", "id"], how="left")
    log(f"info age matched for {Xa.info_age_days.notna().mean():.3%} of origins")
    log("information age (days between a listing's own scrape and the origin cutoff), by transition:\n"
        + Xa.groupby("t").info_age_days.describe(percentiles=[.5, .9])[["mean", "50%", "90%", "max"]]
        .to_string(float_format=lambda v: f"{v:.1f}"))
    bins, labs = [-1, 1, 4, 8, 99], ["0-1 d", "2-4 d", "5-8 d", ">8 d"]
    rows = []
    for part, P in (("dev (frozen)", dev_frozen), ("T10 (frozen)", t10)):
        if P is None:
            continue
        g = P.merge(Xa[["t", "id", "info_age_days"]] if "t" in P else Xa[Xa.t == TEST][["id", "info_age_days"]],
                    on=[c for c in ("t", "id") if c in P], how="left")
        g["age"] = pd.cut(g.info_age_days, bins, labels=labs)
        for a, h in g.groupby("age", observed=True):
            m = met(h.y, h.p)
            rows.append({"part": part, "info_age": a, "n": len(h), "event_rate": h.y.mean(), "roc": m["roc"],
                         "pr": m["pr"], "mean_p": h.p.mean()})
    R = pd.DataFrame(rows); R.to_csv(out() / "Q3_info_age_groups.csv", index=False)
    log(R.to_string(index=False, float_format=fmt))
    A5 = C["a5_features"]
    P = pd.concat([r1.rolling(X, A5, "A5_ref"), r1.rolling(Xa, A5 + ["info_age_days"], "A5_plus_info_age")])
    S = r1.summarise(P); S.to_csv(out() / "Q3_info_age_refit.csv", index=False)
    r1.show(S, "Refit with each listing's information age added (difference = spec - A5_ref)")


# ============================================================================ Q4
def q4(X):
    log("\n" + "=" * 100 + "\nQ4  Rental family under release-timing uncertainty")
    A5 = C["a5_features"]
    P = pd.concat([r1.rolling(X, A5, "A5_ref"),
                   r1.rolling(X, [c for c in A5 if c not in C["rental_cols"]], "A5_no_rental")])
    S = r1.summarise(P); S.to_csv(out() / "Q4_no_rental.csv", index=False)
    r1.show(S, "All development windows (difference = spec - A5_ref)")
    rows = []
    for w in [5, 8, 10]:          # transitions whose rental-bond release was verified before the cutoff
        for s, g in P[P.t == w].groupby("spec"):
            m = met(g.y, g.p); rows.append({"t": w, "spec": s, "roc": m["roc"], "pr": m["pr"]})
    V = pd.DataFrame(rows).pivot(index="t", columns="spec", values=["roc", "pr"])
    log("Verified transitions only (T5, T8 development; T10 holdout):\n" + V.to_string(float_format=fmt))
    V.to_csv(out() / "Q4_verified_windows.csv")


# ============================================================================ Q5
def q5(dev_frozen, t10):
    log("\n" + "=" * 100 + "\nQ5  Alternative thresholds (chosen on pooled development predictions, applied to T10)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    y, p = dev_frozen.y.values, dev_frozen.p.values
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    pl = LogisticRegression(C=1e6, max_iter=2000).fit(lg(p), y)
    cal = lambda q: pl.predict_proba(lg(np.asarray(q)))[:, 1]
    rules = {"Youden J (frozen)": C["frozen_threshold"], "max F1": float(th[np.nanargmax(f1)])}
    for q in (0.05, 0.10, 0.20):
        rules[f"capacity {int(q * 100)}% (dev quantile)"] = float(np.quantile(p, 1 - q))
    # cost-based: flag when calibrated risk exceeds c_FP / (c_FP + c_FN)
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 4001)))
    pc_grid = cal(grid)
    for ratio in (10, 20, 40):
        target = 1 / (1 + ratio)
        rules[f"cost FN:FP = {ratio}:1 (calibrated >= {target:.4f})"] = float(grid[np.argmax(pc_grid >= target)])
    rows = []
    yt, pt = t10.y.values, t10.p.values
    for k, t in rules.items():
        f = pt >= t
        tp, fp = int((f & (yt == 1)).sum()), int((f & (yt == 0)).sum())
        rows.append({"rule": k, "raw_threshold": t, "flagged": int(f.sum()), "share_flagged": f.mean(), "TP": tp,
                     "FP": fp, "precision": tp / max(f.sum(), 1), "recall": tp / yt.sum(), "FP_per_TP": fp / max(tp, 1)})
    R = pd.DataFrame(rows); R.to_csv(out() / "Q5_thresholds.csv", index=False)
    log(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ============================================================================ Q6
def q6(X):
    log("\n" + "=" * 100 + "\nQ6  Small hyperparameter grid for A5 and A6 (selection by macro dev PR-AUC)")
    src = C["semantic_path"] or C["features_path"]
    sem = [c for c in r1.cols_of(src) if re.match(C["semantic_cols_regex"], c)]
    if sem:
        S = r1.read_any(src, columns=[C["id_col"], C["t_col"]] + sem)
        S = S.rename(columns={C["id_col"]: "id", C["t_col"]: "t"}); S["t"] = r1.tnum(S["t"]); S["id"] = r1.norm_id(S["id"])
        sem = [c for c in S.columns if re.match(C["semantic_cols_regex"], c)]
        X = X.merge(S[["t", "id"] + sem], on=["t", "id"], how="left")
    log(f"semantic columns found: {len(sem)}")
    grid = [dict(max_depth=6, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0),   # frozen
            dict(max_depth=3, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0),
            dict(max_depth=9, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0),
            dict(max_depth=6, colsample_bytree=0.3, min_child_weight=5, reg_lambda=1.0),
            dict(max_depth=6, colsample_bytree=0.1, min_child_weight=5, reg_lambda=1.0),
            dict(max_depth=6, colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0),
            dict(max_depth=6, colsample_bytree=0.8, min_child_weight=5, reg_lambda=10.0),
            dict(max_depth=4, colsample_bytree=0.3, min_child_weight=20, reg_lambda=10.0)]
    specs = {"A5": C["a5_features"]}
    if len(sem) >= 100:
        specs["A6"] = C["a5_features"] + sem
    else:
        log("A6 skipped: embedding columns not found (set CONFIG semantic_path / semantic_cols_regex)")
    base = dict(C["xgb"]); rows = []
    for name, cols in specs.items():
        for i, g in enumerate(grid):
            C["xgb"] = {**base, **g}
            P = r1.rolling(X, cols, f"{name}_grid{i}", with_t10=False)
            per = [met(h.y, h.p) for _, h in P.groupby("t")]
            rows.append({"spec": name, "config": i, **g, "dev_roc": np.mean([m["roc"] for m in per]),
                         "dev_pr": np.mean([m["pr"] for m in per])})
    C["xgb"] = base
    R = pd.DataFrame(rows); R.to_csv(out() / "Q6_tuning_grid.csv", index=False)
    log(R.to_string(index=False, float_format=fmt))
    best = R.loc[R.groupby("spec").dev_pr.idxmax()]
    log("best configuration per specification (selection is optimistic for both, equally):\n"
        + best.to_string(index=False, float_format=fmt))


# ============================================================================ Q7 / S2
def q7(dev_frozen, t10):
    log("\n" + "=" * 100 + "\nQ7/S2  Transition-level performance of the frozen A5 model")
    rows = []
    for w, g in dev_frozen.groupby("t"):
        m = met(g.y, g.p)
        rows.append({"t": f"T{w}", "n": len(g), "events": int(g.y.sum()), "prevalence": g.y.mean(), "roc": m["roc"],
                     "pr": m["pr"], "pr_lift": m["pr"] / g.y.mean(), "brier": m["brier"],
                     "recall_top10": topk(g.y.values, g.p.values, .1)[0]})
    if t10 is not None:
        m = met(t10.y, t10.p)
        rows.append({"t": "T10 (locked)", "n": len(t10), "events": int(t10.y.sum()), "prevalence": t10.y.mean(),
                     "roc": m["roc"], "pr": m["pr"], "pr_lift": m["pr"] / t10.y.mean(), "brier": m["brier"],
                     "recall_top10": topk(t10.y.values, t10.p.values, .1)[0]})
    R = pd.DataFrame(rows); R.to_csv(out() / "S2_transition_level.csv", index=False)
    log(R.to_string(index=False, float_format=fmt))
    d = R[R.t.str.match(r"T\d$")]
    log("temporal variability across the six development windows (SD; min-max):")
    for c in ("roc", "pr", "pr_lift", "recall_top10"):
        log(f"  {c:13s} SD {d[c].std(ddof=1):.4f}; range {d[c].min():.4f}-{d[c].max():.4f}")


# ============================================================================ Q8
def q8(X):
    log("\n" + "=" * 100 + "\nQ8  Between-TA and within-TA construction effects (Mundlak logistic, TA-clustered SE)")
    import statsmodels.api as sm
    D = X[(X.t <= 9) & (X.ta != "999")].copy()
    c12 = C["construction_count_cols"][1]; rent = C["rental_cols"][0]
    n0 = len(D)
    D = D.dropna(subset=[c12, rent]).copy()
    log(f"rows with both 12-month consents and rent available: {len(D):,} of {n0:,} "
        f"({n0 - len(D):,} dropped; TAs kept {D.ta.nunique()})")
    cal, rec = C["calendar_rule_col"], C["recency_col"]
    terms = {}
    for c, lab in ((c12, "consents12"), (rent, "rent")):
        z = (D[c] - D[c].mean()) / D[c].std()
        D[lab + "_between"] = z.groupby(D.ta).transform("mean")
        D[lab + "_within"] = z - D[lab + "_between"]
        terms[lab] = [lab + "_between", lab + "_within"]
    ctrl = [cal] + ([rec] if rec else [])
    for c in ctrl:
        D[c] = pd.to_numeric(D[c], errors="coerce"); D[c] = (D[c].fillna(D[c].median()) - D[c].mean()) / D[c].std()
    Z = pd.concat([D[sum(terms.values(), []) + ctrl], pd.get_dummies(D.t.astype(str), prefix="T", drop_first=True,
                                                                        dtype=float)], axis=1)
    Z = sm.add_constant(Z).astype(float)
    ok = np.isfinite(Z.values).all(axis=1)
    if (~ok).any():
        log(f"  {int((~ok).sum()):,} rows with non-finite values removed")
        Z, D = Z[ok], D[ok]
    rows = []
    for label, cols in (("context only + transition FE", [c for c in Z if c not in ctrl]), ("+ listing controls", list(Z))):
        m = sm.Logit(D.y.values, Z[cols]).fit(disp=0, cov_type="cluster", cov_kwds={"groups": D.ta.astype("category").cat.codes})
        for v in sum(terms.values(), []):
            rows.append({"model": label, "term": v, "OR_per_SD": np.exp(m.params[v]),
                         "lo": np.exp(m.conf_int().loc[v, 0]), "hi": np.exp(m.conf_int().loc[v, 1]), "p": m.pvalues[v]})
    R = pd.DataFrame(rows); R.to_csv(out() / "Q8_mundlak.csv", index=False)
    for lab in terms:
        w = D[lab + "_within"]
        log(f"  {lab}: SD of within-TA deviation {w.std():.4f} (whole variable standardised to SD 1)")
    if ((R.hi / R.lo.clip(lower=1e-300)) > 100).any():
        log("  WARNING: some intervals are extremely wide -> within-TA variation is too small (or collinear with the "
            "transition effects) to be estimated; report the between-TA terms only and say so.")
    log(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sd = D.groupby("ta")[c12].transform("mean")
    log(f"share of {c12} variance between TAs (listing level, T1-T9): "
        f"{sd.var() / D[c12].var():.4f}")


# ============================================================================ Q9
def q9(dev_all):
    log("\n" + "=" * 100 + "\nQ9  Selection of the specification under alternative criteria (stored development predictions)")
    rows, pers = [], {}
    for (mo, sp), g in dev_all.groupby(["model", "spec"]):
        per = pd.DataFrame([{"t": w, "n": len(h), **met(h.y, h.p)} for w, h in g.groupby("t")])
        if len(per) < 6 or per.pr.isna().all():
            continue
        rows.append({"candidate": f"{mo} {sp}", "macro_pr": per.pr.mean(),
                     "weighted_pr": np.average(per.pr, weights=per.n), "worst_pr": per.pr.min(),
                     "mean_minus_sd_pr": per.pr.mean() - per.pr.std(ddof=1), "macro_roc": per.roc.mean(),
                     "worst_roc": per.roc.min(), "mean_rank_pr": np.nan})
        pers[f"{mo} {sp}"] = per.set_index("t").pr
    R = pd.DataFrame(rows)
    ranks = pd.concat(pers, axis=1).rank(axis=1, ascending=False)
    R["mean_rank_pr"] = R.candidate.map(ranks.mean())
    crit = ["macro_pr", "weighted_pr", "worst_pr", "mean_minus_sd_pr", "macro_roc", "worst_roc", "mean_rank_pr"]
    sel = {c: (R.nsmallest(1, c) if c == "mean_rank_pr" else R.nlargest(1, c)).candidate.iloc[0] for c in crit}
    R.sort_values("macro_pr", ascending=False).to_csv(out() / "Q9_selection_criteria.csv", index=False)
    log(R.sort_values("macro_pr", ascending=False).to_string(index=False, float_format=fmt))
    log("candidate selected by each criterion:")
    for c, s in sel.items():
        log(f"  {c:18s} -> {s}")


# ============================================================================ Q10
def q10(X):
    log("\n" + "=" * 100 + "\nQ10  Secondary price-rank outcome: noise, mean reversion, horizon")
    ycol = C["secondary_y_col"] or next((c for c in r1.cols_of(C["features_path"])
                                         if re.search(r"^y_price$|price_resilience_delta|price.*(change|delta)", c, re.I)), None)
    if ycol is None:
        log("SKIPPED: set CONFIG['secondary_y_col'] to the price-rank change column of the feature table"); return
    S = r1.read_any(C["features_path"], columns=[C["id_col"], C["t_col"], ycol])
    S = S.rename(columns={C["id_col"]: "id", C["t_col"]: "t", ycol: "d"}); S["t"] = r1.tnum(S["t"]); S["id"] = r1.norm_id(S["id"])
    S = S.dropna(subset=["d"])
    a = S.d.abs()
    log(f"column '{ycol}': {len(S):,} labels; |change| < 0.01: {(a < .01).mean():.2%}; < 0.05: {(a < .05).mean():.2%}; "
        f"SD {S.d.std():.4f}; quantiles 5/50/95%: {S.d.quantile([.05, .5, .95]).round(4).tolist()}")
    W = S.pivot_table(index="id", columns="t", values="d")
    rows = []
    for t in sorted(W.columns):
        if t + 1 in W.columns:
            z = W[[t, t + 1]].dropna()
            if len(z) > 100 and z[t].var() > 0:
                two = z[t] + z[t + 1]
                rows.append({"pair": f"T{t}->T{t + 1}", "n": len(z), "corr_consecutive_changes": z[t].corr(z[t + 1]),
                             "var_1step": z[t].var(), "var_2step": two.var(), "ratio_2step_to_1step": two.var() / z[t].var()})
    R = pd.DataFrame(rows); R.to_csv(out() / "Q10_secondary_diagnostics.csv", index=False)
    log("Consecutive one-snapshot changes of the same listing (negative correlation = reversal/noise; "
        "variance ratio ~2 = persistent random-walk-like moves, ~1 = transient noise):")
    log(R.to_string(index=False, float_format=fmt))


# ============================================================================ S3
def s3(t10):
    log("\n" + "=" * 100 + "\nS3  Screening workload on T10 (frozen ranking)")
    rows = []
    for q in C["topk"]:
        rec, tp, k = topk(t10.y.values, t10.p.values, q)
        rows.append({"capacity": f"{int(q * 100)}%", "listings_checked": k, "true_positives": int(tp),
                     "false_positives": int(k - tp), "FP_per_TP": (k - tp) / max(tp, 1), "recall": rec,
                     "precision": tp / k})
    R = pd.DataFrame(rows); R.to_csv(out() / "S3_workload.csv", index=False)
    log(R.to_string(index=False, float_format=fmt))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["Q1", "Q3", "Q4", "Q5", "Q7", "Q8", "Q9", "Q10", "S3"])
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_known_args()[0]
    which = [w.upper() for w in a.only] + (["Q6"] if a.tune else [])
    (out() / "RV2_SUMMARY.txt").write_text("", encoding="utf-8")
    log(f"Reviewer 2 analyses  {time.strftime('%Y-%m-%d %H:%M')}  parts: {which}")
    r1.resolve_config()
    X = r1.load_features()
    dev_all = r1.load_dev_all()
    dev_frozen = r1.frozen_dev_a5(dev_all) if dev_all is not None else None
    t10 = r1.load_t10()
    if t10 is not None and "y" not in t10:
        t10 = t10.merge(X[X.t == TEST][["id", "y"]], on="id", how="left")
    info = snapshot_listing_info() if {"Q1", "Q3"} & set(which) else None
    if "S3" in which: s3(t10)
    if "Q7" in which: q7(dev_frozen, t10)
    if "Q9" in which: q9(dev_all)
    if "Q5" in which: q5(dev_frozen, t10)
    if "Q10" in which: q10(X)
    if "Q1" in which: q1(X, dev_frozen, info)
    if "Q3" in which: q3(X, dev_frozen, t10, info)
    if "Q4" in which: q4(X)
    if "Q8" in which: q8(X)
    if "Q6" in which: q6(X)
    log("\nDone. Send back: " + (out() / "RV2_SUMMARY.txt").as_posix())


if __name__ == "__main__":
    main()
