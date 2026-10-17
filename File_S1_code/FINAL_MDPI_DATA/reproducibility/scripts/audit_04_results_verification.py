"""Phase 3 - independent recomputation of every reported metric from the frozen prediction artefacts,
plus provenance checks of the frozen T10 pipeline and a pre-specified horizon-sensitivity re-scoring.

NEVER retrains, re-tunes, re-thresholds or re-calibrates the locked model. The only new model evaluations
are re-scorings of the frozen T1-T9 pipeline on T10 with the two future-date-dependent seasonality predictors
replaced by origin-available values (sensitivity analysis; original outputs are preserved untouched).

Usage: python -I audit_04_results_verification.py <PROJECT_ROOT> <AUDIT_DIR>
"""
import json, sys, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix,
                             roc_curve, mean_absolute_error, mean_squared_error, r2_score)

warnings.filterwarnings("ignore")
root = Path(sys.argv[1]).resolve(); aud = Path(sys.argv[2]).resolve(); ev = aud / "evidence"; ev.mkdir(parents=True, exist_ok=True)
R = root / "05_results"; S = {}


def pm(y, p):
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                log_loss=log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); return np.log(p / (1 - p)).reshape(-1, 1)


# ============================================================ 1. development metrics (IB predictions)
D = pd.read_parquet(R / "development_modelling_IB" / "predictions" / "IB_all_development_predictions.parquet")
S["dev_pred_rows"] = len(D); S["dev_transitions"] = sorted(D.transition.unique().tolist())
S["dev_contains_T10"] = bool((D.transition == "T10").any())
S["dev_duplicate_keys"] = int(D.duplicated(["model", "feature_set", "transition", "listing_id"]).sum())
S["dev_prob_out_of_range"] = int(((D.p_disappear < 0) | (D.p_disappear > 1) | D.p_disappear.isna()).sum())
win = D.groupby(["model", "feature_set", "transition"]).apply(lambda g: pd.Series({**pm(g.y_disappear, g.p_disappear), "n": len(g)})).reset_index()
win.to_csv(ev / "dev_metrics_by_window_recomputed.csv", index=False)
mac = win.groupby(["model", "feature_set"]).agg(macro_roc_auc=("roc_auc", "mean"), macro_pr_auc=("pr_auc", "mean"), macro_brier=("brier", "mean"),
                                                min_roc=("roc_auc", "min"), max_roc=("roc_auc", "max"), min_pr=("pr_auc", "min"), max_pr=("pr_auc", "max")).reset_index()
wt = win.assign(wr=win.roc_auc * win.n, wp=win.pr_auc * win.n, wb=win.brier * win.n).groupby(["model", "feature_set"]).agg(
    wr=("wr", "sum"), wp=("wp", "sum"), wb=("wb", "sum"), n=("n", "sum")).reset_index()
mac["weighted_roc_auc"] = (wt.wr / wt.n).values; mac["weighted_pr_auc"] = (wt.wp / wt.n).values; mac["weighted_brier"] = (wt.wb / wt.n).values
frozen = pd.read_csv(R / "development_modelling_IB" / "tables" / "IB_model_summary.csv")
cmp = mac.merge(frozen[["model", "feature_set", "macro_roc_auc", "macro_pr_auc_disappearance", "macro_brier"]], on=["model", "feature_set"], suffixes=("", "_frozen"))
cmp["max_abs_diff"] = np.maximum.reduce([(cmp.macro_roc_auc - cmp.macro_roc_auc_frozen).abs(), (cmp.macro_pr_auc - cmp.macro_pr_auc_disappearance).abs(), (cmp.macro_brier - cmp.macro_brier_frozen).abs()])
cmp.sort_values("macro_pr_auc", ascending=False).to_csv(ev / "dev_model_summary_recomputed.csv", index=False)
S["dev_summary_max_abs_diff"] = float(cmp.max_abs_diff.max())
S["dev_n_configs"] = int(len(cmp))
a5 = D[(D.model == "XGB") & (D.feature_set == "A5_STRUCTURED_FULL")].copy()
S["dev_A5"] = mac[(mac.model == "XGB") & (mac.feature_set == "A5_STRUCTURED_FULL")].iloc[0].drop(["model", "feature_set"]).to_dict()
S["dev_A5_n"] = len(a5); S["dev_A5_event_rate"] = float(a5.y_disappear.mean())

# selection rule: best macro PR-AUC; ties within 1% relative ordered by ROC then Brier
best = cmp.macro_pr_auc.max(); tied = cmp[cmp.macro_pr_auc >= best * 0.99].sort_values(["macro_roc_auc", "macro_brier"], ascending=[False, True])
S["selection_rule_winner"] = f"{tied.iloc[0].model}/{tied.iloc[0].feature_set}"
S["selection_rule_candidates_within_1pct"] = tied.feature_set.tolist()

# ============================================================ 2. threshold (Youden J on pooled A5 OOF)
fpr, tpr, thr = roc_curve(a5.y_disappear, a5.p_disappear)
j = tpr - fpr; k = int(np.argmax(j)); S["youden_threshold_recomputed"] = float(thr[k]); S["youden_J"] = float(j[k])
spec = json.loads((R / "development_modelling_IB" / "freeze" / "FINAL_PRIMARY_MODEL_SPEC_FROZEN.json").read_text(encoding="utf-8"))
TH = float(spec["classification_threshold"]["threshold"]); S["threshold_frozen"] = TH
pred = (a5.p_disappear >= TH).astype(int); tn, fp, fn, tp = confusion_matrix(a5.y_disappear, pred).ravel()
S["dev_pooled_at_threshold"] = dict(sens=tp / (tp + fn), spec=tn / (tn + fp), bal_acc=(tp / (tp + fn) + tn / (tn + fp)) / 2)
bw = []
for t, g in a5.groupby("transition"):
    pr = (g.p_disappear >= TH).astype(int); tn_, fp_, fn_, tp_ = confusion_matrix(g.y_disappear, pr).ravel()
    bw.append((tp_ / (tp_ + fn_) + tn_ / (tn_ + fp_)) / 2)
S["dev_bal_acc_range"] = [float(min(bw)), float(max(bw))]

# ============================================================ 3. calibration (leave-one-window-out, development only)
cal = {"RAW": [], "PLATT": [], "ISOTONIC": []}; ll = {"RAW": [], "PLATT": [], "ISOTONIC": []}
for t in sorted(a5.transition.unique()):
    tr, te = a5[a5.transition != t], a5[a5.transition == t]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(tr.p_disappear), tr.y_disappear)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(tr.p_disappear, tr.y_disappear)
    for name, p in [("RAW", te.p_disappear.values), ("PLATT", lr.predict_proba(logit(te.p_disappear))[:, 1]), ("ISOTONIC", iso.predict(te.p_disappear))]:
        cal[name].append(brier_score_loss(te.y_disappear, p)); ll[name].append(log_loss(te.y_disappear, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
S["lowo_macro_brier"] = {k: float(np.mean(v)) for k, v in cal.items()}
S["lowo_macro_logloss"] = {k: float(np.mean(v)) for k, v in ll.items()}
S["platt_relative_improvement_pct"] = 100 * (1 - S["lowo_macro_brier"]["PLATT"] / S["lowo_macro_brier"]["RAW"])
platt_frozen = joblib.load(R / "final_locked_T10_primary" / "models" / "J_final_PLATT_development_OOF_T4_T9.joblib")
S["platt_frozen_type"] = type(platt_frozen).__name__

# ============================================================ 4. locked T10 primary
J = pd.read_parquet(R / "final_locked_T10_primary" / "predictions" / "J_final_locked_T10_primary_predictions.parquet")
S["T10_n"] = len(J); S["T10_events"] = int(J.y_disappear.sum()); S["T10_all_T10"] = bool((J.transition == "T10").all())
S["T10_raw"] = pm(J.y_disappear, J.p_disappear_raw); S["T10_platt"] = pm(J.y_disappear, J.p_disappear_calibrated)
try:
    pf = platt_frozen.predict_proba(logit(J.p_disappear_raw))[:, 1]
    S["T10_platt_reapplied_max_abs_diff"] = float(np.abs(pf - J.p_disappear_calibrated).max())
except Exception as e:
    S["T10_platt_reapplied_max_abs_diff"] = f"not computable: {e}"
pred = (J.p_disappear_raw >= TH).astype(int); tn, fp, fn, tp = confusion_matrix(J.y_disappear, pred).ravel()
S["T10_confusion"] = dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), sens=tp / (tp + fn), spec=tn / (tn + fp),
                          bal_acc=(tp / (tp + fn) + tn / (tn + fp)) / 2, precision=tp / (tp + fp))
S["T10_saved_flag_matches_threshold"] = bool((J.predicted_disappear_frozen_threshold.astype(int) == pred).all())
order = np.argsort(-J.p_disappear_raw.values, kind="stable"); ys = J.y_disappear.values[order]; N = len(J); prev = ys.mean()
topk = []
for q in [0.01, 0.05, 0.10, 0.20, 0.30]:
    kk = int(np.ceil(q * N)); ev_ = int(ys[:kk].sum())
    topk.append(dict(share=q, k=kk, events=ev_, precision_pct=100 * ev_ / kk, recall_pct=100 * ev_ / ys.sum(), lift=(ev_ / kk) / prev))
pd.DataFrame(topk).to_csv(ev / "T10_topk_recomputed.csv", index=False); S["T10_topk"] = topk
S["T10_prob_ties"] = int(J.p_disappear_raw.duplicated().sum())
J["dec"] = pd.qcut(J.p_disappear_raw.rank(method="first"), 10, labels=False) + 1
dec = J.groupby("dec").agg(n=("y_disappear", "size"), events=("y_disappear", "sum")).reset_index(); dec.to_csv(ev / "T10_deciles_recomputed.csv", index=False)
S["T10_decile_low_high"] = [int(dec.events.iloc[0]), int(dec.events.iloc[-1])]

# ============================================================ 5. frozen pipeline provenance: re-score T10, check imputer fit data
pipe = joblib.load(R / "final_locked_T10_primary" / "models" / "J_final_XGB_A5_T1_T9_pipeline.joblib")
FEAT = spec["final_primary_model"]["predictors"]
P = pd.read_parquet(root / "02_intermediate" / "modelling" / "modelling_panel_T1_T10.parquet", columns=["transition", "listing_id", "anchor_date", "ta_code"] + [c for c in FEAT if c != "ta_code"])
P["listing_id"] = P.listing_id.astype("int64")
T10 = P[P.transition == "T10"].set_index("listing_id").loc[J.listing_id.astype("int64")].reset_index()
p_re = pipe.predict_proba(T10[FEAT])[:, 1]
S["T10_rescore_max_abs_diff_vs_saved"] = float(np.abs(p_re - J.p_disappear_raw.values).max())
pre = pipe.named_steps["preprocess"]; S["pipeline_steps"] = list(pipe.named_steps)
# compare numeric imputer medians with medians over T1-T9 and over T1-T10
try:
    num_tr = [tr for tr in pre.transformers_ if tr[0] != "remainder"]
    info = []
    for name, trans, cols_ in num_tr:
        imp = trans.named_steps.get("imputer") if hasattr(trans, "named_steps") else None
        if imp is not None and getattr(imp, "strategy", "") == "median":
            dev = P[P.transition != "T10"][cols_].median().values; allm = P[cols_].median().values
            st = imp.statistics_.astype(float)
            info.append(dict(block=name, n_cols=len(cols_), max_diff_vs_T1_T9=float(np.nanmax(np.abs(st - dev))), max_diff_vs_T1_T10=float(np.nanmax(np.abs(st - allm)))))
    S["imputer_fit_check"] = info
except Exception as e:
    S["imputer_fit_check"] = f"not computable: {e}"
xgb = pipe.named_steps["model"]; S["xgb_params_saved"] = {k: v for k, v in xgb.get_params().items() if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "tree_method", "random_state", "scale_pos_weight"]}
S["xgb_params_spec"] = spec.get("xgboost_hyperparameters")

# ============================================================ 6. horizon sensitivity (frozen model, no refit)
SP = pd.read_parquet(root / "02_intermediate" / "seasonality" / "adp_historical_seasonality_ta_month_frozen.parquet")
anchor = pd.Timestamp("2026-07-12")


def season_index(ta, h):
    days = pd.date_range(anchor, anchor + pd.Timedelta(days=h - 1)); w = pd.Series(days.month).value_counts(normalize=True)
    prof = SP[SP.ta_code == ta].set_index("calendar_month")["historical_seasonal_index"]
    return float(sum(wt_ * prof.get(m, np.nan) for m, wt_ in w.items()))


sens = []
order_raw = None
for h in [23, 28, 30, 32, 35, 37]:
    X = T10[FEAT].copy(); X["forecast_horizon_days"] = h
    X["historical_seasonality_index_horizon"] = T10.ta_code.map({ta: season_index(ta, h) for ta in T10.ta_code.unique()}).values
    p = pipe.predict_proba(X)[:, 1]
    o = np.argsort(-p, kind="stable"); kk = int(np.ceil(0.1 * N))
    sens.append(dict(assumed_horizon_days=h, actual=h == 32, roc_auc=roc_auc_score(J.y_disappear, p), pr_auc=average_precision_score(J.y_disappear, p),
                     brier_raw=brier_score_loss(J.y_disappear, p), top10_events=int(J.y_disappear.values[o[:kk]].sum()),
                     top10_recall_pct=100 * J.y_disappear.values[o[:kk]].sum() / J.y_disappear.sum(),
                     spearman_vs_frozen_scores=float(pd.Series(p).corr(pd.Series(J.p_disappear_raw.values), method="spearman")),
                     mean_predicted=float(p.mean())))
pd.DataFrame(sens).to_csv(ev / "T10_horizon_sensitivity_frozen_model.csv", index=False); S["horizon_sensitivity"] = sens

# ============================================================ 7. SHAP recomputation on the stored deterministic sample
from xgboost import DMatrix
samp = J[["listing_id"]].sample(n=3000, random_state=2026).reset_index(drop=True)
Xs = T10.set_index("listing_id").loc[samp.listing_id.astype("int64"), FEAT].reset_index(drop=True)
Xt = pre.transform(Xs); contrib = xgb.get_booster().predict(DMatrix(Xt), pred_contribs=True)
names = pre.get_feature_names_out()
imp = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=names)
# aggregate transformed columns back to source predictor
agg = {}
for n_, v in imp.items():
    base = n_.split("__", 1)[-1]
    src = next((f for f in sorted(FEAT, key=len, reverse=True) if base == f or base.startswith(f + "_") or base == "missingindicator_" + f), base)
    agg[src] = agg.get(src, 0) + v
shap = pd.Series(agg).sort_values(ascending=False)
shap.rename("mean_abs_shap").to_csv(ev / "T10_SHAP_recomputed.csv")
fro = pd.read_csv(R / "manuscript_final_L_to_O" / "tables" / "M_SHAP_53_frozen_features.csv")
S["shap_top10_recomputed"] = shap.head(10).round(5).to_dict(); S["shap_frozen_head"] = fro.head(10).to_dict("records")

# ============================================================ 8. bootstrap CIs (exact replication, seed 2026)
geo = P[P.transition == "T10"][["listing_id", "ta_code"]]
pb = J.merge(geo, on="listing_id", how="left")
K = pd.read_parquet(R / "secondary_price_resilience_K" / "K_final_locked_T10_secondary_predictions.parquet"); K["listing_id"] = K.listing_id.astype("int64")
kb = K.merge(geo, on="listing_id", how="left")
rng = np.random.default_rng(2026)


def boot(df, yc, pc, fn):
    cl = df.groupby("ta_code", dropna=False, sort=False).indices; groups = [np.asarray(v) for v in cl.values()]
    y = df[yc].to_numpy(float); p = df[pc].to_numpy(float); reps = []
    for b in range(600):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]); reps.append(fn(y[idx], p[idx]))
    reps = pd.DataFrame(reps); return {c: (float(reps[c].quantile(.025)), float(reps[c].quantile(.975))) for c in reps}


fp_ = lambda y, p: dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p))
fs_ = lambda y, p: dict(mae=mean_absolute_error(y, p), rmse=np.sqrt(mean_squared_error(y, p)), r2=r2_score(y, p))
S["boot_raw"] = boot(pb, "y_disappear", "p_disappear_raw", fp_); S["boot_platt"] = boot(pb, "y_disappear", "p_disappear_calibrated", fp_)
S["boot_secondary"] = boot(kb, "y_secondary", "predicted_secondary", fs_)
S["boot_n_ta_primary"] = int(pb.ta_code.nunique()); S["boot_n_ta_secondary"] = int(kb.ta_code.nunique())

# ============================================================ 9. secondary outcome
S["sec_T10"] = dict(n=len(K), mae=mean_absolute_error(K.y_secondary, K.predicted_secondary), rmse=float(np.sqrt(mean_squared_error(K.y_secondary, K.predicted_secondary))),
                    r2=r2_score(K.y_secondary, K.predicted_secondary), all_zero=bool(np.allclose(K.predicted_secondary, 0)),
                    y_min=float(K.y_secondary.min()), y_max=float(K.y_secondary.max()))
KD = pd.read_parquet(R / "secondary_price_resilience_K" / "K_development_rolling_predictions.parquet")
S["sec_dev_transitions"] = sorted(KD.transition.unique().tolist())
kd = KD.groupby(["model", "transition"]).apply(lambda g: pd.Series(dict(mae=mean_absolute_error(g.y_secondary, g.prediction), rmse=np.sqrt(mean_squared_error(g.y_secondary, g.prediction))))).reset_index()
S["sec_dev_macro"] = kd.groupby("model")[["mae", "rmse"]].mean().to_dict("index")

json.dump(S, open(ev / "phase3_summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(S, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
