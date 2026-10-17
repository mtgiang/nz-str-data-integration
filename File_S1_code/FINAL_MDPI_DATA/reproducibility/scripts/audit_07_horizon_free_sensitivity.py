"""Post-freeze SENSITIVITY analysis (does not replace any frozen result).

(a) Retraining reproducibility: refit the frozen A5 specification on T1-T9 and compare T10 scores with the frozen file.
(b) Development-only rolling evaluation T4-T9 for A5 (53) and A5 without the two predictors that depend on the
    future snapshot date (forecast_horizon_days, historical_seasonality_index_horizon) -> 51 predictors.
(c) One post-hoc T10 evaluation of the 51-predictor variant, reported only as a sensitivity check.
Hyper-parameters, preprocessing and threshold are the frozen ones; nothing is tuned.

Usage: python -I audit_07_horizon_free_sensitivity.py <PROJECT_ROOT> <AUDIT_DIR>
"""
import json, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

root = Path(sys.argv[1]).resolve(); ev = Path(sys.argv[2]).resolve() / "evidence"
R = root / "05_results"
spec = json.loads((R / "development_modelling_IB" / "freeze" / "FINAL_PRIMARY_MODEL_SPEC_FROZEN.json").read_text(encoding="utf-8"))
FEAT = spec["final_primary_model"]["predictors"]; TH = float(spec["classification_threshold"]["threshold"])
HOR = ["forecast_horizon_days", "historical_seasonality_index_horizon"]
frozen = joblib.load(R / "final_locked_T10_primary" / "models" / "J_final_XGB_A5_T1_T9_pipeline.joblib")
P = pd.read_parquet(root / "02_intermediate" / "modelling" / "modelling_panel_T1_T10.parquet", columns=["transition", "listing_id", "next_period_persistence"] + FEAT)
P["y"] = 1 - P.next_period_persistence.astype(int)
order = [f"T{i}" for i in range(1, 11)]


def make_pipe(drop):
    p = clone(frozen)
    ct = p.named_steps["preprocess"]
    ct.transformers = [(n, t, [c for c in cols if c not in drop]) for n, t, cols in ct.transformers]
    return p


def metrics(y, p):
    o = np.argsort(-p, kind="stable"); k = int(np.ceil(0.1 * len(p)))
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                top10_recall=y[o[:k]].sum() / y.sum(), top10_precision=y[o[:k]].mean())


rows = []
# (a) reproducibility of the frozen final fit
t0 = time.time()
tr = P[P.transition.isin(order[:9])]; te = P[P.transition == "T10"]
pa = make_pipe([]).fit(tr[FEAT], tr.y).predict_proba(te[FEAT])[:, 1]
J = pd.read_parquet(R / "final_locked_T10_primary" / "predictions" / "J_final_locked_T10_primary_predictions.parquet")
fz = J.set_index("listing_id").loc[te.listing_id, "p_disappear_raw"].values
rep = dict(max_abs_diff_vs_frozen=float(np.abs(pa - fz).max()), spearman=float(pd.Series(pa).corr(pd.Series(fz), method="spearman")),
           **{f"refit_{k}": v for k, v in metrics(te.y.values, pa).items()}, seconds=round(time.time() - t0, 1))
print("reproducibility", rep, flush=True)

# (b) development rolling windows
for name, drop in [("A5_frozen_53", []), ("A5_minus_horizon_51", HOR)]:
    feats = [f for f in FEAT if f not in drop]
    for k in range(3, 9):
        tr = P[P.transition.isin(order[:k])]; te = P[P.transition == order[k]]
        p = make_pipe(drop).fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
        rows.append(dict(stage="development", spec=name, test=order[k], n=len(te), **metrics(te.y.values, p)))
        print(rows[-1], flush=True)
# (c) post-hoc T10 for the horizon-free variant
feats = [f for f in FEAT if f not in HOR]
tr = P[P.transition.isin(order[:9])]; te = P[P.transition == "T10"]
p = make_pipe(HOR).fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
m = metrics(te.y.values, p); pred = p >= TH
m.update(tp=int((pred & (te.y.values == 1)).sum()), fp=int((pred & (te.y.values == 0)).sum()))
rows.append(dict(stage="post_hoc_T10_sensitivity", spec="A5_minus_horizon_51", test="T10", n=len(te), **m))
rows.append(dict(stage="frozen_reference", spec="A5_frozen_53", test="T10", n=len(te), **metrics(te.y.values, fz)))
out = pd.DataFrame(rows); out.to_csv(ev / "horizon_free_sensitivity.csv", index=False)
dev = out[out.stage == "development"].groupby("spec")[["roc_auc", "pr_auc", "brier"]].mean()
json.dump({"retraining_reproducibility": rep, "development_macro": dev.to_dict("index")}, open(ev / "horizon_free_sensitivity_summary.json", "w"), indent=1)
print(dev); print(out.tail(2).to_string())
