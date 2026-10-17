"""Post-freeze manuscript production L–O. Run: %run ".../production_L_to_O.py"
Never trains, selects, calibrates or modifies the locked J/K models.
"""
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=FutureWarning)
SEED=2026
BOOTSTRAPS=600
SHAP_SAMPLE=3000

def require(ok,msg):
    if not ok: raise RuntimeError('HARD FAIL: '+msg)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def root():
    for p in [Path.cwd(),*Path.cwd().parents,Path(r'C:\Users\yxt2919\7th_NZ_STR_Construction_AI')]:
        if (p/'02_intermediate/modelling/modelling_panel_T1_T10.parquet').exists():return p
    raise FileNotFoundError('Project root not found; run in project directory')
def read(path):
    require(path.exists(),f'Missing {path}')
    return pd.read_parquet(path)
def metric_primary(y,p):
    if len(np.unique(y))<2:return None
    return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),brier=float(brier_score_loss(y,p)))
def metric_secondary(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))
def bootstrap(df,ycol,pcol,fn,tag,out,rng):
    # TA-cluster bootstrap; each resample draws TAs with replacement, retaining all their T10 listings.
    clusters=df.groupby('ta_code',dropna=False,sort=False).indices
    keys=list(clusters); require(len(keys)>1,tag+' needs >=2 TA clusters')
    groups=[np.asarray(clusters[k],dtype=np.int64) for k in keys]
    y=df[ycol].to_numpy(float); p=df[pcol].to_numpy(float)
    observed=fn(y,p);require(observed is not None,tag+' observed metric undefined')
    rows=[]; skipped=0
    for b in range(BOOTSTRAPS):
        draw=rng.integers(0,len(groups),len(groups))
        idx=np.concatenate([groups[i] for i in draw])
        m=fn(y[idx],p[idx])
        if m is None:skipped+=1;continue
        rows.append(dict(replicate=b,**m))
    require(len(rows)>=int(BOOTSTRAPS*.90),tag+' too many invalid bootstrap replicates')
    rep=pd.DataFrame(rows);rep.to_csv(out/f'{tag}_cluster_bootstrap_replicates.csv',index=False)
    summary=[]
    for key,v in observed.items():
        vals=rep[key].to_numpy()
        summary.append(dict(outcome=tag,metric=key,estimate=v,ci95_low=float(np.quantile(vals,.025)),ci95_high=float(np.quantile(vals,.975)),n=len(df),n_ta=len(keys),bootstrap_valid=len(rep),bootstrap_skipped=skipped))
    return pd.DataFrame(summary)
def plot_save(fig,out,stem):
    fig.savefig(out/(stem+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)
def main():
    r=root(); base=r/'05_results';out=base/'manuscript_final_L_to_O'
    for d in ['tables','figures','manuscript','provenance']:(out/d).mkdir(parents=True,exist_ok=True)
    t=out/'tables';fig=out/'figures';rng=np.random.default_rng(SEED)
    j=base/'final_locked_T10_primary';k=base/'secondary_price_resilience_K'
    jp=j/'predictions/J_final_locked_T10_primary_predictions.parquet'
    kp=k/'K_final_locked_T10_secondary_predictions.parquet'
    jf=j/'models/J_final_XGB_A5_T1_T9_pipeline.joblib'
    kfreeze=k/'K_DEVELOPMENT_MODEL_FREEZE.json'
    ksha=k/'K_DEVELOPMENT_MODEL_FREEZE.sha256'
    jspec=base/'development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json'
    jsha=base/'development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.sha256'
    for p in [jp,kp,jf,kfreeze,ksha,jspec,jsha]:require(p.exists(),f'Missing {p}')
    require(ksha.read_text().strip()==sha(kfreeze),'K frozen spec hash mismatch')
    require(jsha.read_text().split()[0]==sha(jspec),'J frozen spec hash mismatch')
    require(json.loads(kfreeze.read_text())['selected_model']=='BASELINE_ZERO_CHANGE','K final model differs from frozen zero-change baseline')
    primary=read(jp);secondary=read(kp)
    require(len(primary)==51096 and len(secondary)==43857,'Unexpected locked sample counts')
    require(primary.listing_id.is_unique and secondary.listing_id.is_unique,'Duplicate listing IDs')
    require((primary.transition=='T10').all() and (secondary.transition=='T10').all(),'Non-T10 rows in final predictions')
    require(np.allclose(secondary.predicted_secondary.to_numpy(float),0),'Secondary predictions not frozen zero-change')
    require((primary.y_disappear.sum()==1147),'Primary labels differ from frozen J')
    panel_path=r/'02_intermediate/modelling/modelling_panel_T1_T10.parquet'
    cols=pd.read_parquet(panel_path).columns.tolist()
    require('ta_code' in cols,'TA code missing; cannot perform TA cluster bootstrap')
    geo=pd.read_parquet(panel_path,columns=['transition','listing_id','ta_code'])
    geo=geo[geo.transition=='T10'].drop(columns='transition')
    require(geo.listing_id.is_unique,'Duplicate TA mapping')
    p=primary.merge(geo,on='listing_id',validate='one_to_one',how='left')
    s=secondary.merge(geo,on='listing_id',validate='one_to_one',how='left')
    require(p.ta_code.notna().all() and s.ta_code.notna().all(),'Missing TA code')
    require(p[['y_disappear','p_disappear_raw','p_disappear_calibrated']].notna().all().all(),'Missing J outcome/probabilities')
    require(s[['y_secondary','predicted_secondary']].notna().all().all(),'Missing K outcome/predictions')
    print('L — TA-cluster bootstrap,',BOOTSTRAPS,'replicates')
    l1=bootstrap(p,'y_disappear','p_disappear_raw',metric_primary,'L_primary_raw',t,rng)
    l2=bootstrap(p,'y_disappear','p_disappear_calibrated',metric_primary,'L_primary_platt',t,rng)
    l3=bootstrap(s,'y_secondary','predicted_secondary',metric_secondary,'L_secondary',t,rng)
    ci=pd.concat([l1,l2,l3],ignore_index=True);ci.to_csv(t/'L_FINAL_95CI.csv',index=False)
    # Geographic heterogeneity, descriptive not independent holdout metrics.
    area=[]
    for ta,g in p.groupby('ta_code'):
        y=g.y_disappear.to_numpy(int);v=g.p_disappear_raw.to_numpy(float)
        area.append(dict(ta_code=ta,n=len(g),events=int(y.sum()),observed_event_rate=float(y.mean()),mean_predicted_risk=float(v.mean()),roc_auc=float(roc_auc_score(y,v)) if len(np.unique(y))==2 else np.nan))
    pd.DataFrame(area).sort_values('n',ascending=False).to_csv(t/'L_primary_TA_heterogeneity.csv',index=False)
    print('M — SHAP / exact tree contribution decomposition on a fixed, deterministic T10 sample')
    spec=json.loads(jspec.read_text());features=spec['final_primary_model']['predictors']
    require(len(features)==53,'Frozen feature count mismatch')
    pipe=joblib.load(jf)
    require(hasattr(pipe,'named_steps') and {'preprocess','model'}.issubset(pipe.named_steps),'Unknown frozen J pipeline')
    sample=p[['listing_id']].sample(n=min(SHAP_SAMPLE,len(p)),random_state=SEED).reset_index(drop=True)
    full=pd.read_parquet(panel_path,columns=['transition','listing_id']+features)
    full=full[full.transition=='T10'].drop(columns='transition')
    sample=sample.merge(full,on='listing_id',how='left',validate='one_to_one',sort=False)
    require(len(sample)==min(SHAP_SAMPLE,len(p)),'SHAP sample mismatch')
    X=pipe.named_steps['preprocess'].transform(sample[features])
    from xgboost import DMatrix
    booster=pipe.named_steps['model'].get_booster()
    contrib=booster.predict(DMatrix(X),pred_contribs=True)
    require(contrib.shape[0]==len(sample) and contrib.shape[1]==X.shape[1]+1,'XGBoost contribution dimensions mismatch')
    transformed=pipe.named_steps['preprocess'].get_feature_names_out()
    require(len(transformed)==X.shape[1],'Transformed feature names mismatch')
    # Map one-hot categories back to original frozen predictor; avoid naive underscore split.
    prep=pipe.named_steps['preprocess'];num_cols=list(prep.transformers_[0][2]);cat_cols=list(prep.transformers_[1][2]);
    numeric_pipe=prep.named_transformers_['numeric']; numeric_imp=numeric_pipe.named_steps['imputer']
    num_names=list(numeric_imp.get_feature_names_out(num_cols))
    onehot=prep.named_transformers_['categorical'].named_steps['onehot']
    cat_out=onehot.get_feature_names_out(cat_cols)
    cat_orig=[]
    for c,cats in zip(cat_cols,onehot.categories_):cat_orig.extend([c]*len(cats))
    mapped=[]
    for name in num_names:
        mapped.append(name[len('missingindicator_'):] if name.startswith('missingindicator_') else name)
    mapped+=cat_orig
    require(len(mapped)==X.shape[1],'SHAP attribution mapping mismatch')
    check=booster.predict(DMatrix(X),output_margin=True)
    require(np.allclose(contrib.sum(axis=1),check,atol=1e-4,rtol=1e-4),'SHAP additivity mismatch')
    shap_df=pd.DataFrame(contrib[:,:-1],columns=transformed)
    aggregated=pd.DataFrame(index=np.arange(len(sample)))
    for feature in features:
        idx=[i for i,name in enumerate(mapped) if name==feature]
        require(len(idx)>0,'Frozen feature missing in SHAP mapping: '+feature)
        aggregated[feature]=contrib[:,idx].sum(axis=1)
    importance=pd.DataFrame({'feature':features,'mean_abs_shap_log_odds':[float(np.abs(aggregated[f]).mean()) for f in features]})
    importance=importance.sort_values('mean_abs_shap_log_odds',ascending=False)
    importance.to_csv(t/'M_SHAP_53_frozen_features.csv',index=False)
    shap_sample=sample[['listing_id']].copy();shap_sample['base_margin']=contrib[:,-1];shap_sample['predicted_margin']=check
    shap_sample.to_parquet(t/'M_SHAP_sample_audit.parquet',index=False)
    fig1,ax=plt.subplots(figsize=(9,7));top=importance.head(20).iloc[::-1]
    ax.barh(top.feature,top.mean_abs_shap_log_odds);ax.set_xlabel('Mean absolute SHAP contribution (log-odds)');ax.set_title('Primary disappearance model: top 20 predictors')
    plot_save(fig1,fig,'M_SHAP_top20')
    # N: prospective risk tiers; retrospective secondary associations strictly labelled diagnostic.
    print('N — Prospective risk stratification and retrospective secondary diagnostics')
    p['risk_decile']=pd.qcut(p.p_disappear_raw.rank(method='first'),10,labels=False).astype(int)+1
    risk=p.groupby('risk_decile').agg(n=('listing_id','size'),observed_disappearances=('y_disappear','sum'),observed_disappearance_rate=('y_disappear','mean'),mean_predicted_disappearance=('p_disappear_raw','mean'),min_predicted_risk=('p_disappear_raw','min'),max_predicted_risk=('p_disappear_raw','max')).reset_index()
    risk.to_csv(t/'N_primary_prospective_risk_deciles.csv',index=False)
    joined=p[['listing_id','ta_code','risk_decile','p_disappear_raw','y_disappear']].merge(s[['listing_id','y_secondary','predicted_secondary']],on='listing_id',how='left',validate='one_to_one')
    joined['secondary_label_available']=joined.y_secondary.notna()
    retrospective=joined.groupby('risk_decile').agg(n=('listing_id','size'),secondary_available=('secondary_label_available','sum'),observed_disappearance_rate=('y_disappear','mean'),observed_mean_price_rank_change=('y_secondary','mean'),observed_median_price_rank_change=('y_secondary','median')).reset_index()
    retrospective['secondary_coverage']=retrospective.secondary_available/retrospective.n
    retrospective.to_csv(t/'N_RETROSPECTIVE_price_resilience_by_predicted_risk.csv',index=False)
    joined.to_parquet(t/'N_joined_listing_level_diagnostic.parquet',index=False)
    fig2,ax=plt.subplots(figsize=(8,5));ax.plot(risk.risk_decile,risk.observed_disappearance_rate,marker='o',label='Observed');ax.plot(risk.risk_decile,risk.mean_predicted_disappearance,marker='s',label='Predicted raw');ax.set(xlabel='Predicted disappearance risk decile (1 = lowest)',ylabel='Disappearance rate',title='Locked T10: prospective risk ranking versus observed outcomes');ax.legend();plot_save(fig2,fig,'N_primary_risk_deciles')
    fig3,ax=plt.subplots(figsize=(8,5));ax.plot(retrospective.risk_decile,retrospective.observed_mean_price_rank_change,marker='o');ax.axhline(0,linewidth=1,linestyle='--');ax.set(xlabel='Predicted primary risk decile',ylabel='Observed subsequent advertised-price percentile change',title='Retrospective diagnostic only — NOT prospective price prediction');plot_save(fig3,fig,'N_RETROSPECTIVE_price_rank_change')
    # O: manuscript tables and text (descriptive claims only, no causality).
    print('O — Manuscript-ready results and provenance')
    for name,frame in [('Table_L_confidence_intervals',ci),('Table_M_SHAP_top20',importance.head(20)),('Table_N_risk_deciles',risk),('Table_N_retrospective_price',retrospective)]:
        (out/'manuscript'/f'{name}.tex').write_text(frame.to_latex(index=False,float_format=lambda x:f'{x:.4f}',escape=True),encoding='utf-8')
    get=lambda o,m:ci.loc[(ci.outcome==o)&(ci.metric==m)].iloc[0]
    auc=get('L_primary_raw','roc_auc');pr=get('L_primary_raw','pr_auc');brier=get('L_primary_raw','brier');mae=get('L_secondary','mae');rmse=get('L_secondary','rmse');r2=get('L_secondary','r2')
    text=f'''\\subsection{{Locked holdout evaluation and uncertainty}}\nThe final primary model was evaluated on 51,096 T10 listing transitions, of which 1,147 were labelled as next-period platform disappearances (2.245\\%). Raw XGBoost predictions achieved ROC--AUC {auc.estimate:.4f} (95\\% TA-cluster bootstrap CI {auc.ci95_low:.4f}--{auc.ci95_high:.4f}), PR--AUC {pr.estimate:.4f} (95\\% CI {pr.ci95_low:.4f}--{pr.ci95_high:.4f}), and Brier score {brier.estimate:.5f} (95\\% CI {brier.ci95_low:.5f}--{brier.ci95_high:.5f}). The frozen raw-probability threshold was not re-optimised on T10.\n\nFor the secondary outcome, the development-selected zero-change baseline was evaluated on 43,857 eligible T10 listings. It achieved MAE {mae.estimate:.5f} (95\\% TA-cluster bootstrap CI {mae.ci95_low:.5f}--{mae.ci95_high:.5f}), RMSE {rmse.estimate:.5f} (95\\% CI {rmse.ci95_low:.5f}--{rmse.ci95_high:.5f}), and $R^2$ {r2.estimate:.5f} (95\\% CI {r2.ci95_low:.5f}--{r2.ci95_high:.5f}). Spearman correlation is undefined because all zero-change predictions are identical. The secondary outcome measures changes in within-peer advertised-price percentile, not realised prices, revenue, occupancy, or investment returns.\n\n\\subsection{{Interpretability and screening diagnostics}}\nTreeSHAP-style exact XGBoost contributions were calculated for a fixed random sample of {len(sample):,} T10 listings and aggregated from transformed model inputs back to the 53 frozen predictors. The ranking describes model associations on the log-odds scale and must not be interpreted causally. Predicted primary disappearance risk was summarised by decile. Subsequent observed advertised-price-percentile changes were examined across these risk groups strictly as retrospective diagnostics; they are not available at the prediction origin and do not form a prospective two-dimensional investment signal.\n'''
    (out/'manuscript'/'O_manuscript_results_READY_TO_REVIEW.tex').write_text(text,encoding='utf-8')
    (out/'manuscript'/'O_interpretation_notes.txt').write_text('Primary disappearance is a platform-observation event, not confirmed property exit or business failure.\nPrice resilience is advertised-price rank change, not transaction price, realised rental income, occupancy, or investment return.\nPlatt calibration did not improve T10 Brier; report raw and Platt results from J.\nK secondary zero-change baseline has undefined Spearman, not Spearman=0.\nRisk deciles are predictive screening; observed future price changes are retrospective only.\nTA bootstrap quantifies geographic-cluster resampling variability conditional on this one T10 window, not temporal transfer uncertainty.\n',encoding='utf-8')
    inputs=[jp,kp,jf,kfreeze,jspec,panel_path]
    artifacts=[x for x in out.rglob('*') if x.is_file() and x.name!='O_output_hashes.csv']
    pd.DataFrame([{'path':str(x.relative_to(out)),'sha256':sha(x)} for x in artifacts]).to_csv(out/'provenance/O_output_hashes.csv',index=False)
    (out/'provenance/O_run_manifest.json').write_text(json.dumps({'seed':SEED,'bootstrap_replicates':BOOTSTRAPS,'shap_sample':len(sample),'cluster_variable':'ta_code','test_transition':'T10','secondary_model':'BASELINE_ZERO_CHANGE','no_model_retraining':True,'input_sha256':{str(x.relative_to(r)):sha(x) for x in inputs}},indent=2),encoding='utf-8')
    print('\nFINAL L–O RESULTS — 95% TA-CLUSTER BOOTSTRAP\n',ci.to_string(index=False))
    print('\nTOP SHAP FEATURES\n',importance.head(15).to_string(index=False))
    print('\nRISK DECILES\n',risk.to_string(index=False))
    print('\nOUTPUT:',out)
    print('PRODUCTION L–O STATUS: PASS — manuscript tables, figures, uncertainty and interpretability saved')
if __name__=='__main__':main()
