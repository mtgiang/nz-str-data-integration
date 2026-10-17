"""Block K: development freeze, then one-time locked secondary T10 evaluation.
Run in Jupyter: %run "C:/path/to/block_K_secondary.py" --stage dev
After reviewing frozen dev outputs: %run "C:/path/to/block_K_secondary.py" --stage final
"""
import argparse, hashlib, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SEED=2026
DEV_TESTS=['T6','T7','T8','T9']
DEV_TRAIN=[['T1'],['T1','T6'],['T1','T6','T7'],['T1','T6','T7','T8']]
ALL_TRAIN=['T1','T6','T7','T8','T9']
EXPECTED_N={'T1':42368,'T6':43063,'T7':42810,'T8':43624,'T9':43977,'T10':43857}
PARAMS=dict(n_estimators=500,max_depth=6,learning_rate=.05,subsample=.8,
            colsample_bytree=.8,min_child_weight=5,reg_lambda=1.,tree_method='hist',
            objective='reg:squarederror',random_state=SEED,n_jobs=4)
# Frozen modelling family and model configuration chosen before secondary T10 is accessed.
CANDIDATES=['BASELINE_ZERO_CHANGE','BASELINE_TRAIN_MEAN','XGB_A5_STRUCTURED_FULL']


def root():
    opts=[Path.cwd(),*Path.cwd().parents,Path(r'C:\Users\yxt2919\7th_NZ_STR_Construction_AI')]
    for p in opts:
        if (p/'02_intermediate/modelling/modelling_panel_T1_T10.parquet').exists():return p
    raise FileNotFoundError('Project root not found')


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fail(cond,msg):
    if not cond:raise RuntimeError('HARD FAIL: '+msg)


def read_secondary(p):
    # Do not guess a target if the outcome schema uses an unfamiliar label.
    names=['snapshot_advertised_price_resilience','advertised_price_resilience',
           'relative_advertised_price_resilience','relative_price_percentile_change',
           'delta_price_percentile','price_percentile_change','price_rank_change',
           'secondary_target','y_secondary','y_price_resilience',
           'delta_relative_advertised_price_position',
           'delta_advertised_price_percentile', 'delta_percentile_rank',
           'advertised_price_percentile_change']
    d=pd.read_parquet(p)
    fail({'transition','listing_id'}.issubset(d.columns),f'Outcome missing keys: {list(d.columns)}')
    available=[c for c in names if c in d.columns]
    if len(available)!=1:
        numeric=[c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        semantic=[c for c in numeric if ('delta' in c.lower() or 'resilien' in c.lower() or 'change' in c.lower())
                  and ('price' in c.lower() or 'percentile' in c.lower())]
        if len(available)==0 and len(semantic)==1:available=semantic
    fail(len(available)==1, 'Cannot uniquely identify secondary target. Available columns: '+str(list(d.columns)))
    col=available[0]
    print('Frozen secondary outcome target column:',col)
    return d[['transition','listing_id',col]].rename(columns={col:'y_secondary'}),col


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    fail(np.isfinite(y).all() and np.isfinite(p).all(),'Nonfinite target or predictions')
    s=spearmanr(y,p).statistic if np.unique(p).size>1 and np.unique(y).size>1 else np.nan
    return dict(n=len(y),mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),
                r2=float(r2_score(y,p)),spearman=float(s) if np.isfinite(s) else None)


def prepare(x,features):
    cat=[c for c in features if pd.api.types.is_object_dtype(x[c]) or
         isinstance(x[c].dtype,pd.CategoricalDtype) or pd.api.types.is_string_dtype(x[c])]
    num=[c for c in features if c not in cat]
    # Type conversion is explicitly forbidden here: unexpected nonnumeric values should fail.
    fail(all(pd.api.types.is_numeric_dtype(x[c]) for c in num),'Unexpected nonnumeric predictor')
    # Fail if an entire feature is unavailable in training; do not learn its imputation from test.
    fail(all(x[c].notna().any() for c in num),'All-missing numeric feature in training')
    fail(all(x[c].notna().any() for c in cat),'All-missing categorical feature in training')
    return ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True))]),num),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                 ('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)],
        remainder='drop',sparse_threshold=.3)


def xgb_model(train,features):
    prep=prepare(train[features],features)
    model=Pipeline([('preprocess',prep),('model',XGBRegressor(**PARAMS))])
    model.fit(train[features],train.y_secondary.to_numpy(float))
    return model


def save_json(obj,p):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def main(stage):
    project=root();out=project/'05_results/secondary_price_resilience_K';out.mkdir(parents=True,exist_ok=True)
    panel_path=project/'02_intermediate/modelling/modelling_panel_T1_T10.parquet'
    outcome_path=project/'02_intermediate/outcomes/secondary_snapshot_advertised_price_resilience_T1_T10_frozen.parquet'
    spec_path=project/'05_results/development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json'
    fail(all(p.exists() for p in [panel_path,outcome_path,spec_path]),'Required frozen input missing')
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    features=spec['final_primary_model']['predictors']
    fail(len(features)==53 and len(set(features))==53,'Frozen A5 predictors invalid')
    # Exclude future price or secondary-target-derived fields, and raw anchor price.
    forbidden=['future','next_','target','resilien','percentile','price_rank','price_percentile']
    fail(not any(any(k in c.lower() for k in forbidden) for c in features),
         'Feature name resembles future/secondary outcome; manual audit required')
    fail('price' not in features,'Raw listing price is forbidden in A5')
    outcome,target_col=read_secondary(outcome_path)
    fail(not outcome.duplicated(['transition','listing_id']).any(),'Duplicate secondary keys')
    # Load only keys+frozen features; avoid any other panel outcome column.
    panel=pd.read_parquet(panel_path,columns=['transition','listing_id']+features)
    fail(not panel.duplicated(['transition','listing_id']).any(),'Duplicate panel keys')
    fail(len(panel)==491140,'Unexpected panel size')
    data=panel.merge(outcome,on=['transition','listing_id'],how='left',validate='one_to_one',indicator=True)
    fail((data['_merge']=='both').all(),'Secondary outcome row coverage mismatch')
    data=data.drop(columns='_merge')
    fail(data.y_secondary.dropna().between(-1,1).all(),'Secondary percentile difference outside [-1,1]')
    counts=data.groupby('transition').y_secondary.count().to_dict()
    print('SECONDARY LABEL COVERAGE:',counts)
    fail(all(counts.get(t)==n for t,n in EXPECTED_N.items()),'Secondary frozen counts mismatch')
    fail(all(counts.get(t,0)==0 for t in ['T2','T3','T4','T5']),'Missing-price transitions unexpectedly labelled')
    freeze_path=out/'K_DEVELOPMENT_MODEL_FREEZE.json'
    dev_path=out/'K_development_rolling_metrics.csv'
    devpred_path=out/'K_development_rolling_predictions.parquet'
    if stage=='dev':
        fail(not freeze_path.exists(),'Development freeze already exists; do not overwrite it')
        rows=[];preds=[]
        for training,test_t in zip(DEV_TRAIN,DEV_TESTS):
            tr=data[data.transition.isin(training)&data.y_secondary.notna()].copy()
            te=data[(data.transition==test_t)&data.y_secondary.notna()].copy()
            fail(len(tr)>0 and len(te)==EXPECTED_N[test_t],f'Invalid split {test_t}')
            y=te.y_secondary.to_numpy(float)
            predictions={'BASELINE_ZERO_CHANGE':np.zeros(len(te)),
                         'BASELINE_TRAIN_MEAN':np.full(len(te),tr.y_secondary.mean())}
            model=xgb_model(tr,features)
            predictions['XGB_A5_STRUCTURED_FULL']=model.predict(te[features])
            for name,p in predictions.items():
                score=metrics(y,p)
                rows.append(dict(test_transition=test_t,train_transitions=','.join(training),model=name,**score))
                preds.append(pd.DataFrame(dict(transition=test_t,listing_id=te.listing_id.to_numpy(),
                                               y_secondary=y,prediction=p,model=name)))
            print('Development fold',test_t,'complete')
        table=pd.DataFrame(rows)
        # Pre-specified primary selection metric: lowest macro MAE; RMSE tie-breaker.
        rank=(table.groupby('model',as_index=False)[['mae','rmse']].mean()
              .sort_values(['mae','rmse','model'],kind='stable'))
        selected=str(rank.iloc[0]['model'])
        freeze={'block':'K','status':'DEVELOPMENT_FROZEN_BEFORE_T10','selected_model':selected,
                'candidate_models':CANDIDATES,'selection_rule':'lowest unweighted macro MAE; RMSE tie-break',
                'features':features,'n_features':53,'xgb_parameters':PARAMS,
                'development_train_transitions':ALL_TRAIN,'development_tests':DEV_TESTS,
                'target_column':target_col,'target_definition':'future within-peer snapshot advertised-price percentile minus anchor percentile',
                'source_outcome_sha256':digest(outcome_path),'source_panel_sha256':digest(panel_path),
                'source_primary_spec_sha256':digest(spec_path),'t10_used_for_selection':False,
                'secondary_eligible_counts':{t:int(n) for t,n in EXPECTED_N.items()}}
        table.to_csv(dev_path,index=False)
        pd.concat(preds,ignore_index=True).to_parquet(devpred_path,index=False)
        save_json(freeze,freeze_path)
        (out/'K_DEVELOPMENT_MODEL_FREEZE.sha256').write_text(digest(freeze_path)+'\n',encoding='utf-8')
        rank.to_csv(out/'K_development_model_ranking.csv',index=False)
        print('\nDEVELOPMENT MODEL RANKING (LOWER MAE BETTER)\n',rank.to_string(index=False))
        print('\nK DEVELOPMENT STATUS: PASS — SECONDARY MODEL FROZEN; T10 NOT EVALUATED')
        print('NEXT: review the frozen development output, then run --stage final exactly once')
    elif stage=='final':
        fail(freeze_path.exists(),'Development freeze missing; run --stage dev first')
        hashfile=out/'K_DEVELOPMENT_MODEL_FREEZE.sha256'
        fail(hashfile.exists() and hashfile.read_text().strip()==digest(freeze_path),'Development freeze hash mismatch')
        freeze=json.loads(freeze_path.read_text(encoding='utf-8'))
        fail(freeze['status']=='DEVELOPMENT_FROZEN_BEFORE_T10','Invalid freeze status')
        fail(freeze['source_outcome_sha256']==digest(outcome_path) and
             freeze['source_panel_sha256']==digest(panel_path) and
             freeze['source_primary_spec_sha256']==digest(spec_path),'Input changed since development freeze')
        fail(freeze['features']==features and freeze['target_column']==target_col,'Feature/target mismatch')
        fail(freeze['selected_model'] in CANDIDATES,'Unknown frozen model')
        final_pred_path=out/'K_final_locked_T10_secondary_predictions.parquet'
        fail(not final_pred_path.exists(),'T10 already evaluated; do not overwrite final predictions')
        train=data[data.transition.isin(ALL_TRAIN)&data.y_secondary.notna()].copy()
        test=data[(data.transition=='T10')&data.y_secondary.notna()].copy()
        fail(len(test)==43857,'Unexpected T10 eligible secondary sample')
        selected=freeze['selected_model']
        if selected=='BASELINE_ZERO_CHANGE':p=np.zeros(len(test)); fitted=None
        elif selected=='BASELINE_TRAIN_MEAN':p=np.full(len(test),train.y_secondary.mean());fitted=None
        else:
            fitted=xgb_model(train,features)
            p=fitted.predict(test[features])
        y=test.y_secondary.to_numpy(float)
        score=metrics(y,p)
        finalpred=pd.DataFrame({'transition':'T10','listing_id':test.listing_id.to_numpy(),
                                'y_secondary':y,'predicted_secondary':p,
                                'selected_model':selected})
        finalpred.to_parquet(final_pred_path,index=False)
        pd.DataFrame([dict(transition='T10',model=selected,**score)]).to_csv(out/'K_final_locked_T10_metrics.csv',index=False)
        if fitted is not None:joblib.dump(fitted,out/'K_final_secondary_model_T1_T9_eligible.joblib')
        save_json({'block':'K','status':'FINAL_T10_EVALUATED','selected_model':selected,
                   'frozen_development_sha256':digest(freeze_path),'n_train':len(train),
                   'n_test':len(test),'metrics':score,'no_t10_tuning':True},out/'K_final_provenance.json')
        print('\nFINAL LOCKED T10 SECONDARY RESULTS\n',pd.DataFrame([score]).to_string(index=False))
        print('\nBLOCK K FINAL STATUS: PASS — T10 SECONDARY EVALUATED; DO NOT RETUNE')
    else:raise ValueError('Stage must be dev or final')
    print('Outputs:',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['dev','final'],required=True)
    args=parser.parse_args()
    main(args.stage)
