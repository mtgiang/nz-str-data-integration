# -*- coding: utf-8 -*-
"""ONE-RUN READ-ONLY publication evidence package. Does not fit or evaluate models."""
from pathlib import Path
from datetime import datetime
import json, hashlib, zipfile, shutil
import pandas as pd

ROOT=Path(r'C:\Users\yxt2919\7th_NZ_STR_Construction_AI')
if not ROOT.exists():
    ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'02_intermediate').exists()),ROOT)
if not ROOT.exists():raise FileNotFoundError('Project root not found')
OUT=ROOT/'05_results'/'FINAL_PUBLICATION_PACKAGE'
OUT.mkdir(parents=True,exist_ok=True)
FREEZE=ROOT/'05_results/development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json'
if not FREEZE.exists():raise FileNotFoundError(f'Frozen spec missing: {FREEZE}')
spec=json.loads(FREEZE.read_text(encoding='utf-8'))
features=spec['final_primary_model']['predictors']
assert len(features)==53 and len(set(features))==53
registry_options=[ROOT/'02_intermediate/modelling/feature_registry.csv',ROOT/'05_results/development_modelling_IB/freeze/feature_registry_FINAL_FROZEN.csv',ROOT/'02_intermediate/modelling/feature_registry_FINAL_FROZEN.csv']
reg_path=next((p for p in registry_options if p.exists()),None)
rows=[]
for f in features:
    family=('RENTAL_MARKET' if f.startswith('rental_') else 'CONSTRUCTION' if f.startswith('construction_') else 'CALENDAR' if f.startswith(('open_ratio_','median_minimum_nights_')) else 'REVIEW_HISTORY' if f.startswith(('review_','listing_review_','days_since_last_review')) else 'SEASONALITY' if f in ('forecast_horizon_days','historical_seasonality_index_horizon','month_sin','month_cos') else 'GEOGRAPHY' if f=='ta_code' else 'LISTING_PLATFORM')
    status='SOURCE_CUTOFF_NOT_VERIFIED' if family in ('RENTAL_MARKET','CONSTRUCTION','CALENDAR','REVIEW_HISTORY','LISTING_PLATFORM') else 'CHECK_FEATURE_LINEAGE'
    rows.append(dict(feature=f,source_family=family,point_in_time_status=status))
audit=pd.DataFrame(rows)
if reg_path is not None:
    reg=pd.read_csv(reg_path)
    if 'feature' in reg.columns:
        audit=audit.merge(reg.drop_duplicates('feature'),on='feature',how='left',validate='one_to_one')
        audit['registry_present']=audit['dtype'].notna() if 'dtype' in audit else True
else:
    audit['registry_present']=False
audit.to_csv(OUT/'01_frozen_53_predictor_evidence.csv',index=False)

# Verify known source timeline, without claiming real-time data availability.
backbone=ROOT/'02_intermediate/integration/listing_origin_backbone_T1_T10.parquet'
if not backbone.exists():raise FileNotFoundError(backbone)
b=pd.read_parquet(backbone,columns=['transition','listing_id','snapshot_date','last_scraped','price'])
assert len(b)==491140 and not b.duplicated(['transition','listing_id']).any()
b['snapshot_date']=pd.to_datetime(b['snapshot_date']);b['last_scraped']=pd.to_datetime(b['last_scraped'])
b['delay_days']=(b['last_scraped']-b['snapshot_date']).dt.total_seconds()/86400
b['price_numeric']=pd.to_numeric(b['price'],errors='coerce')
timeline=b.groupby('transition').agg(n=('listing_id','size'),price_available=('price_numeric','count'),price_median=('price_numeric','median'),scraped_after_nominal=('delay_days',lambda x:int(x.gt(0).sum())),scraped_on_nominal=('delay_days',lambda x:int(x.eq(0).sum())),median_scrape_delay_days=('delay_days','median'),max_scrape_delay_days=('delay_days','max')).reset_index()
timeline['transition_number']=timeline.transition.str.extract(r'(\d+)').astype(int)
timeline=timeline.sort_values('transition_number').drop(columns='transition_number')
timeline['price_missing']=timeline.n-timeline.price_available
timeline['price_missing_pct']=(100*timeline.price_missing/timeline.n).round(3)
timeline.to_csv(OUT/'02_price_and_scrape_timeline_VERIFIED.csv',index=False)
assert int(timeline.n.sum())==491140
assert int(timeline.price_available.sum())==318242
assert int(timeline.scraped_after_nominal.sum())==483683

# Existing evidence only: copy selected compact outputs without changing originals.
patterns={
'03_locked_primary_metrics.csv':['05_results/final_locked_T10_primary/**/*metrics*.csv','05_results/final_locked_T10_primary/**/*summary*.csv'],
'04_locked_secondary_metrics.csv':['05_results/secondary_price_resilience_K/K_final_locked_T10_metrics.csv'],
'05_T10_bootstrap_CI.csv':['05_results/manuscript_final_L_to_O/tables/L_FINAL_95CI.csv'],
'06_T10_risk_deciles.csv':['05_results/manuscript_final_L_to_O/tables/N_primary_prospective_risk_deciles.csv'],
'07_SHAP_53.csv':['05_results/manuscript_final_L_to_O/tables/M_SHAP_53_frozen_features.csv'],
'08_frozen_spec.json':['05_results/development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json'],
}
status=[]
for target,pats in patterns.items():
    found=next((x for pattern in pats for x in sorted(ROOT.glob(pattern)) if x.is_file()),None)
    if found:
        shutil.copy2(found,OUT/target)
        status.append((target,'COPIED',str(found.relative_to(ROOT))))
    else:status.append((target,'NOT_FOUND','No matching frozen output; not fabricated'))
# A source-code calendar is not proof of official publication dates.
code=ROOT/'04_code/03_build_construction_temporal_alignment.py'
if code.exists():shutil.copy2(code,OUT/'09_construction_alignment_SOURCE_TO_VERIFY.py')

report=f'''# Final publication evidence package — {datetime.now():%Y-%m-%d %H:%M}

## Frozen scientific contract
- 53 frozen A5 predictors; no retraining, tuning, threshold change, or T10 rerun.
- Primary: next-snapshot platform disappearance, NOT confirmed business failure.
- Secondary: within-peer advertised-price percentile change, NOT returns.

## Independently checked from backbone
- {len(b):,} rows; unique (transition, listing_id) keys.
- {int(b.price_numeric.notna().sum()):,} valid prices; {int(b.price_numeric.isna().sum()):,} missing ({b.price_numeric.isna().mean():.2%}).
- T3, T4, T5 price fully missing.
- {int(b.delay_days.gt(0).sum()):,} rows have last_scraped later than nominal snapshot_date ({b.delay_days.gt(0).mean():.2%}).
- Date discrepancy does NOT by itself establish leakage.

## Unresolved temporal provenance (must be QUALIFIED, not claimed VERIFIED)
- Stats NZ construction release-date source verification and actual panel as-of join.
- Rental-market release-date lag and as-of join.
- Calendar capture date versus nominal snapshot and use of forward calendar days.
- Review history cutoff and availability of review-derived predictors.
- Listing last_scraped versus the operational prediction-origin definition.

## Publication guidance
- Report locked primary and secondary performance only from copied frozen results.
- Explain secondary label ineligibility at T3–T5; do not claim price trend across missing periods.
- Do not claim 'leakage-free', 'real-time prospective', causal effects, booking/occupancy, or ROI until proven.
- Preserve existing locked T10 numbers and identify unresolved availability as a limitation.

## Files copied from existing frozen results
'''+''.join(f'- {a}: {s} — {p}\n' for a,s,p in status)
(OUT/'00_FINAL_README.md').write_text(report,encoding='utf-8')
manifest=[]
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.name!='10_FILE_MANIFEST.csv':
        manifest.append(dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
pd.DataFrame(manifest).to_csv(OUT/'10_FILE_MANIFEST.csv',index=False)
zip_path=OUT.parent/'FINAL_PUBLICATION_AUDIT_PACKAGE.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(OUT.iterdir()):
        if f.is_file():z.write(f,arcname=f.name)
print('\nFINAL PUBLICATION PACKAGE CREATED:',zip_path)
print('\n'.join(f'{a}: {s}' for a,s,_ in status))
print('Rows:',len(b),'Frozen predictors:',len(features),'Price missing:',int(b.price_numeric.isna().sum()),'Scraped after nominal:',int(b.delay_days.gt(0).sum()))
print('NEXT: Upload only FINAL_PUBLICATION_AUDIT_PACKAGE.zip. No other analysis run needed.')
