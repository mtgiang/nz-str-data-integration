"""Read-only, post-freeze scientific audit for NZ short-term-rental panel.
Run in Jupyter: %run "C:/path/to/07_postfreeze_one_run_audit.py"
Only writes to 05_results/postfreeze_scientific_audit/ONE_RUN_... .
"""
from pathlib import Path
from datetime import datetime
import json, hashlib, zipfile, traceback
import numpy as np
import pandas as pd

ROOT = Path(r'C:\Users\yxt2919\7th_NZ_STR_Construction_AI')
OUT = ROOT / '05_results' / 'postfreeze_scientific_audit' / ('ONE_RUN_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True, exist_ok=True)
LOG=[]; STATUS=[]; FILES=[]
def note(test,status,details):
    STATUS.append({'test':test,'status':status,'details':str(details)})
    print(f'[{status}] {test}: {details}')
def save(df,name):
    p=OUT/name; df.to_csv(p,index=False); FILES.append(p); return p
def read(path):
    if not path.exists(): raise FileNotFoundError(path)
    return pd.read_parquet(path)
def dates(s): return pd.to_datetime(s,errors='coerce')
def signature(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

try:
    outcome_path=ROOT/'02_intermediate/outcomes/primary_persistence_outcome_T1_T10_frozen.parquet'
    backbone_path=ROOT/'02_intermediate/integration/listing_origin_backbone_T1_T10.parquet'
    t11_path=ROOT/'02_intermediate/listings/listings_2026-08-13.parquet'
    outcome=read(outcome_path); backbone=read(backbone_path); t11=read(t11_path)
    keys=['transition','listing_id']
    for label,df in [('outcome',outcome),('backbone',backbone)]:
        n=int(df.duplicated(keys).sum()); note(label+' key uniqueness','PASS' if n==0 else 'FAIL',f'duplicates={n}')
    ka=pd.MultiIndex.from_frame(outcome[keys]); kb=pd.MultiIndex.from_frame(backbone[keys]); same=set(ka)==set(kb)
    note('outcome/backbone key sets','PASS' if same else 'FAIL',f'same keys={same}; same row order={ka.equals(kb)}')
    if not same or outcome.duplicated(keys).any() or backbone.duplicated(keys).any(): raise RuntimeError('Key integrity failed; cannot proceed safely')
    timeline=outcome[['transition','snapshot_date','outcome_date','outcome_horizon_days']].copy()
    timeline['snapshot_date']=dates(timeline['snapshot_date']); timeline['outcome_date']=dates(timeline['outcome_date'])
    snap=timeline.groupby('transition',as_index=False).agg(origin=('snapshot_date','first'),outcome=('outcome_date','first'),rows=('snapshot_date','size'),origin_dates=('snapshot_date','nunique'),outcome_dates=('outcome_date','nunique'),horizon_values=('outcome_horizon_days','nunique'))
    snap=snap.sort_values('origin').reset_index(drop=True); ts=snap.transition.tolist()
    snap['actual_horizon_days']=(snap.outcome-snap.origin).dt.days
    snap['next_origin']=snap.origin.shift(-1)
    snap['outcome_equals_next_origin']=(snap.outcome==snap.next_origin)
    snap.loc[snap.index[-1],'outcome_equals_next_origin']=pd.NA
    save(snap,'01_snapshot_timeline.csv')
    valid_timeline=(len(ts)==10 and snap.origin_dates.eq(1).all() and snap.outcome_dates.eq(1).all() and snap.horizon_values.eq(1).all() and (snap.actual_horizon_days==outcome.groupby('transition').outcome_horizon_days.first().reindex(ts).to_numpy()).all())
    note('timeline and horizon consistency','PASS' if valid_timeline else 'FAIL',f'transitions={len(ts)}')
    pres={t:set(backbone.loc[backbone.transition==t,'listing_id']) for t in ts}
    if 'listing_id' not in t11: raise RuntimeError('T11 lacks listing_id')
    n11=int(t11.listing_id.duplicated().sum()); note('T11 unique IDs','PASS' if n11==0 else 'FAIL',f'rows={len(t11)}; duplicates={n11}')
    pres['T11']=set(t11.listing_id.dropna())
    # Verify label for ALL ten transitions
    label_rows=[]
    for i,t in enumerate(ts):
        next_t=ts[i+1] if i+1<len(ts) else 'T11'
        sub=outcome.loc[outcome.transition==t]
        actual=sub.listing_id.isin(pres[next_t]).astype(int)
        mismatch=int((actual.to_numpy()!=sub.next_period_persistence.to_numpy()).sum())
        label_rows.append({'transition':t,'next_snapshot':next_t,'rows':len(sub),'mismatches':mismatch,'disappearances':int((sub.next_period_persistence==0).sum())})
    labels=pd.DataFrame(label_rows);save(labels,'02_label_consistency.csv')
    note('all next-snapshot labels','PASS' if labels.mismatches.sum()==0 else 'FAIL',f"mismatches={labels.mismatches.sum()}; rows={labels.rows.sum()}")
    if labels.mismatches.sum(): raise RuntimeError('Label mismatches; stop reappearance interpretation')
    # Fixed T+2, eligible T1..T9, with complete T11
    rep=[]; events=[]
    for i,t in enumerate(ts[:-1]):
        missing_t=ts[i+1]; later_t=ts[i+2] if i+2<len(ts) else 'T11'
        d=outcome.loc[(outcome.transition==t)&(outcome.next_period_persistence==0),['transition','listing_id']].copy()
        d['missing_at']=missing_t;d['reappeared_at']=later_t;d['reappeared_t_plus_2']=d.listing_id.isin(pres[later_t]);events.append(d)
        n=len(d);r=int(d.reappeared_t_plus_2.sum())
        rep.append({'transition':t,'disappearance_events':n,'reappeared_t_plus_2':r,'rate_pct':100*r/n if n else np.nan,'missing_at':missing_t,'reappeared_at':later_t})
    reapp=pd.DataFrame(rep); save(reapp,'03_reappearance_fixed_Tplus2.csv');save(pd.concat(events,ignore_index=True),'04_reappearance_events.csv')
    note('fixed T+2 reappearance','PASS',f"eligible events={reapp.disappearance_events.sum()}, reappeared={reapp.reappeared_t_plus_2.sum()}, T10 right-censored")
    # T3 diagnostic
    t3=outcome.loc[outcome.transition=='T3',['listing_id','next_period_persistence']].copy()
    t3['group']=np.where(t3.next_period_persistence.eq(1),'persistent',np.where(t3.listing_id.isin(pres['T5']),'temporary_absence','continued_absence'))
    t3=t3.merge(backbone.loc[backbone.transition=='T3',['listing_id','room_type','property_type','price','number_of_reviews','number_of_reviews_l30d','last_scraped']],on='listing_id',how='left',validate='one_to_one')
    t3['price_numeric']=pd.to_numeric(t3.price,errors='coerce').astype(float)
    t3_summary=t3.groupby('group',dropna=False).agg(n=('listing_id','size'),price_n=('price_numeric','count'),median_price=('price_numeric','median'),median_reviews=('number_of_reviews','median'),median_reviews_l30d=('number_of_reviews_l30d','median')).reset_index()
    save(t3_summary,'05_T3_group_summary.csv');save(pd.crosstab(t3.room_type,t3.group,normalize='columns').reset_index(),'06_T3_room_type_share.csv')
    # Price missingness and scrape availability
    b=backbone[['transition','listing_id','snapshot_date','last_scraped','price']].copy()
    b['snapshot_date']=dates(b.snapshot_date);b['last_scraped']=dates(b.last_scraped)
    b['delay_days']=(b.last_scraped-b.snapshot_date).dt.total_seconds()/86400
    b['price_numeric']=pd.to_numeric(b.price,errors='coerce').astype(float)
    availability=b.groupby('transition',as_index=False).agg(n=('listing_id','size'),missing_scraped=('last_scraped',lambda x:int(x.isna().sum())),after_nominal=('delay_days',lambda x:int(x.gt(0).sum())),on_nominal=('delay_days',lambda x:int(x.eq(0).sum())),before_nominal=('delay_days',lambda x:int(x.lt(0).sum())),median_delay_days=('delay_days','median'),max_delay_days=('delay_days','max'),nonmissing_price=('price_numeric','count'),median_price=('price_numeric','median'))
    availability['pct_after_nominal']=100*availability.after_nominal/availability.n
    save(availability,'07_snapshot_scrape_availability.csv')
    pair=b.groupby(['transition','snapshot_date','last_scraped'],dropna=False).size().reset_index(name='n');save(pair,'08_snapshot_scrape_date_pairs.csv')
    n_after=int(b.delay_days.gt(0).sum())
    note('nominal origin vs last_scraped','REVIEW',f'{n_after}/{len(b)} rows scraped after nominal snapshot; availability cutoff requires provenance validation')
    # Temporal source files inventory, not pretending that filename implies leakage-free
    roots=[ROOT/'02_intermediate',ROOT/'05_results']
    keywords=('construction','rental','season','review','calendar','alignment','release','manifest','registry','modelling_panel')
    inventory=[]
    for folder in roots:
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.parquet','.csv','.json','.py') and any(k in p.name.lower() for k in keywords):
                inventory.append({'relative_path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'source_family':next((k for k in keywords if k in p.name.lower()),'other')})
    save(pd.DataFrame(inventory,columns=['relative_path','size_bytes','source_family']),'09_temporal_source_inventory.csv')
    note('construction release dates','NOT_VERIFIED','requires official publication records + effective as-of join per row')
    note('rental-market lag joins','NOT_VERIFIED','requires original source availability and join lineage')
    note('reviews and embeddings','NOT_VERIFIED','requires review timestamps and feature-generation lineage')
    note('calendar and seasonality','NOT_VERIFIED','requires as-of extraction and aggregation windows')
    # Modelling panel check without assuming every candidate feature is selected
    panel_path=ROOT/'02_intermediate/modelling/modelling_panel_T1_T10.parquet'
    if panel_path.exists():
        try:
            # schema-only read when pyarrow available; avoid loading massive 470-column panel
            import pyarrow.parquet as pq
            pf=pq.ParquetFile(panel_path);cols=pf.schema_arrow.names
            save(pd.DataFrame({'column':cols}),'10_model_panel_columns.csv')
            note('modelling panel schema','PASS' if pf.metadata.num_rows==len(outcome) else 'FAIL',f'rows={pf.metadata.num_rows}, columns={len(cols)}')
        except Exception as ex:
            note('modelling panel schema','NOT_VERIFIED',str(ex))
    else: note('modelling panel schema','NOT_VERIFIED','panel file not found')
    # Provenance hashes for inputs, may take seconds
    input_hashes=[]
    for p in [outcome_path,backbone_path,t11_path]:
        input_hashes.append({'relative_path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':signature(p)})
    save(pd.DataFrame(input_hashes),'11_input_sha256.csv')
except Exception as ex:
    note('audit execution','FAIL',f'{type(ex).__name__}: {ex}')
    (OUT/'ERROR_TRACEBACK.txt').write_text(traceback.format_exc(),encoding='utf-8')

save(pd.DataFrame(STATUS),'00_audit_status.csv')
counts={s:sum(x['status']==s for x in STATUS) for s in ('PASS','REVIEW','NOT_VERIFIED','FAIL')}
report=['# Post-freeze scientific audit — automatically generated','',f'Run: {datetime.now().isoformat()}',f'Project root: {ROOT}', '', '## Status counts',str(counts),'','## Tests']
report.extend(f"- **{x['status']}** {x['test']}: {x['details']}" for x in STATUS)
report += ['','## Interpretation guardrails','- Disappearance means absence of the same listing ID at the next observed snapshot, not business failure or permanent exit.','- T+2 reappearance has equal snapshot-step follow-up, but calendar-day intervals vary. T10 cannot be assessed without T12.','- `snapshot_date` is nominal; `last_scraped` is later for many listings. Never assert real-time availability at nominal date without verified provenance.','- Temporal source inventory is not proof of release-date correctness or absence of future leakage.','- No training, model selection, calibration, or threshold adjustment is performed.','- Do not declare submission readiness while temporal source audits are NOT_VERIFIED.']
rp=OUT/'AUDIT_REPORT.md';rp.write_text('\n'.join(report)+'\n',encoding='utf-8');FILES.append(rp)
zip_path=OUT/'POSTFREEZE_AUDIT_PACKAGE.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file() and p!=zip_path:z.write(p,arcname=p.name)
print('\n===== COMPLETE =====')
print('Audit folder:',OUT)
print('ZIP:',zip_path)
print('Statuses:',counts)
print('IMPORTANT: NOT_VERIFIED means not proven, not PASS.')
