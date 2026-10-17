"""One-run publication figures. Uses frozen J/K/L-O artifacts; no retraining.
Run: %run "C:/Users/yxt2919/7th_NZ_STR_Construction_AI/04_code/generate_manuscript_figures.py"
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import PercentFormatter

ROOT = Path(r'C:\Users\yxt2919\7th_NZ_STR_Construction_AI')
if not ROOT.exists():
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'05_results').exists()), ROOT)
if not ROOT.exists(): raise FileNotFoundError('Set ROOT to your project directory')
BASE=ROOT/'05_results'/'manuscript_final_L_to_O'
OUT=BASE/'figures'/'journal_ready'
OUT.mkdir(parents=True,exist_ok=True)
# Windows-standard serif: Times New Roman. Visually compatible with a serif LaTeX paper,
# but NOT identical to Latin Modern used by \usepackage{lmodern}.
# Matplotlib may cache installed fonts; refresh with fm.fontManager.addfont if needed.
FONTS={f.name for f in fm.fontManager.ttflist}
if 'Times New Roman' in FONTS:
    FONT='Times New Roman'
else:
    from pathlib import Path as _Path
    win_fonts = _Path('C:/Windows/Fonts')
    for filename in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        fp = win_fonts / filename
        if fp.exists():
            fm.fontManager.addfont(str(fp))
    FONTS={f.name for f in fm.fontManager.ttflist}
    FONT='Times New Roman' if 'Times New Roman' in FONTS else 'DejaVu Serif'
    if FONT != 'Times New Roman':
        print('WARNING: Times New Roman unavailable; using DejaVu Serif fallback.')
print('FIGURE FONT:', FONT)
print('NOTE: Overleaf lmodern uses Latin Modern; Times New Roman is a practical Windows substitute, not an exact match.')
plt.rcParams.update({'font.family':FONT,'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
 'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':600,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'xtick.major.width':.7,
 'ytick.major.width':.7,'legend.frameon':False})
INK='#213047'; BLUE='#315A88'; TEAL='#008B88'; PINK='#B91469'; GRAY='#A3AAB3'
def source(patterns):
    for pattern in patterns:
        matches=sorted(BASE.rglob(pattern))
        if matches:return matches[0]
    raise FileNotFoundError('No file matching '+str(patterns)+' under '+str(BASE))
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('SAVED:',name)
def pick(df,*alternatives):
    for name in alternatives:
        if name in df.columns:return name
    raise KeyError('Expected one of '+str(alternatives)+'; available: '+str(df.columns.tolist()))

# Figure A: risk deciles, from the exact T10 decile table.
risk=pd.read_csv(source(['N_primary_prospective_risk_deciles.csv','*risk*decile*.csv','*decile*.csv']))
d=pick(risk,'risk_decile','decile')
obs=pick(risk,'observed_disappearance_rate','observed_rate')
pred=pick(risk,'mean_predicted_disappearance','mean_predicted_risk','mean_predicted_probability')
risk=risk.sort_values(d)
fig,ax=plt.subplots(figsize=(6.6,3.5),layout='constrained')
ax.plot(risk[d],risk[obs]*100,'o-',color=INK,lw=1.8,ms=4,label='Observed disappearance')
ax.plot(risk[d],risk[pred]*100,'s--',color=TEAL,lw=1.5,ms=3.5,label='Mean predicted risk (raw)')
ax.set(xlabel='Predicted-risk decile (lowest to highest)',ylabel='Disappearance rate (%)',xticks=range(1,11))
ax.grid(axis='y',alpha=.18)
ax.legend(loc='upper left',fontsize=8)
save(fig,'F1_T10_risk_deciles')

# Figure B: TA cluster bootstrap, point and 95% intervals, metric-specific facets.
ci=pd.read_csv(source(['L_FINAL_95CI.csv','*bootstrap*summary*.csv','*confidence*interval*.csv','*bootstrap*ci*.csv','*uncertainty*.csv']))
outcol=pick(ci,'outcome'); metcol=pick(ci,'metric'); estcol=pick(ci,'estimate')
locol=pick(ci,'ci95_low'); hicol=pick(ci,'ci95_high')
items=[('L_primary_raw','roc_auc','Primary ROC-AUC'),('L_primary_raw','pr_auc','Primary PR-AUC'),('L_primary_raw','brier','Primary Brier'),('L_secondary','mae','Secondary MAE'),('L_secondary','rmse','Secondary RMSE'),('L_secondary','r2','Secondary R²')]
fig,axes=plt.subplots(2,3,figsize=(9.2,4.5),layout='constrained')
for ax,(o,m,label) in zip(axes.flat,items):
    sel=ci[(ci[outcol].astype(str)==o)&(ci[metcol].astype(str)==m)]
    if len(sel)!=1:raise ValueError('Expected exactly one CI row for '+o+' / '+m)
    row=sel.iloc[0]; est=float(row[estcol]); low=float(row[locol]); high=float(row[hicol])
    ax.errorbar(est,0,xerr=[[est-low],[high-est]],fmt='o',color=BLUE,capsize=4,lw=1.6)
    pad=max(high-low,abs(est)*.05,.001)*.65
    ax.set(xlim=(low-pad,high+pad),ylim=(-.6,.6),yticks=[],title=label)
    ax.axvline(est,color=GRAY,ls=':',lw=.8)
    ax.text(.5,.08,f'{est:.4f} [{low:.4f}, {high:.4f}]',transform=ax.transAxes,ha='center',fontsize=7)
    ax.grid(axis='x',alpha=.15)
save(fig,'F2_T10_TA_bootstrap_CIs')

# Figure C: SHAP importance; exact contributions are on the log-odds scale.
shap=pd.read_csv(source(['M_SHAP_53_frozen_features.csv','*shap*importance*.csv','*SHAP*importance*.csv','*feature*importance*.csv']))
fcol=pick(shap,'feature','predictor'); vcol=pick(shap,'mean_abs_shap_log_odds','mean_abs_shap','importance')
shap=shap.nlargest(15,vcol).sort_values(vcol)
fig,ax=plt.subplots(figsize=(7.2,5.0),layout='constrained')
ax.barh(shap[fcol],shap[vcol],color=BLUE,height=.68)
ax.set(xlabel='Mean absolute SHAP contribution (log-odds)',ylabel='')
ax.grid(axis='x',alpha=.14)
save(fig,'F3_T10_SHAP_top15')

# Figure D: Secondary T10 observed distribution. No fabricated predictions or Spearman.
secondary_candidates=list((ROOT/'05_results'/'secondary_price_resilience_K').rglob('*T10*predictions*.parquet'))
if not secondary_candidates:
    print('SKIPPED F4: secondary listing-level T10 predictions parquet not found')
else:
    sec=pd.read_parquet(secondary_candidates[0])
    print('Secondary prediction columns:',sec.columns.tolist())
    possible=['y_true','y_actual','price_resilience_delta','actual_delta','y','observed_price_resilience_delta']
    actual=next((c for c in possible if c in sec.columns),None)
    if actual is None:
        print('SKIPPED F4: target column ambiguous; set actual from printed columns, do not guess')
    else:
        y=pd.to_numeric(sec[actual],errors='coerce').dropna()
        fig,ax=plt.subplots(figsize=(6.6,3.4),layout='constrained')
        ax.hist(y,bins=70,color=BLUE,alpha=.83)
        ax.axvline(0,color=PINK,lw=1.6,ls='--',label='Frozen zero-change prediction')
        ax.set(xlabel='Observed change in advertised-price peer percentile',ylabel='Listing count')
        ax.legend(fontsize=8)
        save(fig,'F4_T10_secondary_observed_change')

print('DONE. Figures:',OUT)
print('Font:',FONT,'— Overleaf source uses \\usepackage{lmodern}.')
