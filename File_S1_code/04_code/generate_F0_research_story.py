# -*- coding: utf-8 -*-
"""Publication vector figures F0A-F0C; factual design and frozen development metrics.
No synthetic observations, no model training, no internet. Display + save automatically.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager
from IPython.display import display

ROOT=Path(r'C:\Users\yxt2919\7th_NZ_STR_Construction_AI')
if not ROOT.exists():
    ROOT=Path(__file__).resolve().parent
OUT=ROOT/'05_results'/'manuscript_final_L_to_O'/'figures'/'journal_ready'
OUT.mkdir(parents=True,exist_ok=True)
FONT='Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
plt.rcParams.update({'font.family':FONT,'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
NAVY='#17324d'; TEAL='#168a89'; PINK='#bd4777'; GREY='#657487'; PALE='#edf3f7'; LIGHT='#dce9ef'

def box(ax,x,y,w,h,title,sub='',color=NAVY,fc='white',fs=10):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.014',
        facecolor=fc,edgecolor=color,linewidth=1.25)
    ax.add_patch(p)
    ax.text(x+w/2,y+h*(.65 if sub else .50),title,ha='center',va='center',color=color,
            fontsize=fs,fontweight='bold',linespacing=1.15)
    if sub: ax.text(x+w/2,y+h*.28,sub,ha='center',va='center',fontsize=8,color=GREY,linespacing=1.2)

def arrow(ax,x1,y1,x2,y2,color=GREY,lw=1.4):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,
        lw=lw,color=color,connectionstyle='arc3'))

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.13)
    fig.savefig(OUT/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.13)
    print('SAVED + DISPLAYED:',name,'| PDF vector + PNG 600 dpi')
    display(fig)
    plt.close(fig)

# F0A — logical joins are methodological; does not claim unverified exact key names
fig,ax=plt.subplots(figsize=(13.5,7.4)); ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.5,.965,'Multi-source information integration and prediction design',ha='center',va='top',fontsize=16,fontweight='bold',color=NAVY)
ax.text(.5,.922,'Sources and temporal alignment at listing-origin; outcome labels use the subsequent snapshot',ha='center',fontsize=10,color=GREY)
source=[('Platform listings','Listing attributes and price'),('Review histories','365-day history; text features'),('Calendar snapshots','First 28 observed days; 4 weeks'),('Construction pipeline','TA and temporal context'),('Residential rental market','TA and temporal context'),('Tourism seasonality','Historical ADP context')]
ys=[.79,.665,.54,.415,.29,.165]
for (title,sub),y in zip(source,ys):
    box(ax,.025,y,.25,.096,title,sub,TEAL if y<.5 else NAVY,fs=10)
    arrow(ax,.275,y+.048,.385,.52)
box(ax,.39,.42,.235,.20,'Listing-origin feature panel','T1–T10 | 491,140 rows\n437 candidate predictors',NAVY,PALE,fs=12)
arrow(ax,.625,.52,.715,.52)
box(ax,.72,.43,.25,.18,'Leakage-controlled modelling','Origin-side features only\nTemporal train/test separation',NAVY,PALE,fs=11)
arrow(ax,.845,.43,.845,.355)
box(ax,.67,.19,.305,.13,'Primary: platform disappearance','T1–T9 train; T10 locked test\nXGBoost A5: 53 features',TEAL,fs=10)
box(ax,.67,.025,.305,.13,'Secondary: advertised-price rank change','Eligible T1, T6–T10\nZero-change baseline selected',PINK,fs=10)
arrow(ax,.845,.19,.845,.155)
ax.text(.025,.045,'Join logic: listing-specific history + territorial-area/period context;\nfuture-snapshot outcomes are not predictors.',fontsize=9,color=GREY,va='bottom')
fig.tight_layout();save(fig,'F0A_multisource_integration_framework')

# F0B — exact transition chronology from the frozen design
fig,ax=plt.subplots(figsize=(13.8,7.4));ax.set_xlim(-.8,10.65);ax.set_ylim(-1.2,8.3)
ax.set_yticks([]);ax.set_xticks(range(10),[f'T{i}' for i in range(1,11)]);ax.tick_params(axis='x',length=0,labelsize=10)
for x in range(10):ax.axvline(x,color='#e8eef2',lw=.9,zorder=0)
ax.text(4.5,8.0,'Rolling-origin development and locked final evaluation',ha='center',fontsize=15,fontweight='bold',color=NAVY)
ax.text(4.5,7.55,'Each transition predicts the next observed platform snapshot',ha='center',fontsize=10,color=GREY)
folds=[(3,'T4'),(4,'T5'),(5,'T6'),(6,'T7'),(7,'T8'),(8,'T9')]
for row,(end,name) in enumerate(folds):
    y=6.65-row*.82
    ax.add_patch(Rectangle((-.4,y-.18),end-.1,.36,color=LIGHT,ec='none'))
    ax.add_patch(Rectangle((end-.4,y-.18),.8,.36,color=TEAL,ec='none'))
    ax.text(-.72,y,name,ha='right',va='center',fontsize=9,color=NAVY,fontweight='bold')
    ax.text(end,y,'VALIDATE',ha='center',va='center',fontsize=7,color='white',fontweight='bold')
ax.add_patch(Rectangle((-.4,.95),8.8,.42,color=LIGHT,ec='none'))
ax.add_patch(Rectangle((8.6,.95),.8,.42,color=PINK,ec='none'))
ax.text(-.72,1.16,'Final primary',ha='right',va='center',fontsize=9,color=NAVY,fontweight='bold')
ax.text(9,1.16,'T10',ha='center',va='center',fontsize=9,color='white',fontweight='bold')
ax.text(4,1.16,'Train on T1–T9',ha='center',va='center',fontsize=10,color=NAVY)
ax.text(-.72,.28,'Secondary labels',ha='right',va='center',fontsize=9,color=NAVY,fontweight='bold')
for x in [0,5,6,7,8,9]:
    ax.add_patch(Rectangle((x-.37,.10),.74,.36,color=TEAL if x!=9 else PINK,ec='none'))
for x in [1,2,3,4]:
    ax.add_patch(Rectangle((x-.37,.10),.74,.36,facecolor='white',edgecolor=GREY,hatch='///',lw=.5))
ax.text(4.5,-.48,'Secondary: T2–T5 unavailable due to missing eligible listing-price labels',ha='center',fontsize=9,color=GREY)
ax.text(4.5,-.92,'Blue-grey = training | teal = development/eligible | pink = locked T10 | hatched = unavailable',ha='center',fontsize=9,color=GREY)
for s in ax.spines.values():s.set_visible(False)
fig.tight_layout();save(fig,'F0B_rolling_origin_and_label_coverage')

# F0C — frozen I-B macro metrics; all numbers from recorded development results
models=['XGB A5: structured full','XGB A4: + calendar','XGB A10: no reviews','XGB A6: full multimodal','MLP: structured','MLP: full multimodal']
roc=[.738356,.7317,.7297,.7228,.6786,.6693]
pr=[.074924,.0720,.0704,.0678,.0561,.0519]
brier=[.0292427,.02753,.02768,.02933,.03933,.02714]
fig,axes=plt.subplots(1,3,figsize=(14.3,5.3),sharey=True)
for ax,vals,label,lower in zip(axes,[roc,pr,brier],['ROC-AUC ↑','PR-AUC ↑','Brier ↓'],[False,False,True]):
    y=np.arange(len(models));ax.scatter(vals,y,c=[PINK]+[NAVY]*5,s=[90]+[50]*5,zorder=3)
    ax.set_xlabel(label,fontsize=11);ax.grid(axis='x',alpha=.18);ax.invert_yaxis()
    for i,v in enumerate(vals):ax.annotate(f'{v:.4f}',(v,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8,color=NAVY)
    lo,hi=min(vals),max(vals);d=(hi-lo)*.33;ax.set_xlim(lo-d*.4,hi+d*1.3)
    if lower:ax.axvline(min(vals),color=GREY,lw=.7,ls='--',alpha=.6)
axes[0].set_yticks(np.arange(len(models)),models)
axes[0].tick_params(axis='y',labelsize=10)
fig.suptitle('Development model comparison: rolling-origin T4–T9',fontsize=15,color=NAVY,fontweight='bold',y=1.01)
fig.text(.5,-.02,'Macro-average across development windows; metrics reflect different objectives. Selection followed frozen development protocol.',ha='center',fontsize=9,color=GREY)
fig.tight_layout(w_pad=2.0);save(fig,'F0C_development_model_comparison')
print('\nALL DONE —',OUT,'\nFont:',FONT)
