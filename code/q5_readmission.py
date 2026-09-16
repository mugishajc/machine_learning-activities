"""
Q5 - hospital readmission: partitioning, and why a careless split flatters the model.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, StratifiedKFold,
                                     StratifiedGroupKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
RS=42; rng=np.random.default_rng(RS)

# ---- admissions, several per patient, with a patient-level frailty term ----
n_pat=1200
pat=pd.DataFrame({"patient_id":np.arange(n_pat),
  "age":rng.normal(58,17,n_pat).clip(18,98).round(),
  "sex":rng.choice(["F","M"],n_pat),
  "insurance":rng.choice(["mutuelle","private","none"],n_pat,p=[.62,.26,.12]),
  "comorbidity_index":rng.poisson(2.2,n_pat),
  "frailty":rng.normal(0,1.0,n_pat)})
# a chronic-risk score recorded once per patient: a noisy, observable read on frailty.
# It is patient-stable and near-unique, which is exactly the kind of field that lets a
# flexible model recognise an individual it has already been trained on.
pat["baseline_risk_score"]=(0.88*pat.frailty+0.34*rng.normal(0,1,n_pat)).round(4)
n_adm=rng.integers(1,6,n_pat)
rows=[]
for i,k in enumerate(n_adm):
    for j in range(k):
        rows.append({"patient_id":i,"admission_seq":j,
            "los_days":max(1,rng.gamma(2.0,2.4)),
            "n_procedures":rng.poisson(2.0),
            "emergency":rng.binomial(1,.42),
            "day":int(rng.integers(0,720))})
adm=pd.DataFrame(rows).merge(pat,on="patient_id")
# the uninsured subgroup follows a different relationship, deliberately
extra=np.where(adm.insurance=="none",0.55*adm.los_days/adm.los_days.mean(),0)
logit=(-2.1+1.35*adm.frailty+0.21*adm.comorbidity_index+0.095*adm.los_days
       +0.38*adm.emergency+0.016*(adm.age-58)+extra+rng.normal(0,.35,len(adm)))
adm["readmitted"]=rng.binomial(1,1/(1+np.exp(-logit)))
print("="*80); print("1. DATA")
print(f"  admissions={len(adm)}  patients={adm.patient_id.nunique()}  "
      f"admissions/patient mean={len(adm)/n_pat:.2f} max={adm.groupby('patient_id').size().max()}")
print(f"  readmission rate={adm.readmitted.mean():.3f}")
print(f"  patients with more than one admission: {(adm.groupby('patient_id').size()>1).sum()} "
      f"({(adm.groupby('patient_id').size()>1).mean():.0%})")

feat=["age","comorbidity_index","los_days","n_procedures","emergency","admission_seq",
      "baseline_risk_score"]
cats=["sex","insurance"]
X=adm[feat+cats]; y=adm.readmitted; g=adm.patient_id
pre=lambda: ColumnTransformer([("n",StandardScaler(),feat),
                               ("c",OneHotEncoder(drop="first",handle_unknown="ignore"),cats)])
# deliberately flexible: capacity is what turns repeated patients into memorised patients
mk=lambda: Pipeline([("pre",pre()),
    ("m",GradientBoostingClassifier(random_state=RS,n_estimators=400,max_depth=5,
                                    learning_rate=0.08,subsample=0.9))])

print("\n"+"="*80); print("2. Q5(b) THE SAME MODEL UNDER THREE SPLITS")
# (a) careless: random over ROWS, so one patient lands in both sides
tr,te=train_test_split(np.arange(len(X)),test_size=.25,random_state=RS,stratify=y)
m=mk().fit(X.iloc[tr],y.iloc[tr]); auc_row=roc_auc_score(y.iloc[te],m.predict_proba(X.iloc[te])[:,1])
shared=len(set(g.iloc[tr])&set(g.iloc[te]))
print(f"  random split over admissions   AUC={auc_row:.4f}   "
      f"patients appearing on BOTH sides: {shared}")
# (b) grouped by patient
gtr,gte=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=RS).split(X,y,groups=g))
m=mk().fit(X.iloc[gtr],y.iloc[gtr]); auc_grp=roc_auc_score(y.iloc[gte],m.predict_proba(X.iloc[gte])[:,1])
print(f"  grouped by patient             AUC={auc_grp:.4f}   "
      f"patients appearing on BOTH sides: {len(set(g.iloc[gtr])&set(g.iloc[gte]))}")
# (c) temporal
cut=adm.day.quantile(.75); ttr=adm.day<=cut; tte=~ttr
m=mk().fit(X[ttr],y[ttr]); auc_tmp=roc_auc_score(y[tte],m.predict_proba(X[tte])[:,1])
print(f"  temporal (train on first 75% of days) AUC={auc_tmp:.4f}   "
      f"patients on both sides: {len(set(g[ttr])&set(g[tte]))}")
print(f"\n  the careless split reports {auc_row-auc_grp:+.4f} AUC more than the grouped split "
      f"({(auc_row-auc_grp)/auc_grp*100:+.1f}%).")
print("  A single split is a noisy instrument, so this gap alone proves little. The repeated")
print("  experiment in q5b_leakage_scaling.py estimates the effect properly.")

print("\n"+"="*80); print("3. Q5(b) CROSS-VALIDATION, GROUPED VS NOT")
a=cross_val_score(mk(),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=RS),scoring="roc_auc")
b=cross_val_score(mk(),X,y,groups=g,cv=StratifiedGroupKFold(5,shuffle=True,random_state=RS),scoring="roc_auc")
print(f"  StratifiedKFold      AUC={a.mean():.4f} (+/-{a.std():.4f})   <- optimistic")
print(f"  StratifiedGroupKFold AUC={b.mean():.4f} (+/-{b.std():.4f})   <- honest")

print("\n"+"="*80); print("4. Q5(d) THE PARTITION ACTUALLY USED: train / validation / test")
gss=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=RS)
rest,test=next(gss.split(X,y,groups=g))
gss2=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=RS)   # 0.25 of 80% = 20%
tr2,val=next(gss2.split(X.iloc[rest],y.iloc[rest],groups=g.iloc[rest]))
tr_i,val_i=rest[tr2],rest[val]
parts={"train":tr_i,"validation":val_i,"test":test}
print(f"  {'subset':12s} {'admissions':>11s} {'patients':>9s} {'share':>7s} {'readmit rate':>13s}")
for k,ix in parts.items():
    print(f"  {k:12s} {len(ix):11d} {g.iloc[ix].nunique():9d} {len(ix)/len(X):6.1%} {y.iloc[ix].mean():13.3f}")
ov=[(a_,b_,len(set(g.iloc[parts[a_]])&set(g.iloc[parts[b_]])))
    for a_,b_ in [("train","validation"),("train","test"),("validation","test")]]
print("  patient overlap between subsets: "+", ".join(f"{a_}/{b_}={n}" for a_,b_,n in ov))
assert all(n==0 for *_,n in ov)

print("\n"+"="*80); print("5. Q5(c) PERFORMANCE BY PATIENT SUBGROUP")
m=mk().fit(X.iloc[tr_i],y.iloc[tr_i])
p=m.predict_proba(X.iloc[test])[:,1]
thr=np.quantile(p,0.75)
te_df=adm.iloc[test].copy(); te_df["p"]=p; te_df["pred"]=(p>=thr).astype(int)
print(f"  overall AUC={roc_auc_score(y.iloc[test],p):.4f}  "
      f"recall={recall_score(y.iloc[test],te_df.pred):.3f}  "
      f"precision={precision_score(y.iloc[test],te_df.pred):.3f}\n")
def by(col,bins=None,labels=None):
    s=te_df[col] if bins is None else pd.cut(te_df[col],bins,labels=labels)
    out=[]
    for k,gdf in te_df.groupby(s, observed=True):
        if gdf.readmitted.nunique()<2: continue
        out.append({"subgroup":f"{col}={k}","n":len(gdf),"base rate":gdf.readmitted.mean(),
            "AUC":roc_auc_score(gdf.readmitted,gdf.p),"recall":recall_score(gdf.readmitted,gdf.pred),
            "precision":precision_score(gdf.readmitted,gdf.pred,zero_division=0),
            "flag rate":gdf.pred.mean()})
    return pd.DataFrame(out)
sub=pd.concat([by("insurance"),by("sex"),
               by("age",[17,45,65,100],["18-45","46-65","66+"])],ignore_index=True)
print(sub.to_string(index=False,float_format=lambda v:f"{v:9.3f}"))
print(f"\n  widest AUC gap across subgroups: {sub.AUC.max()-sub.AUC.min():.4f} "
      f"({sub.loc[sub.AUC.idxmax(),'subgroup']} vs {sub.loc[sub.AUC.idxmin(),'subgroup']})")
print(f"  widest recall gap              : {sub.recall.max()-sub.recall.min():.4f} "
      f"({sub.loc[sub.recall.idxmax(),'subgroup']} vs {sub.loc[sub.recall.idxmin(),'subgroup']})")
print("  a single overall AUC conceals both gaps, which is why the aggregate number is not enough")
sub.to_csv("output/q5_subgroups.csv",index=False)

fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]
ax.bar(["random over\nadmissions","grouped by\npatient","temporal"],[auc_row,auc_grp,auc_tmp],
       color=["#b2182b","#2166ac","#4d9221"])
ax.set_ylim(0.5,max(auc_row,auc_grp,auc_tmp)+0.04); ax.set_ylabel("test AUC")
for i,v in enumerate([auc_row,auc_grp,auc_tmp]): ax.text(i,v+0.004,f"{v:.3f}",ha="center",fontsize=9)
ax.set_title("Q5 Fig 1a: careless partitioning inflates the score")
ax=axes[1]
s=sub.sort_values("AUC"); yp=np.arange(len(s))
ax.barh(yp,s.AUC,color="#4393c3"); ax.set_yticks(yp); ax.set_yticklabels(s.subgroup,fontsize=8)
ax.axvline(roc_auc_score(y.iloc[test],p),ls="--",c="k",lw=1.2,label="overall")
ax.set_xlim(0.5,1.0); ax.set_xlabel("AUC"); ax.legend()
ax.set_title("Q5 Fig 1b: performance is not uniform across subgroups")
fig.tight_layout(); fig.savefig("figures/q5_fig1.png",dpi=150); plt.close(fig)
print("\nfigure: figures/q5_fig1.png")
