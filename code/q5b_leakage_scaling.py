"""
Q5(b) supplement - how large is group leakage, and what governs it?
Single splits are noisy, so the comparison is repeated over seeds and swept
across the number of admissions recorded per patient.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

FEAT=["age","comorbidity_index","los_days","n_procedures","emergency","admission_seq","baseline_risk_score"]
CATS=["sex","insurance"]
def build(seed, adm_per_patient, n_pat=1200):
    rng=np.random.default_rng(seed)
    pat=pd.DataFrame({"patient_id":np.arange(n_pat),
      "age":rng.normal(58,17,n_pat).clip(18,98).round(),
      "sex":rng.choice(["F","M"],n_pat),
      "insurance":rng.choice(["mutuelle","private","none"],n_pat,p=[.62,.26,.12]),
      "comorbidity_index":rng.poisson(2.2,n_pat),
      "frailty":rng.normal(0,1.0,n_pat)})
    pat["baseline_risk_score"]=(0.88*pat.frailty+0.34*rng.normal(0,1,n_pat)).round(4)
    rows=[]
    for i in range(n_pat):
        for j in range(adm_per_patient):
            rows.append({"patient_id":i,"admission_seq":j,"los_days":max(1,rng.gamma(2.0,2.4)),
                         "n_procedures":rng.poisson(2.0),"emergency":rng.binomial(1,.42)})
    adm=pd.DataFrame(rows).merge(pat,on="patient_id")
    extra=np.where(adm.insurance=="none",0.55,0)
    logit=(-2.1+1.35*adm.frailty+0.21*adm.comorbidity_index+0.095*adm.los_days
           +0.38*adm.emergency+0.016*(adm.age-58)+extra+rng.normal(0,.35,len(adm)))
    adm["readmitted"]=rng.binomial(1,1/(1+np.exp(-logit)))
    return adm
def mk():
    return Pipeline([("pre",ColumnTransformer([("n",StandardScaler(),FEAT),
        ("c",OneHotEncoder(drop="first",handle_unknown="ignore"),CATS)])),
        ("m",GradientBoostingClassifier(random_state=0,n_estimators=400,max_depth=5,
                                        learning_rate=0.08,subsample=0.9))])

print("="*80); print("GROUP LEAKAGE VS RECORDS PER PATIENT (12 seeds each)")
print(f"  {'adm/patient':>11s} {'random AUC':>11s} {'grouped AUC':>12s} {'inflation':>10s} "
      f"{'95% CI':>18s} {'p':>9s}")
res=[]
for k in [1,2,4,8,16]:
    diffs=[];ra=[];ga=[]
    for s in range(12):
        adm=build(1000+s,k); X=adm[FEAT+CATS]; y=adm.readmitted; g=adm.patient_id
        tr,te=train_test_split(np.arange(len(X)),test_size=.25,random_state=s,stratify=y)
        a=roc_auc_score(y.iloc[te],mk().fit(X.iloc[tr],y.iloc[tr]).predict_proba(X.iloc[te])[:,1])
        gtr,gte=next(GroupShuffleSplit(1,test_size=.25,random_state=s).split(X,y,groups=g))
        b=roc_auc_score(y.iloc[gte],mk().fit(X.iloc[gtr],y.iloc[gtr]).predict_proba(X.iloc[gte])[:,1])
        diffs.append(a-b); ra.append(a); ga.append(b)
    d=np.array(diffs); se=d.std(ddof=1)/np.sqrt(len(d))
    ci=(d.mean()-1.96*se, d.mean()+1.96*se)
    t,p=stats.ttest_1samp(d,0)
    print(f"  {k:11d} {np.mean(ra):11.4f} {np.mean(ga):12.4f} {d.mean():+10.4f} "
          f"  [{ci[0]:+.4f},{ci[1]:+.4f}] {p:9.4f}")
    res.append({"adm_per_patient":k,"random_auc":np.mean(ra),"grouped_auc":np.mean(ga),
                "inflation":d.mean(),"ci_lo":ci[0],"ci_hi":ci[1],"p_value":p})
r=pd.DataFrame(res); r.to_csv("output/q5b_leakage_scaling.csv",index=False)
print("\n  With one admission per patient the two splits are identical by construction and the")
print("  difference is noise. Inflation appears once patients repeat and grows with the number")
print("  of records each patient contributes, because a larger share of the test set consists of")
print("  patients the model has already fitted.")

fig,ax=plt.subplots(figsize=(8,5))
ax.errorbar(r.adm_per_patient,r.inflation,yerr=[r.inflation-r.ci_lo,r.ci_hi-r.inflation],
            fmt="o-",capsize=4,lw=1.8,color="#b2182b")
ax.axhline(0,ls="--",c="k",lw=1)
ax.set_xscale("log",base=2); ax.set_xticks(r.adm_per_patient); ax.set_xticklabels(r.adm_per_patient)
ax.set_xlabel("admissions recorded per patient"); ax.set_ylabel("AUC inflation (random minus grouped)")
ax.set_title("Q5 Fig 2: optimism from a careless split grows with records per patient")
fig.tight_layout(); fig.savefig("figures/q5_fig2_leakage.png",dpi=150); plt.close(fig)
print("figure: figures/q5_fig2_leakage.png")
