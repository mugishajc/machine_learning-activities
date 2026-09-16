"""
Q4 - subscription renewal: encoding and scaling choices, measured rather than asserted.
"""
import numpy as np, pandas as pd, joblib, warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
RS=42; rng=np.random.default_rng(RS); N=4000

tiers=["basic","standard","premium"]; tier_v=dict(zip(tiers,[0,1,2]))
regions=["Kigali","Northern","Southern","Eastern","Western"]
region_eff=dict(zip(regions,[0.9,-0.3,-0.5,0.2,-0.4]))          # NOT monotone in list order
pays=["mobile_money","card","bank_transfer","cash"]
pay_eff=dict(zip(pays,[0.5,0.7,0.1,-0.9]))
d=pd.DataFrame({
 "age":rng.normal(36,11,N).clip(18,80).round(),
 "monthly_spend":rng.lognormal(9.6,0.6,N).round(),
 "n_prev_transactions":rng.poisson(9,N),
 "account_duration_days":rng.gamma(2.6,150,N).round(),
 "subscription_type":rng.choice(tiers,N,p=[.45,.35,.20]),
 "region":rng.choice(regions,N,p=[.40,.15,.15,.15,.15]),
 "payment_method":rng.choice(pays,N,p=[.45,.25,.20,.10]),
 "customer_status":rng.choice(["active","dormant","suspended"],N,p=[.7,.22,.08])})
logit=(-1.4+0.55*d.subscription_type.map(tier_v)+d.region.map(region_eff)+d.payment_method.map(pay_eff)
       +0.0016*d.account_duration_days+0.045*d.n_prev_transactions
       -0.9*(d.customer_status=="suspended")+rng.normal(0,.55,N))
d["renewed"]=rng.binomial(1,1/(1+np.exp(-logit)))
y=d.pop("renewed")
num=["age","monthly_spend","n_prev_transactions","account_duration_days"]
nominal=["region","payment_method","customer_status"]; ordinal=["subscription_type"]
print("="*80); print("DATA"); print(f"  n={N}  renewal rate={y.mean():.3f}")
print(f"  cardinality: "+", ".join(f"{c}={d[c].nunique()}" for c in nominal+ordinal))

cv=StratifiedKFold(5,shuffle=True,random_state=RS)
def run(name,pre,model):
    p=Pipeline([("pre",pre),("m",model)])
    s=cross_val_score(p,d,y,cv=cv,scoring="roc_auc")
    p.fit(d,y); w=p.named_steps["pre"].transform(d).shape[1]
    print(f"  {name:48s} AUC={s.mean():.4f} (+/-{s.std():.4f})  width={w}")
    return s.mean()

print("\n"+"="*80); print("Q4(b) ENCODING STRATEGIES, LOGISTIC REGRESSION")
A=run("One-hot nominal + ordinal tier (correct)",
    ColumnTransformer([("n",StandardScaler(),num),
        ("o",OrdinalEncoder(categories=[tiers]),ordinal),
        ("c",OneHotEncoder(drop="first",handle_unknown="ignore"),nominal)]),
    LogisticRegression(max_iter=2000,random_state=RS))
B=run("Ordinal codes forced onto NOMINAL region/payment",
    ColumnTransformer([("n",StandardScaler(),num),
        ("o",OrdinalEncoder(),ordinal+nominal)]),
    LogisticRegression(max_iter=2000,random_state=RS))
C=run("One-hot everything incl. tier (order discarded)",
    ColumnTransformer([("n",StandardScaler(),num),
        ("c",OneHotEncoder(drop="first",handle_unknown="ignore"),nominal+ordinal)]),
    LogisticRegression(max_iter=2000,random_state=RS))
D=run("Correct encoding but NO scaling",
    ColumnTransformer([("n","passthrough",num),
        ("o",OrdinalEncoder(categories=[tiers]),ordinal),
        ("c",OneHotEncoder(drop="first",handle_unknown="ignore"),nominal)]),
    LogisticRegression(max_iter=2000,random_state=RS))
print(f"\n  cost of mis-encoding nominal variables as ordinal: {A-B:+.4f} AUC")
print(f"  cost of discarding the tier ordering            : {A-C:+.4f} AUC")

print("\n"+"="*80); print("Q4(b) THE SAME CHOICES UNDER A TREE MODEL")
rf=lambda: RandomForestClassifier(n_estimators=250,random_state=RS,n_jobs=-1,min_samples_leaf=5)
E=run("Ordinal codes on nominal, random forest",
    ColumnTransformer([("n","passthrough",num),("o",OrdinalEncoder(),ordinal+nominal)]),rf())
F=run("One-hot nominal, random forest",
    ColumnTransformer([("n","passthrough",num),("o",OrdinalEncoder(categories=[tiers]),ordinal),
        ("c",OneHotEncoder(handle_unknown="ignore"),nominal)]),rf())
print(f"\n  a tree splits on thresholds and can isolate any integer code, so the penalty is"
      f" {abs(E-F):.4f} AUC, far smaller than the {abs(A-B):.4f} seen in the linear model")

print("\n"+"="*80); print("Q4(b) WHY ARBITRARY INTEGER CODES MISLEAD A LINEAR MODEL")
codes=pd.DataFrame({"region":regions,"alphabetical code":range(len(regions)),
                    "true effect on renewal":[region_eff[r] for r in regions]})
print(codes.to_string(index=False))
r=np.corrcoef(codes["alphabetical code"],codes["true effect on renewal"])[0,1]
print(f"  correlation between the integer code and the real effect: {r:+.3f}")
print("  the code imposes Kigali < Northern < Southern < Eastern < Western and equal spacing;")
print("  neither holds, so one coefficient is fitted to a sequence that carries no meaning")

print("\n"+"="*80); print("Q4(d) DEPLOYMENT: PERSIST, DO NOT REFIT")
pre=ColumnTransformer([("n",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
    ("o",Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                   ("e",OrdinalEncoder(categories=[tiers],handle_unknown="use_encoded_value",unknown_value=-1))]),ordinal),
    ("c",Pipeline([("i",SimpleImputer(strategy="constant",fill_value="not_reported")),
                   ("e",OneHotEncoder(handle_unknown="ignore"))]),nominal)])
full=Pipeline([("pre",pre),("m",LogisticRegression(max_iter=2000,random_state=RS))]).fit(d,y)
joblib.dump(full,"output/q4_pipeline.joblib")
loaded=joblib.load("output/q4_pipeline.joblib")
batch=d.sample(5,random_state=1)
print(f"  saved artefact reproduces training-time output exactly: "
      f"{np.allclose(full.predict_proba(batch)[:,1], loaded.predict_proba(batch)[:,1])}")
print(f"  training feature width frozen at {full.named_steps['pre'].transform(batch).shape[1]} columns")

nov=batch.copy(); nov.loc[:,"region"]="Musanze"; nov.loc[:,"payment_method"]="crypto"
print(f"  an unseen region and payment method still score without error: "
      f"{np.round(loaded.predict_proba(nov)[:,1],4)}")
refit=Pipeline([("pre",ColumnTransformer([("n",StandardScaler(),num),
    ("o",OrdinalEncoder(categories=[tiers]),ordinal),
    ("c",OneHotEncoder(drop="first",handle_unknown="ignore"),nominal)])),
    ("m",LogisticRegression(max_iter=2000,random_state=RS))]).fit(d.sample(400,random_state=7),
                                                                 y.sample(400,random_state=7))
delta=np.abs(refit.predict_proba(batch)[:,1]-full.predict_proba(batch)[:,1])
print(f"  refitting the transformer on a later batch instead of loading it shifts the same five"
      f" customers by up to {delta.max():.4f} in predicted probability (train/serve skew)")

print("\n"+"="*80); print("Q4(d) THE REFERENCE-LEVEL TRAP")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    enc=OneHotEncoder(drop="first",handle_unknown="ignore").fit(d[["region"]])
    ref=enc.transform(pd.DataFrame({"region":["Eastern"]})).toarray()
    unk=enc.transform(pd.DataFrame({"region":["Musanze"]})).toarray()
print(f"  reference level 'Eastern' -> {ref.astype(int).tolist()}")
print(f"  unseen level 'Musanze'    -> {unk.astype(int).tolist()}")
print("  identical vectors: an unrecognised region is silently scored as the reference region.")
print("  drop='first' and handle_unknown='ignore' must not be combined in production; keep all")
print("  levels, or add an explicit 'unknown' category, so the two cases stay distinguishable.")
pd.DataFrame({"strategy":["one-hot nominal + ordinal tier","ordinal on nominal","one-hot all","no scaling",
                          "RF ordinal on nominal","RF one-hot"],
              "cv_auc":[A,B,C,D,E,F]}).to_csv("output/q4_encoding.csv",index=False)
