"""
Q6 - US Crime dataset (47 states): EDA, correlation structure, feature selection.
Crime = offences per 100,000 population.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

RS = 42
LAB = {"M":"males 14-24 per 1000","So":"southern state (0/1)","Ed":"mean schooling (years x10)",
 "Po1":"police spend 1960","Po2":"police spend 1959","LF":"labour-force participation",
 "M.F":"males per 100 females","Pop":"population (100k)","NW":"non-white per 1000",
 "U1":"unemployment 14-24","U2":"unemployment 35-39","Wealth":"median assets/income",
 "Ineq":"income inequality","Prob":"probability of imprisonment","Time":"months served",
 "Crime":"offences per 100k"}
df = pd.read_csv("data/uscrime.txt", sep="\t")
y = df["Crime"]; X = df.drop(columns=["Crime"])

print("="*80); print("1. STRUCTURE")
print(f"observations (states): {len(df)}   variables: {df.shape[1]}   predictors: {X.shape[1]}")
print(f"missing: {df.isna().sum().sum()}   observations per predictor: {len(df)/X.shape[1]:.1f}")
print("\nDescriptive statistics:")
d = df.describe().T[["mean","std","min","50%","max"]]
d["skew"] = df.skew()
d.insert(0, "meaning", [LAB.get(i,"") for i in d.index])
print(d.to_string(float_format=lambda v: f"{v:10.3f}"))

print("\n"+"="*80); print("2. CORRELATION WITH THE TARGET")
c = X.corrwith(y).sort_values(key=abs, ascending=False)
ct = pd.DataFrame({"meaning":[LAB[i] for i in c.index], "r_with_Crime": c.values})
# significance of each simple correlation
from scipy import stats
ct["p_value"] = [stats.pearsonr(X[i], y)[1] for i in c.index]
print(ct.to_string(float_format=lambda v: f"{v:9.4f}"))

print("\n"+"="*80); print("3. REDUNDANCY AMONG PREDICTORS (|r| >= 0.60)")
cm = X.corr()
seen = set(); pair_rows=[]
for a in cm.columns:
    for b in cm.columns:
        if a!=b and (b,a) not in seen:
            seen.add((a,b))
            if abs(cm.loc[a,b])>=0.60:
                pair_rows.append({"var_1":a,"var_2":b,"r":cm.loc[a,b]})
pairs = pd.DataFrame(pair_rows).sort_values("r", key=abs, ascending=False)
print(pairs.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

print("\n"+"="*80); print("4. VARIANCE INFLATION FACTORS (all 15 predictors)")
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
Xc = sm.add_constant(Xs)
vif = pd.DataFrame({"predictor":X.columns,
    "VIF":[variance_inflation_factor(Xc.values,i) for i in range(1,Xc.shape[1])]}).sort_values("VIF",ascending=False)
vif["severity"] = np.where(vif.VIF>10,"severe",np.where(vif.VIF>5,"moderate","acceptable"))
print(vif.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

print("\n"+"="*80); print("5. WHY CORRELATION ALONE MISLEADS: Po1")
print(f"  simple correlation  Po1 ~ Crime            r = {np.corrcoef(X.Po1,y)[0,1]:.3f}")
m1 = sm.OLS(y, sm.add_constant(Xs[["Po1"]])).fit()
print(f"  univariate OLS      Crime ~ Po1            beta = {m1.params['Po1']:8.2f}  p = {m1.pvalues['Po1']:.4f}")
m2 = sm.OLS(y, sm.add_constant(Xs[["Po1","Ineq","Ed","Prob"]])).fit()
print(f"  controlled OLS      + Ineq, Ed, Prob       beta = {m2.params['Po1']:8.2f}  p = {m2.pvalues['Po1']:.4f}")
print(f"  partial r (Po1,Crime | Ineq,Ed,Prob)       = {np.sign(m2.params['Po1'])*abs(m2.tvalues['Po1'])/np.sqrt(m2.tvalues['Po1']**2+m2.df_resid):.3f}")
print("\n  Suppression: a predictor can look irrelevant alone yet matter once others are held constant")
for v in ["Ineq","M","U2"]:
    r = np.corrcoef(X[v], y)[0,1]
    mm = sm.OLS(y, sm.add_constant(Xs[[v,"Po1","Wealth"]])).fit()
    print(f"    {v:7s} simple r={r:6.3f} (p={stats.pearsonr(X[v],y)[1]:.3f})   "
          f"controlled beta={mm.params[v]:8.2f} (p={mm.pvalues[v]:.4f})")

print("\n"+"="*80); print("6. FEATURE SELECTION, THREE FAMILIES")
cvr = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RS)
def score(cols, label):
    p = Pipeline([("s",StandardScaler()),("m",LinearRegression())])
    s = -cross_val_score(p, X[cols], y, scoring="neg_root_mean_squared_error", cv=cvr)
    r2 = cross_val_score(p, X[cols], y, scoring="r2", cv=cvr)
    print(f"  {label:34s} k={len(cols):2d}  CV RMSE={s.mean():7.2f} (+/-{s.std():5.2f})  CV R2={r2.mean():6.3f}")
    return {"strategy":label,"k":len(cols),"cv_rmse":s.mean(),"cv_rmse_sd":s.std(),"cv_r2":r2.mean(),
            "features":", ".join(cols)}
rows=[score(list(X.columns), "All 15 predictors")]
filt = [c for c in X.columns if abs(np.corrcoef(X[c],y)[0,1])>=0.30]
rows.append(score(filt, "Filter: |r with Crime| >= 0.30"))
drop_red = [c for c in X.columns if c not in ("Po2","U1","Wealth")]
rows.append(score(drop_red, "Drop redundant (Po2,U1,Wealth)"))
las = LassoCV(cv=cvr, random_state=RS, max_iter=100000).fit(StandardScaler().fit_transform(X), y)
emb = list(X.columns[np.abs(las.coef_)>1e-8])
rows.append(score(emb, f"Embedded: LassoCV (a={las.alpha_:.2f})"))
rfe = RFECV(LinearRegression(), cv=cvr, scoring="neg_root_mean_squared_error", min_features_to_select=2)
rfe.fit(StandardScaler().fit_transform(X), y)
wrap = list(X.columns[rfe.support_])
rows.append(score(wrap, "Wrapper: RFE with CV"))
sel = pd.DataFrame(rows); sel.to_csv("output/q6_selection.csv", index=False)
print(f"\n  LassoCV retained : {', '.join(emb)}")
print(f"  RFE-CV retained  : {', '.join(wrap)}")

print("\n"+"="*80); print("7. PCA AS AN ALTERNATIVE TO DROPPING VARIABLES")
pc = PCA().fit(StandardScaler().fit_transform(X))
cum = np.cumsum(pc.explained_variance_ratio_)
print("  components for 80%/90%/95% of variance: "
      f"{np.argmax(cum>=.80)+1} / {np.argmax(cum>=.90)+1} / {np.argmax(cum>=.95)+1}  (of {X.shape[1]})")
print("  first 5 explained-variance ratios:", np.round(pc.explained_variance_ratio_[:5],3))
for k in (4,6,8):
    p = Pipeline([("s",StandardScaler()),("p",PCA(n_components=k)),("m",LinearRegression())])
    s = -cross_val_score(p, X, y, scoring="neg_root_mean_squared_error", cv=cvr)
    print(f"  PCR with {k} components: CV RMSE={s.mean():.2f}")

print("\n"+"="*80); print("8. RIDGE AS THE COLLINEARITY REMEDY (keeps all predictors)")
rc = RidgeCV(alphas=np.logspace(-2,4,80), cv=5).fit(StandardScaler().fit_transform(X), y)
p = Pipeline([("s",StandardScaler()),("m",RidgeCV(alphas=np.logspace(-2,4,80), cv=5))])
s = -cross_val_score(p, X, y, scoring="neg_root_mean_squared_error", cv=cvr)
print(f"  RidgeCV alpha={rc.alpha_:.2f}  CV RMSE={s.mean():.2f}")
cf = pd.DataFrame({"predictor":X.columns,"OLS":sm.OLS(y,Xc).fit().params[1:].values,"Ridge":rc.coef_})
cf["shrunk %"] = (1-np.abs(cf.Ridge)/np.abs(cf.OLS))*100
print(cf.sort_values("OLS",key=abs,ascending=False).to_string(index=False,float_format=lambda v:f"{v:9.2f}"))
cf.to_csv("output/q6_coefficients.csv", index=False)

# ---- figures ----
fig, ax = plt.subplots(figsize=(10,8.5))
full = df.corr()
im = ax.imshow(full, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(full))); ax.set_xticklabels(full.columns, rotation=90)
ax.set_yticks(range(len(full))); ax.set_yticklabels(full.columns)
for i in range(len(full)):
    for j in range(len(full)):
        v=full.iloc[i,j]
        ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.5,
                color="white" if abs(v)>0.6 else "black")
ax.set_title("Q6 Fig 1: Correlation matrix, US Crime dataset (47 states)")
fig.colorbar(im,shrink=0.8); fig.tight_layout(); fig.savefig("figures/q6_fig1_corr.png",dpi=150); plt.close(fig)

fig, axes = plt.subplots(2,3,figsize=(15,8.5))
for axx,v in zip(axes.ravel(), c.index[:6]):
    axx.scatter(X[v], y, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
    b,a = np.polyfit(X[v],y,1); xs=np.linspace(X[v].min(),X[v].max(),50)
    axx.plot(xs,a+b*xs,"r-",lw=1.3)
    axx.set_xlabel(f"{v} ({LAB[v]})",fontsize=8); axx.set_ylabel("Crime per 100k",fontsize=8)
    axx.set_title(f"r = {np.corrcoef(X[v],y)[0,1]:.3f}",fontsize=9)
fig.suptitle("Q6 Fig 2: Six strongest bivariate relationships with Crime",y=1.0)
fig.tight_layout(); fig.savefig("figures/q6_fig2_scatter.png",dpi=150,bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(13,5))
ax=axes[0]
ax.scatter(X.Po1,X.Po2,s=30,edgecolor="k",linewidth=0.3)
ax.set_xlabel("Po1 (police spend 1960)"); ax.set_ylabel("Po2 (police spend 1959)")
ax.set_title(f"Q6 Fig 3a: near-duplicate predictors, r = {np.corrcoef(X.Po1,X.Po2)[0,1]:.3f}")
ax=axes[1]
vs=vif.sort_values("VIF")
ax.barh(vs.predictor, vs.VIF, color=np.where(vs.VIF>10,"#b2182b",np.where(vs.VIF>5,"#ef8a62","#67a9cf")))
ax.axvline(5,ls=":",c="k",lw=1); ax.axvline(10,ls="--",c="k",lw=1)
ax.set_xlabel("VIF"); ax.set_title("Q6 Fig 3b: variance inflation factors")
fig.tight_layout(); fig.savefig("figures/q6_fig3_collinearity.png",dpi=150); plt.close(fig)
print("\nFigures: figures/q6_fig1..3")
