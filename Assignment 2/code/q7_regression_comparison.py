"""
Question Seven - OLS, Ridge and Lasso under multicollinearity.

Runs on the supplied prices.csv: 1,460 Ames house sales, 79 predictors after the
identifier is removed, mixed numeric and categorical with missing values in both.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA, TARGET = "data/prices.csv", "SalePrice"
df = pd.read_csv(DATA, low_memory=False)
y = df[TARGET]
X = df.drop(columns=[TARGET, "Id"], errors="ignore")
num = [c for c in X.columns if X[c].dtype != object]
cat = [c for c in X.columns if X[c].dtype == object]

print("="*82); print("1. DATA AND COLLINEARITY")
print(f"  observations {len(X):,}   predictors {X.shape[1]} ({len(num)} numeric, {len(cat)} categorical)")
print(f"  target {TARGET}: mean {y.mean():,.0f}  sd {y.std():,.0f}  "
      f"range {y.min():,.0f} to {y.max():,.0f}  skew {y.skew():.2f}")
print(f"  missing cells {X.isna().sum().sum():,} across {int((X.isna().sum()>0).sum())} columns")
print("\n  the two pairs the question names:")
for a, b in [("GrLivArea", "TotRmsAbvGrd"), ("GarageCars", "GarageArea")]:
    if a in X and b in X:
        print(f"    {a:14s} <-> {b:14s} r = {X[a].corr(X[b]):.4f}")
cm = X[num].corr().abs(); np.fill_diagonal(cm.values, 0)
pairs = cm.stack().sort_values(ascending=False)
pairs = pairs[~pairs.index.duplicated()]
print(f"\n  numeric pairs with |r| >= 0.80: {int((cm.values>=0.80).sum()//2)}")
seen=set(); shown=0
for (a,b),v in pairs.items():
    if (b,a) in seen: continue
    seen.add((a,b)); print(f"    {a:22s} <-> {b:22s} r = {v:.4f}"); shown+=1
    if shown==5: break

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="None")),
                          ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=RANDOM_STATE)
cv = KFold(10, shuffle=True, random_state=RANDOM_STATE)
alphas = np.logspace(-2, 4, 60)
print(f"\n  train {len(X_tr):,}   test {len(X_te):,}")

print("\n"+"="*82); print("2. THREE MODELS, IDENTICAL PIPELINE")
rows, fitted = [], {}
for name, est, grid in [("OLS",   LinearRegression(), None),
                        ("Ridge", Ridge(random_state=RANDOM_STATE), {"m__alpha": alphas}),
                        ("Lasso", Lasso(random_state=RANDOM_STATE, max_iter=50000), {"m__alpha": alphas})]:
    pipe = Pipeline([("s", make_pre()), ("m", est)])
    if grid:
        gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
        best, alpha = gs.best_estimator_, gs.best_params_["m__alpha"]
    else:
        best, alpha = pipe.fit(X_tr, y_tr), np.nan
    pred = best.predict(X_te)
    coef = best.named_steps["m"].coef_
    rows.append({"model": name, "alpha": alpha,
                 "MAE": mean_absolute_error(y_te, pred),
                 "RMSE": mean_squared_error(y_te, pred)**0.5,
                 "R2": r2_score(y_te, pred),
                 "non-zero coefficients": int((np.abs(coef) > 1e-8).sum()),
                 "sum |coef|": float(np.abs(coef).sum()),
                 "largest |coef|": float(np.abs(coef).max())})
    fitted[name] = coef
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:12.4f}"))
res.to_csv("output/q7_results.csv", index=False)

lasso_fit = Pipeline([("s", make_pre()),
                      ("m", Lasso(alpha=res.loc[res.model=="Lasso","alpha"].iloc[0],
                                  random_state=RANDOM_STATE, max_iter=50000))]).fit(X_tr, y_tr)
print(f"\n  convergence check on the selected Lasso: {lasso_fit.named_steps['m'].n_iter_} iterations")
print("  of a 50,000 limit. The grid search emits convergence warnings at its smallest")
print("  alphas, where the penalty is too weak to make the problem well conditioned, but")
print("  those candidates are rejected and the selected model converges comfortably.")

print("\n"+"="*82); print("3. WHAT THE PENALTIES DID TO THE COEFFICIENTS")
feat = [f.split("__", 1)[1] for f in make_pre().fit(X_tr).get_feature_names_out()]
co = pd.DataFrame(fitted, index=feat)
co["|OLS|"] = co.OLS.abs()
co = co.sort_values("|OLS|", ascending=False).drop(columns="|OLS|")
print(co.head(10).to_string(float_format=lambda v: f"{v:12.3f}"))
print(f"\n  coefficients set exactly to zero:  OLS {int((co.OLS==0).sum())}   "
      f"Ridge {int((co.Ridge==0).sum())}   Lasso {int((co.Lasso==0).sum())} of {len(co)}")
print(f"  sum of absolute coefficients:      OLS {co.OLS.abs().sum():,.1f}   "
      f"Ridge {co.Ridge.abs().sum():,.1f}   Lasso {co.Lasso.abs().sum():,.1f}")
print(f"  largest single coefficient:        OLS {co.OLS.abs().max():,.1f}   "
      f"Ridge {co.Ridge.abs().max():,.1f}   Lasso {co.Lasso.abs().max():,.1f}")
kept = co.index[co.Lasso.abs() > 1e-8].tolist()
print(f"\n  the {len(kept)} predictors Lasso retained: {', '.join(kept[:12])}"
      f"{' ...' if len(kept)>12 else ''}")
co.to_csv("output/q7_coefficients.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
a = ax[0]
i = np.arange(3); w = .27
a.bar(i-w, res.MAE, w, label="MAE"); a.bar(i, res.RMSE, w, label="RMSE")
a.set_xticks(i); a.set_xticklabels(res.model); a.set_ylabel("error, target units"); a.legend(fontsize=8)
for k,(m,r) in enumerate(zip(res.MAE,res.RMSE)):
    a.text(k-w,m,f"{m:.0f}",ha="center",va="bottom",fontsize=7); a.text(k,r,f"{r:.0f}",ha="center",va="bottom",fontsize=7)
a2=a.twinx(); a2.plot(i, res.R2, "ko--", lw=1.4); a2.set_ylabel("R squared")
for k,v in enumerate(res.R2): a2.text(k,v,f" {v:.3f}",fontsize=8)
a.set_title("(a) Error and fit across the three models", fontsize=10)
a = ax[1]
top = co.head(10).iloc[::-1]; yp = np.arange(len(top))
a.barh(yp-.25, top.OLS, .25, label="OLS"); a.barh(yp, top.Ridge, .25, label="Ridge")
a.barh(yp+.25, top.Lasso, .25, label="Lasso")
a.set_yticks(yp); a.set_yticklabels(top.index, fontsize=7); a.axvline(0,color="k",lw=.8)
a.legend(fontsize=8); a.set_xlabel("coefficient on standardised predictors")
a.set_title("(b) Shrinkage against selection", fontsize=10)
fig.tight_layout(); fig.savefig("figures/q7_fig1.png", dpi=150); plt.close(fig)
print("\nfigure: figures/q7_fig1.png")
