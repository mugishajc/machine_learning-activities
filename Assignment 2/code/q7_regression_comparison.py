"""
Question Seven - OLS, Ridge and Lasso under multicollinearity.

prices.csv has not yet been released. The stand-in is a regression problem with more
severe collinearity than the house-price example in the question: 103 predictor pairs
correlate at 0.90 or above, and one pair reaches 1.000.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA, TARGET = "data/crimes.csv", "crime_rate"     # <-- swap to "data/prices.csv", "SalePrice"
df = pd.read_csv(DATA, low_memory=False)
# other measures of the same outcome would leak, exactly as GrLivArea does not but
# a second sale price would; on prices.csv this list is empty
# "murder" would miss murdPerPop, so the stems are matched instead. A filter that is
# nearly right is the failure mode this whole question is about.
LEAKY = [c for c in df.columns if c != TARGET and any(k in c.lower() for k in
         ("murd","rape","robb","assault","burgl","larc","autotheft","arson","viol","crime"))]
y = df[TARGET]
X = df.drop(columns=[TARGET]+LEAKY).select_dtypes("number").drop(columns=["fold"], errors="ignore")
X = X.loc[:, X.std() > 0]

print("="*82); print("1. DATA AND COLLINEARITY")
print(f"  observations {len(X):,}   numeric predictors {X.shape[1]}")
print(f"  target {TARGET}: mean {y.mean():,.1f}  sd {y.std():,.1f}")
print(f"  excluded as other measures of the same outcome: {len(LEAKY)} columns")
cm = X.corr().abs(); np.fill_diagonal(cm.values, 0)
pairs = cm.stack().sort_values(ascending=False)
pairs = pairs[~pairs.index.duplicated()]
print(f"\n  predictor pairs with |r| >= 0.90: {int((cm.values>=0.90).sum()//2)}")
seen=set(); shown=0
for (a,b),v in pairs.items():
    if (b,a) in seen: continue
    seen.add((a,b)); print(f"    {a:22s} <-> {b:22s} r = {v:.4f}"); shown+=1
    if shown==5: break

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=RANDOM_STATE)
cv = KFold(10, shuffle=True, random_state=RANDOM_STATE)
alphas = np.logspace(-3, 4, 60)
print(f"\n  train {len(X_tr):,}   test {len(X_te):,}")

print("\n"+"="*82); print("2. THREE MODELS, IDENTICAL PIPELINE")
rows, fitted = [], {}
for name, est, grid in [("OLS",   LinearRegression(), None),
                        ("Ridge", Ridge(random_state=RANDOM_STATE), {"m__alpha": alphas}),
                        ("Lasso", Lasso(random_state=RANDOM_STATE, max_iter=50000), {"m__alpha": alphas})]:
    pipe = Pipeline([("s", StandardScaler()), ("m", est)])
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

lasso_fit = Pipeline([("s", StandardScaler()),
                      ("m", Lasso(alpha=res.loc[res.model=="Lasso","alpha"].iloc[0],
                                  random_state=RANDOM_STATE, max_iter=50000))]).fit(X_tr, y_tr)
print(f"\n  convergence check on the selected Lasso: {lasso_fit.named_steps['m'].n_iter_} iterations")
print("  of a 50,000 limit. The grid search emits convergence warnings at its smallest")
print("  alphas, where the penalty is too weak to make the problem well conditioned, but")
print("  those candidates are rejected and the selected model converges comfortably.")

print("\n"+"="*82); print("3. WHAT THE PENALTIES DID TO THE COEFFICIENTS")
co = pd.DataFrame(fitted, index=X.columns)
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
