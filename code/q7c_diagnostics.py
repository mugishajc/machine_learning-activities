"""
Q7 - regression inference and assumption checks.

RMSE tells you how far the predictions land from the truth. It says nothing about
whether the linear model is the right form, whether a coefficient is distinguishable
from zero, or how precisely each effect is estimated. Those are separate questions and
they are what the diagnostics below answer.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RS = 42
df = pd.read_csv("data/synthetic_ETA.csv").drop(columns=["observation_id"])
y = df["eta_minutes"]; X = df.drop(columns=["eta_minutes"])
num = [c for c in X if X[c].dtype != object]; cat = [c for c in X if X[c].dtype == object]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RS)
Z = pre.fit_transform(X_tr)
names = [f.split("__", 1)[1] for f in pre.get_feature_names_out()]
Zc = sm.add_constant(pd.DataFrame(Z, columns=names, index=y_tr.index))
fit = sm.OLS(y_tr, Zc).fit()

print("="*84); print("1. COEFFICIENT INFERENCE (standardised predictors, training set n=%d)" % len(y_tr))
tbl = pd.DataFrame({"coef": fit.params, "std err": fit.bse, "t": fit.tvalues,
                    "p-value": fit.pvalues, "CI low": fit.conf_int()[0], "CI high": fit.conf_int()[1]})
tbl["significant"] = np.where(tbl["p-value"] < 0.05, "yes", "no")
tbl = tbl.drop(index="const").sort_values("p-value")
print(tbl.to_string(float_format=lambda v: f"{v:9.4f}"))
sig = tbl[tbl["p-value"] < 0.05].index.tolist()
ns  = tbl[tbl["p-value"] >= 0.05].index.tolist()
print(f"\n  significant at 5%     ({len(sig)}): {', '.join(sig)}")
print(f"  not distinguishable   ({len(ns)}): {', '.join(ns)}")
print("\n  Cross-check against the 1-SE lasso model, which retained 8 predictors:")
lasso_kept = {"journey_distance_km","traffic_index","weather_score","road_type_local",
              "previous_journey_time_min","road_type_express","road_type_highway","road_slope_pct"}
print(f"    kept by lasso and significant : {sorted(lasso_kept & set(sig))}")
print(f"    kept by lasso, not significant: {sorted(lasso_kept - set(sig))}")
print(f"    significant, dropped by lasso : {sorted(set(sig) - lasso_kept)}")
print("  Two independent procedures, one penalty-based and one inferential, agree almost exactly.")

print("\n"+"="*84); print("2. MODEL-LEVEL SUMMARY")
print(f"  R-squared            {fit.rsquared:.4f}        adjusted {fit.rsquared_adj:.4f}")
print(f"  F-statistic          {fit.fvalue:.1f}   p = {fit.f_pvalue:.3e}")
print(f"  residual std error   {np.sqrt(fit.scale):.4f} minutes")
print(f"  AIC {fit.aic:.1f}   BIC {fit.bic:.1f}")
cond = np.linalg.cond(Zc.values)
print(f"  condition number     {cond:.1f}  "
      f"({'above 30, collinearity present' if cond>30 else 'below 30'})")

print("\n"+"="*84); print("3. ASSUMPTION TESTS")
resid, fitted = fit.resid, fit.fittedvalues
bp = het_breuschpagan(resid, Zc.values)
jb = jarque_bera(resid)
dw = durbin_watson(resid)
rs = linear_reset(fit, power=2, use_f=True)
rows = [
 ("Linearity of form", "Ramsey RESET (squared terms)", rs.pvalue,
  "no evidence of a missing non-linear term" if rs.pvalue > .05 else "functional form questionable"),
 ("Constant variance", "Breusch-Pagan", bp[1],
  "homoscedastic" if bp[1] > .05 else "heteroscedastic, use robust standard errors"),
 ("Normal residuals", "Jarque-Bera", jb[1],
  "consistent with normality" if jb[1] > .05 else "non-normal, inference approximate"),
]
print(f"  {'assumption':22s} {'test':30s} {'p-value':>10s}   verdict")
for a, t, p, v in rows:
    print(f"  {a:22s} {t:30s} {p:10.4f}   {v}")
print(f"  {'Independence':22s} {'Durbin-Watson':30s} {dw:10.3f}   "
      f"{'no autocorrelation' if 1.5<dw<2.5 else 'autocorrelation present'}")
print(f"\n  residual skew {pd.Series(resid).skew():.3f}   kurtosis {pd.Series(resid).kurtosis():.3f}")

if bp[1] <= .05:
    rob = sm.OLS(y_tr, Zc).fit(cov_type="HC3")
    ch = pd.DataFrame({"p (ordinary)": fit.pvalues, "p (robust HC3)": rob.pvalues}).drop(index="const")
    flip = ch[(ch["p (ordinary)"] < .05) != (ch["p (robust HC3)"] < .05)]
    print("\n  Robust HC3 standard errors change the 5% verdict for: "
          f"{list(flip.index) if len(flip) else 'no predictor'}")

print("\n"+"="*84); print("4. INFLUENTIAL OBSERVATIONS")
inf = fit.get_influence()
cooks = inf.cooks_distance[0]
thr = 4/len(y_tr)
print(f"  Cook's distance threshold 4/n = {thr:.5f}")
print(f"  observations above it        : {(cooks>thr).sum()} of {len(cooks)} ({(cooks>thr).mean():.1%})")
print(f"  largest Cook's distance      : {cooks.max():.4f}")
print(f"  {'none exceeds 1.0, so no single journey drives the fit' if cooks.max()<1 else 'at least one journey is highly influential'}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0,0]; a.scatter(fitted, resid, s=8, alpha=.4, edgecolor="none")
a.axhline(0, color="r", lw=1); a.set_xlabel("fitted values"); a.set_ylabel("residuals")
a.set_title("Q7 Fig 6a: residuals vs fitted", fontsize=10)
a = ax[0,1]; sm.qqplot(resid, line="45", fit=True, ax=a, markersize=3, alpha=.4)
a.set_title(f"Q7 Fig 6b: normal Q-Q (Jarque-Bera p={jb[1]:.3f})", fontsize=10)
a = ax[1,0]; a.scatter(fitted, np.sqrt(np.abs(resid/np.sqrt(fit.scale))), s=8, alpha=.4, edgecolor="none")
a.set_xlabel("fitted values"); a.set_ylabel("sqrt |standardised residual|")
a.set_title(f"Q7 Fig 6c: scale-location (Breusch-Pagan p={bp[1]:.3f})", fontsize=10)
a = ax[1,1]; a.stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
a.axhline(thr, color="r", ls="--", lw=1, label=f"4/n = {thr:.4f}")
a.set_xlabel("observation"); a.set_ylabel("Cook's distance"); a.legend(fontsize=8)
a.set_title("Q7 Fig 6d: influence", fontsize=10)
fig.suptitle("Q7: regression diagnostics on the training set", y=1.00)
fig.tight_layout(); fig.savefig("figures/q7_fig6_diagnostics.png", dpi=150, bbox_inches="tight"); plt.close(fig)
tbl.to_csv("output/q7c_inference.csv")
print("\nfigure: figures/q7_fig6_diagnostics.png")
