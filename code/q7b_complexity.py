"""
Q7 (c)/(d) - why the three models tie on RMSE, and where regularisation still pays:
  A. 1-SE rule: cheapest model statistically indistinguishable from the best
  B. Coefficient stability under multicollinearity (bootstrap)
  C. Small-sample behaviour (learning curve) - when regularisation actually matters
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

RS = 42; rng = np.random.default_rng(RS)
df = pd.read_csv("data/synthetic_ETA.csv").drop(columns=["observation_id"])
y = df["eta_minutes"]; X = df.drop(columns=["eta_minutes"])
num = [c for c in X if X[c].dtype != object]; cat = [c for c in X if X[c].dtype == object]
pre = lambda: ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RS)
cv = KFold(10, shuffle=True, random_state=RS)
alphas = np.logspace(-3, 2, 60)

# ---------------- A. one-standard-error rule ----------------
print("="*78); print("A. ONE-STANDARD-ERROR RULE (Lasso)")
gs = GridSearchCV(Pipeline([("pre", pre()), ("model", Lasso(random_state=RS, max_iter=50000))]),
                  {"model__alpha": alphas}, scoring="neg_root_mean_squared_error",
                  cv=cv, n_jobs=-1, return_train_score=False).fit(X_tr, y_tr)
cvr = pd.DataFrame(gs.cv_results_)
mean_rmse = -cvr["mean_test_score"].values
se_rmse = cvr["std_test_score"].values / np.sqrt(cv.get_n_splits())
best_i = mean_rmse.argmin()
thresh = mean_rmse[best_i] + se_rmse[best_i]
elig = np.where(mean_rmse <= thresh)[0]
a_1se = alphas[elig.max()]
print(f"best alpha            = {alphas[best_i]:.4f}   CV RMSE = {mean_rmse[best_i]:.4f} (SE {se_rmse[best_i]:.4f})")
print(f"1-SE threshold        = {thresh:.4f}")
print(f"largest alpha within  = {a_1se:.4f}   CV RMSE = {mean_rmse[elig.max()]:.4f}")

feat_names = None; rows = []
for label, alpha in [("Lasso (CV-optimal)", alphas[best_i]), ("Lasso (1-SE rule)", a_1se)]:
    m = Pipeline([("pre", pre()), ("model", Lasso(alpha=alpha, random_state=RS, max_iter=50000))]).fit(X_tr, y_tr)
    feat_names = [f.split("__", 1)[1] for f in m.named_steps["pre"].get_feature_names_out()]
    c = m.named_steps["model"].coef_
    p = m.predict(X_te)
    rows.append({"Model": label, "alpha": alpha, "predictors kept": int((np.abs(c) > 1e-8).sum()),
                 "Test RMSE": mean_squared_error(y_te, p)**0.5, "Test R2": r2_score(y_te, p)})
    if "1-SE" in label:
        kept = pd.Series(c, index=feat_names)
        print(f"\nRetained by the 1-SE model ({(np.abs(c)>1e-8).sum()} of {len(c)}):")
        for f, v in kept[kept.abs() > 1e-8].sort_values(key=abs, ascending=False).items():
            print(f"    {f:28s} {v:8.3f}")
        print(f"Dropped ({(np.abs(c)<=1e-8).sum()}): " + ", ".join(kept[kept.abs() <= 1e-8].index))
full = Pipeline([("pre", pre()), ("model", LinearRegression())]).fit(X_tr, y_tr)
pf = full.predict(X_te)
rows.insert(0, {"Model": "OLS (all 15 predictors)", "alpha": np.nan, "predictors kept": 15,
                "Test RMSE": mean_squared_error(y_te, pf)**0.5, "Test R2": r2_score(y_te, pf)})
tblA = pd.DataFrame(rows)
print("\n"+tblA.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
tblA.to_csv("output/q7b_1se.csv", index=False)

# ---------------- B. coefficient stability ----------------
print("\n"+"="*78); print("B. COEFFICIENT STABILITY UNDER MULTICOLLINEARITY (300 bootstrap refits, n=200)")
B, nb = 300, 200
store = {"OLS": [], "Ridge": []}
for _ in range(B):
    idx = rng.choice(len(X_tr), nb, replace=True)
    xb, yb = X_tr.iloc[idx], y_tr.iloc[idx]
    for nm, est in [("OLS", LinearRegression()), ("Ridge", Ridge(alpha=10.0, random_state=RS))]:
        m = Pipeline([("pre", pre()), ("model", est)]).fit(xb, yb)
        store[nm].append(m.named_steps["model"].coef_)
stab = pd.DataFrame({
    "predictor": feat_names,
    "OLS sd":   np.std(np.array(store["OLS"]), axis=0),
    "Ridge sd": np.std(np.array(store["Ridge"]), axis=0),
})
stab["reduction %"] = (1 - stab["Ridge sd"] / stab["OLS sd"]) * 100
stab["OLS sign flips %"] = [ (np.sign(np.array(store["OLS"])[:, i]) != np.sign(np.median(np.array(store["OLS"])[:, i]))).mean()*100
                             for i in range(len(feat_names)) ]
stab = stab.sort_values("OLS sd", ascending=False)
print(stab.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
stab.to_csv("output/q7b_stability.csv", index=False)

# ---------------- C. learning curve ----------------
print("\n"+"="*78); print("C. TEST RMSE BY TRAINING-SET SIZE (mean of 40 random draws)")
sizes = [30, 50, 80, 120, 200, 400, 800, 1200]
out = []
for n in sizes:
    acc = {"OLS": [], "Ridge": [], "Lasso": []}
    for r in range(40):
        idx = np.random.default_rng(RS + r).choice(len(X_tr), n, replace=False)
        xb, yb = X_tr.iloc[idx], y_tr.iloc[idx]
        for nm, est in [("OLS", LinearRegression()),
                        ("Ridge", Ridge(alpha=10.0, random_state=RS)),
                        ("Lasso", Lasso(alpha=0.3, random_state=RS, max_iter=50000))]:
            m = Pipeline([("pre", pre()), ("model", est)]).fit(xb, yb)
            acc[nm].append(mean_squared_error(y_te, m.predict(X_te))**0.5)
    out.append({"n_train": n, **{k: np.mean(v) for k, v in acc.items()}})
lc = pd.DataFrame(out)
lc["OLS - Ridge"] = lc["OLS"] - lc["Ridge"]
print(lc.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
lc.to_csv("output/q7b_learning_curve.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for k, st in [("OLS", "o-"), ("Ridge", "s-"), ("Lasso", "^-")]:
    ax.plot(lc.n_train, lc[k], st, lw=1.6, ms=5, label=k)
ax.set_xscale("log"); ax.set_xlabel("training observations (log scale)")
ax.set_ylabel("test RMSE (minutes)"); ax.legend()
ax.set_title("Q7 Fig 5a: regularisation pays only when data is scarce")
ax = axes[1]
top = stab.head(8).iloc[::-1]
yp = np.arange(len(top))
ax.barh(yp - 0.2, top["OLS sd"], 0.4, label="OLS")
ax.barh(yp + 0.2, top["Ridge sd"], 0.4, label="Ridge (α=10)")
ax.set_yticks(yp); ax.set_yticklabels(top["predictor"], fontsize=8)
ax.set_xlabel("SD of coefficient across 300 bootstrap refits (n=200)"); ax.legend()
ax.set_title("Q7 Fig 5b: Ridge stabilises collinear coefficients")
fig.tight_layout(); fig.savefig("figures/q7_fig5_complexity.png", dpi=150); plt.close(fig)
print("\nFigure written: figures/q7_fig5_complexity.png")
