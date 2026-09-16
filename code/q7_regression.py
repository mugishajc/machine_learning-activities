"""
Q7 - YEGO ETA prediction: OLS vs Ridge vs Lasso
MIT91207 Machine Learning, Assignment 1
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

RS = 42
np.random.seed(RS)
df = pd.read_csv("data/synthetic_ETA.csv")

# ---------- 1. Prepare ----------
TARGET = "eta_minutes"
df = df.drop(columns=["observation_id"])          # identifier, never a predictor
y = df[TARGET]
X = df.drop(columns=[TARGET])
num_cols = [c for c in X.columns if X[c].dtype != object]
cat_cols = [c for c in X.columns if X[c].dtype == object]

print("="*78); print("1. DATA PREPARATION")
print(f"observations: {len(df)}   predictors: {X.shape[1]} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
print(f"missing values: {df.isna().sum().sum()}   duplicate rows: {df.duplicated().sum()}")
print(f"target {TARGET}: mean={y.mean():.2f} sd={y.std():.2f} min={y.min():.2f} max={y.max():.2f}")

# ---------- 2. Multicollinearity diagnosis ----------
print("\n"+"="*78); print("2. MULTICOLLINEARITY (variance inflation factors, numeric predictors)")
Xn = X[num_cols].copy()
Xs = pd.DataFrame(StandardScaler().fit_transform(Xn), columns=num_cols)
Xs.insert(0, "const", 1.0)
vif = pd.DataFrame({
    "predictor": num_cols,
    "VIF": [variance_inflation_factor(Xs.values, i) for i in range(1, Xs.shape[1])],
}).sort_values("VIF", ascending=False)
print(vif.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

corr = X[num_cols].corrwith(y).sort_values(key=abs, ascending=False)
print("\nCorrelation of each numeric predictor with eta_minutes:")
print(corr.to_string(float_format=lambda v: f"{v:7.3f}"))

pairs = X[num_cols].corr().abs().unstack().sort_values(ascending=False)
pairs = pairs[pairs < 0.999].drop_duplicates()
print("\nStrongest predictor-predictor correlations:")
for (a, b), v in pairs.head(5).items():
    print(f"  {a:26s} <-> {b:26s} r = {v:.3f}")

# ---------- 3. Split: train / test, CV inside train ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RS)
print("\n"+"="*78); print("3. SPLIT")
print(f"train: {len(X_tr)}   test (held out, touched once): {len(X_te)}")

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])
cv = KFold(n_splits=10, shuffle=True, random_state=RS)

# ---------- 4. Fit three models ----------
print("\n"+"="*78); print("4. MODEL FITTING (alpha chosen by 10-fold CV on training set only)")
alphas = np.logspace(-3, 3, 61)
models, results = {}, []

specs = [
    ("Linear Regression (OLS)", LinearRegression(), None),
    ("Ridge Regression",        Ridge(random_state=RS),        {"model__alpha": alphas}),
    ("Lasso Regression",        Lasso(random_state=RS, max_iter=50000), {"model__alpha": alphas}),
]

for name, est, grid in specs:
    pipe = Pipeline([("pre", pre), ("model", est)])
    if grid:
        gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best, alpha = gs.best_estimator_, gs.best_params_["model__alpha"]
    else:
        best, alpha = pipe.fit(X_tr, y_tr), np.nan
    cv_rmse = -cross_val_score(best, X_tr, y_tr, scoring="neg_root_mean_squared_error", cv=cv).mean()
    p_tr, p_te = best.predict(X_tr), best.predict(X_te)
    results.append({
        "Model": name, "alpha": alpha, "CV RMSE": cv_rmse,
        "Train RMSE": mean_squared_error(y_tr, p_tr)**0.5,
        "Test RMSE":  mean_squared_error(y_te, p_te)**0.5,
        "Test MAE":   mean_absolute_error(y_te, p_te),
        "Train R2":   r2_score(y_tr, p_tr), "Test R2": r2_score(y_te, p_te),
    })
    models[name] = best
    print(f"  {name:26s} alpha={alpha if alpha==alpha else 0:8.4f}  CV RMSE={cv_rmse:.4f}")

res = pd.DataFrame(results)
print("\n"+"="*78); print("5. PERFORMANCE COMPARISON")
print(res.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

# ---------- 6. Coefficients ----------
feat = models["Linear Regression (OLS)"].named_steps["pre"].get_feature_names_out()
feat = [f.split("__", 1)[1] for f in feat]
coef = pd.DataFrame({n: m.named_steps["model"].coef_ for n, m in models.items()}, index=feat)
coef["|OLS|"] = coef["Linear Regression (OLS)"].abs()
coef = coef.sort_values("|OLS|", ascending=False).drop(columns="|OLS|")
print("\n"+"="*78); print("6. STANDARDISED COEFFICIENTS (minutes of ETA per 1 SD of predictor)")
print(coef.to_string(float_format=lambda v: f"{v:9.3f}"))

nz = (coef["Lasso Regression"].abs() < 1e-8)
print(f"\nPredictors eliminated by Lasso ({nz.sum()} of {len(coef)}):")
for f in coef.index[nz]: print("  -", f)

res.to_csv("output/q7_metrics.csv", index=False)
coef.to_csv("output/q7_coefficients.csv")
vif.to_csv("output/q7_vif.csv", index=False)

# ---------- 7. Figures ----------
fig, ax = plt.subplots(figsize=(9, 7))
cm = X[num_cols + []].assign(eta_minutes=y).corr()
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=8)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(cm.iloc[i, j]) > 0.6 else "black")
ax.set_title("Q7 Fig 1: Correlation matrix, numeric predictors and ETA")
fig.colorbar(im, shrink=0.8); fig.tight_layout(); fig.savefig("figures/q7_fig1_corr.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
for axx, (n, m) in zip(axes, models.items()):
    p = m.predict(X_te)
    axx.scatter(y_te, p, s=12, alpha=0.5, edgecolor="none")
    lo, hi = y_te.min(), y_te.max()
    axx.plot([lo, hi], [lo, hi], "r--", lw=1)
    axx.set_title(f"{n}\nTest RMSE={mean_squared_error(y_te,p)**0.5:.3f}  R²={r2_score(y_te,p):.4f}", fontsize=10)
    axx.set_xlabel("Actual ETA (min)")
axes[0].set_ylabel("Predicted ETA (min)")
fig.suptitle("Q7 Fig 2: Predicted vs actual on the held-out test set", y=1.02)
fig.tight_layout(); fig.savefig("figures/q7_fig2_pred.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
coef.plot.barh(ax=ax, width=0.8)
ax.axvline(0, color="k", lw=0.8); ax.invert_yaxis()
ax.set_xlabel("Standardised coefficient (minutes per 1 SD)")
ax.set_title("Q7 Fig 3: Coefficient shrinkage, OLS vs Ridge vs Lasso")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/q7_fig3_coef.png", dpi=150); plt.close(fig)

# Lasso path
fig, ax = plt.subplots(figsize=(9, 5.5))
paths = []
for a in alphas:
    p = Pipeline([("pre", pre), ("model", Lasso(alpha=a, random_state=RS, max_iter=50000))]).fit(X_tr, y_tr)
    paths.append(p.named_steps["model"].coef_)
paths = np.array(paths)
for i, f in enumerate(feat):
    ax.plot(alphas, paths[:, i], label=f, lw=1.4)
best_a = res.loc[res.Model == "Lasso Regression", "alpha"].iloc[0]
ax.axvline(best_a, color="k", ls=":", lw=1.2)
ax.text(best_a, ax.get_ylim()[1]*0.92, f" CV-optimal α={best_a:.3f}", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("alpha (log scale)"); ax.set_ylabel("Standardised coefficient")
ax.set_title("Q7 Fig 4: Lasso coefficient paths, uninformative predictors reach zero first")
ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig("figures/q7_fig4_path.png", dpi=150); plt.close(fig)

print("\nFigures written: figures/q7_fig1..4")
print("Tables written:  output/q7_metrics.csv, q7_coefficients.csv, q7_vif.csv")
