"""
Question Four - loan amount prediction: OLS, Ridge and Lasso on loan.csv.

The first task on any regression problem is not to fit a model. It is to establish
that the predictors could have been observed before the target existed. That audit
decides everything that follows here.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

RS = 42
df = pd.read_csv("data/loan.csv")
TARGET = "LoanAmount"
y = df[TARGET]

print("="*86); print("1. DATA")
print(f"  applicants {len(df):,}   variables {df.shape[1]}   missing cells {df.isna().sum().sum()}")
print(f"  target {TARGET}: mean {y.mean():,.0f}  sd {y.std():,.0f}  "
      f"range {y.min():,.0f} to {y.max():,.0f}")

# ---------------------------------------------------------------- leakage audit
print("\n"+"="*86); print("2. LEAKAGE AUDIT, RUN BEFORE ANY MODEL IS FITTED")
r = df.InterestRate/12
recon = df[TARGET]*r/(1-(1+r)**(-df.LoanDuration))
print(f"  MonthlyLoanPayment reconstructed from LoanAmount by the amortisation formula:")
print(f"    max absolute error {np.abs(recon-df.MonthlyLoanPayment).max():.6f}   "
      f"correlation {recon.corr(df.MonthlyLoanPayment):.8f}")
tdti = (df.MonthlyDebtPayments+df.MonthlyLoanPayment)/df.MonthlyIncome
print(f"  TotalDebtToIncomeRatio = (MonthlyDebtPayments + MonthlyLoanPayment) / MonthlyIncome:")
print(f"    mean absolute difference {np.abs(tdti-df.TotalDebtToIncomeRatio).mean():.8f}")
print("\n  Both are arithmetic on the target. Neither exists before the loan is sized, so")
print("  neither can be an input to a model that sizes the loan.")

POST = ["MonthlyLoanPayment", "TotalDebtToIncomeRatio", "InterestRate",
        "BaseInterestRate", "RiskScore", "LoanApproved"]
POST = [c for c in POST if c in df.columns]
print(f"\n  excluded as post-decision: {', '.join(POST)}")

drop = [TARGET] + POST + [c for c in df.columns if df[c].dtype == "O" and df[c].nunique() > 40]
X_all = df.drop(columns=[TARGET])
X = df.drop(columns=drop)
num = [c for c in X if X[c].dtype != object]
cat = [c for c in X if X[c].dtype == object]
print(f"  predictors retained: {len(num)} numeric + {len(cat)} categorical ({', '.join(cat)})")

# ---------------------------------------------------------------- multicollinearity
print("\n"+"="*86); print("3. MULTICOLLINEARITY AMONG THE RETAINED PREDICTORS")
Z = pd.DataFrame(StandardScaler().fit_transform(X[num]), columns=num)
Z.insert(0, "const", 1.0)
vif = pd.DataFrame({"predictor": num,
                    "VIF": [variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])]}
                   ).sort_values("VIF", ascending=False)
print(vif.head(10).to_string(index=False, float_format=lambda v: f"{v:10.2f}"))
pairs = X[num].corr().abs().unstack().sort_values(ascending=False)
pairs = pairs[pairs < 0.999].drop_duplicates()
print("\n  strongest predictor pairs:")
for (a, b), v in pairs.head(5).items():
    print(f"    {a:26s} <-> {b:26s} r = {v:.3f}")

# ---------------------------------------------------------------- fit
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
cv = KFold(10, shuffle=True, random_state=RS)
alphas = np.logspace(-2, 4, 61)

def run(Xd, label):
    Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.25, random_state=RS)
    nn = [c for c in Xd if Xd[c].dtype != object]; cc = [c for c in Xd if Xd[c].dtype == object]
    p = ColumnTransformer([("num", StandardScaler(), nn),
                           ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cc)])
    out = []
    for name, est, grid in [("OLS", LinearRegression(), None),
                            ("Ridge", Ridge(random_state=RS), {"model__alpha": alphas}),
                            ("Lasso", Lasso(random_state=RS, max_iter=20000), {"model__alpha": alphas})]:
        pipe = Pipeline([("pre", p), ("model", est)])
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1).fit(Xtr, ytr)
            best, a = gs.best_estimator_, gs.best_params_["model__alpha"]
        else:
            best, a = pipe.fit(Xtr, ytr), np.nan
        pr = best.predict(Xte)
        nz = int(np.sum(np.abs(best.named_steps["model"].coef_) > 1e-8))
        out.append({"feature set": label, "model": name, "alpha": a,
                    "test RMSE": mean_squared_error(yte, pr)**0.5,
                    "test MAE": mean_absolute_error(yte, pr),
                    "test R2": r2_score(yte, pr), "non-zero coefs": nz})
    return pd.DataFrame(out), best, p, Xtr, ytr

print("\n"+"="*86); print("4. THE COST OF THE AUDIT")
leaky, *_ = run(X_all.drop(columns=[c for c in X_all if X_all[c].dtype=="O" and X_all[c].nunique()>40]),
                "with post-decision variables")
honest, model, p, Xtr, ytr = run(X, "pre-decision variables only")
res = pd.concat([leaky, honest], ignore_index=True)
print(res.to_string(index=False, float_format=lambda v: f"{v:12.4f}"))
res.to_csv("output/q4_results.csv", index=False)
print("\n  With the leaky variables the model is near perfect. Without them none of the three")
print("  explains anything: test R-squared sits at zero, meaning the fitted models predict")
print("  no better than the mean of the training target.")

print("\n"+"="*86); print("5. RIDGE AND LASSO ACT DIFFERENTLY ON THE COEFFICIENTS")
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25, random_state=RS)
fitted = {}
for name, est in [("OLS", LinearRegression()),
                  ("Ridge (a=100)", Ridge(alpha=100, random_state=RS)),
                  ("Lasso (a=100)", Lasso(alpha=100, random_state=RS, max_iter=20000))]:
    m = Pipeline([("pre", pre), ("model", est)]).fit(Xtr2, ytr2)
    fitted[name] = m.named_steps["model"].coef_
names = [f.split("__", 1)[1] for f in Pipeline([("pre", pre)]).fit(Xtr2).named_steps["pre"].get_feature_names_out()]
coef = pd.DataFrame(fitted, index=names)
coef["|OLS|"] = coef["OLS"].abs()
coef = coef.sort_values("|OLS|", ascending=False).drop(columns="|OLS|")
print(coef.head(12).to_string(float_format=lambda v: f"{v:11.3f}"))
print(f"\n  coefficients set exactly to zero:  OLS {int((coef['OLS']==0).sum())}   "
      f"Ridge {int((coef['Ridge (a=100)']==0).sum())}   Lasso {int((coef['Lasso (a=100)']==0).sum())} of {len(coef)}")
print(f"  sum of absolute coefficients:      OLS {coef['OLS'].abs().sum():.1f}   "
      f"Ridge {coef['Ridge (a=100)'].abs().sum():.1f}   Lasso {coef['Lasso (a=100)'].abs().sum():.1f}")
print("  Ridge shrinks every coefficient towards zero but keeps all of them. Lasso removes")
print("  predictors outright, which is selection rather than shrinkage alone.")
coef.to_csv("output/q4_coefficients.csv")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
a = ax[0]
sub = res[res.model.isin(["OLS","Ridge","Lasso"])]
w = 0.35; idx = np.arange(3)
a.bar(idx-w/2, sub[sub["feature set"].str.startswith("with")]["test R2"], w, label="with post-decision variables", color="#b2182b")
a.bar(idx+w/2, sub[sub["feature set"].str.startswith("pre")]["test R2"], w, label="pre-decision only", color="#2166ac")
a.set_xticks(idx); a.set_xticklabels(["OLS","Ridge","Lasso"]); a.set_ylabel("test R squared")
a.axhline(0, color="k", lw=.8); a.legend(fontsize=8)
a.set_title("Q4 Fig 1a: the entire signal comes from leakage", fontsize=10)
a = ax[1]
top = coef.head(10).iloc[::-1]
yp = np.arange(len(top))
a.barh(yp-0.25, top["OLS"], 0.25, label="OLS")
a.barh(yp, top["Ridge (a=100)"], 0.25, label="Ridge")
a.barh(yp+0.25, top["Lasso (a=100)"], 0.25, label="Lasso")
a.set_yticks(yp); a.set_yticklabels(top.index, fontsize=8); a.axvline(0, color="k", lw=.8)
a.legend(fontsize=8); a.set_xlabel("coefficient")
a.set_title("Q4 Fig 1b: shrinkage against selection", fontsize=10)
fig.tight_layout(); fig.savefig("figures/q4_fig1.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for nm, est in [("Ridge", Ridge), ("Lasso", Lasso)]:
    sc = []
    for al in alphas:
        m = Pipeline([("pre", pre), ("model", est(alpha=al, random_state=RS, max_iter=20000)
                                     if est is Lasso else est(alpha=al, random_state=RS))])
        sc.append(-cross_val_score(m, Xtr2, ytr2, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1).mean())
    ax.plot(alphas, sc, label=nm, lw=1.6)
ax.set_xscale("log"); ax.set_xlabel("alpha (log scale)"); ax.set_ylabel("cross-validated RMSE")
ax.legend(); ax.set_title("Q4 Fig 2: penalty selection by cross-validation on the training set")
fig.tight_layout(); fig.savefig("figures/q4_fig2.png", dpi=150); plt.close(fig)
print("\nfigures: figures/q4_fig1.png, figures/q4_fig2.png")
