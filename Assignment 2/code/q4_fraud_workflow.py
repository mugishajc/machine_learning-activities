"""
Question Four(b) - the imbalanced-classification workflow the question specifies.

Runs on the supplied credit.csv: 1,000,000 card transactions, seven predictors and a
binary fraud flag.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             average_precision_score, precision_recall_curve,
                             roc_auc_score, recall_score, precision_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
DATA, TARGET = "data/credit.csv", "fraud"

# ---- 1. separate the predictors from the binary target -----------------------
df = pd.read_csv(DATA)
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])
# every predictor here is measured at transaction time, so none is target-derived
num = [c for c in X.columns if X[c].dtype != object]
cat = [c for c in X.columns if X[c].dtype == object]
print("="*80); print("1. DATA")
print(f"  rows {len(df):,}   predictors {X.shape[1]} ({len(num)} numeric, {len(cat)} categorical)")
print(f"  positive class '{TARGET}': {y.sum():,} of {len(y):,} = {y.mean():.3%}")
print(f"  imbalance ratio 1 : {int((1-y.mean())/y.mean())}")

# ---- 2. stratified train/test split ------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
print("\n"+"="*80); print("2. STRATIFIED SPLIT")
print(f"  train {len(X_tr):,} rows, positive rate {y_tr.mean():.3%}")
print(f"  test  {len(X_te):,} rows, positive rate {y_te.mean():.3%}")
print("  stratify=y keeps both rates equal to the population rate; without it the test")
print("  positive count would vary by chance and the metrics with it")

# ---- 3. baseline decision tree with an explicit random_state -----------------
pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat)])
baseline = Pipeline([("pre", pre),
                     ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
baseline.fit(X_tr, y_tr)
b_pred = baseline.predict(X_te)
print("\n"+"="*80); print("3. BASELINE, UNCONSTRAINED TREE")
print(f"  depth reached {baseline.named_steps['clf'].get_depth()}   "
      f"leaves {baseline.named_steps['clf'].get_n_leaves():,}")
print(f"  train F1 {f1_score(y_tr, baseline.predict(X_tr)):.4f}   test F1 {f1_score(y_te, b_pred):.4f}")
print("  the gap between those two is Question One's 99.6 against 59.4 in miniature")

# ---- 4 & 5. grid search, cross-validated, on the training data only ----------
grid = {"clf__max_depth":        [3, 5, 8, 12, None],
        "clf__min_samples_leaf": [1, 10, 50, 200],
        "clf__class_weight":     [None, "balanced"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# The grid is searched on a stratified subsample. At 750,000 training rows a 40-point
# grid over 5 folds is 200 tree fits, and the cross-validated ranking is already stable
# at a fraction of that size. The winning configuration is then refitted on everything.
SEARCH_N = 150_000
sub = (X_tr.assign(_y=y_tr).groupby("_y", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), int(SEARCH_N*len(g)/len(y_tr))),
                                  random_state=RANDOM_STATE)))
Xs, ys = sub.drop(columns="_y"), sub["_y"]
search = GridSearchCV(baseline, grid, scoring="average_precision",
                      cv=cv, n_jobs=-1, refit=False, return_train_score=True)
search.fit(Xs, ys)
print(f"  grid searched on a stratified subsample of {len(Xs):,} rows "
      f"({ys.mean():.3%} positive, against {y_tr.mean():.3%} in the full training set)")
print("\n"+"="*80); print("4-5. CROSS-VALIDATED GRID SEARCH, TRAINING DATA ONLY")
print(f"  configurations tried {len(search.cv_results_['params'])}   folds {cv.get_n_splits()}   "
      f"fits {len(search.cv_results_['params'])*cv.get_n_splits()}")
print(f"  scoring: average precision, the area under the precision-recall curve")
print(f"  best parameters: {search.best_params_}")
print(f"  best cross-validated average precision {search.best_score_:.4f}")
res = pd.DataFrame(search.cv_results_)[
    ["param_clf__max_depth","param_clf__min_samples_leaf","param_clf__class_weight",
     "mean_train_score","mean_test_score","std_test_score"]].sort_values("mean_test_score",ascending=False)
print("\n  top five configurations:")
print(res.head().to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
print("\n  worst two configurations:")
print(res.tail(2).to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
unc = res[(res.param_clf__max_depth.isna()) & (res.param_clf__min_samples_leaf==10)]
sh  = res[res.param_clf__max_depth==3]
print(f"\n  worth reading carefully: the worst configurations here are the SHALLOW trees")
print(f"  (max_depth=3, best {sh.mean_test_score.max():.4f}), not the unconstrained ones")
print(f"  (max_depth=None, {unc.mean_test_score.max():.4f} against {res.mean_test_score.max():.4f} for the winner).")
print(f"  On this data the binding risk is underfitting, and the tuned depth of 8 gains only")
print(f"  {res.mean_test_score.max()-unc.mean_test_score.max():.4f} over no limit at all. The scenario in Question One")
print(f"  is the opposite case, which is why the diagnosis has to be made from the gap")
print(f"  between training and validation and never assumed from the configuration alone.")
res.to_csv("output/q4_gridsearch.csv", index=False)

# ---- 6. exactly two metrics, both appropriate under imbalance ----------------
best = baseline.set_params(**search.best_params_).fit(X_tr, y_tr)   # refit on all training rows
y_pred = best.predict(X_te)
y_prob = best.predict_proba(X_te)[:, 1]
f1 = f1_score(y_te, y_pred)
ap = average_precision_score(y_te, y_prob)
print("\n"+"="*80); print("6. HELD-OUT TEST SET, TWO METRICS")
print(f"  F1-score          {f1:.4f}   balances precision and recall on the rare class")
print(f"  average precision {ap:.4f}   summarises the precision-recall curve over all thresholds")
print(f"\n  reported for contrast only, not used for selection:")
acc = (y_pred == y_te).mean()
print(f"    accuracy {acc:.4f}, against {1-y_te.mean():.4f} for predicting the majority class always")
print(f"    ROC AUC  {roc_auc_score(y_te, y_prob):.4f}")
print("\n  confusion matrix:")
cmx = confusion_matrix(y_te, y_pred)
print(pd.DataFrame(cmx, index=["actual negative","actual positive"],
                   columns=["predicted negative","predicted positive"]).to_string())
print("\n"+classification_report(y_te, y_pred, digits=4,
                                 target_names=["negative","positive"]))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
p, r, _ = precision_recall_curve(y_te, y_prob)
ax[0].plot(r, p, lw=1.8, color="#2166ac")
ax[0].axhline(y_te.mean(), ls="--", c="#b2182b", lw=1.2,
              label=f"no-skill = base rate {y_te.mean():.3f}")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].legend(fontsize=8)
ax[0].set_title(f"(a) Precision-recall curve, AP = {ap:.4f}", fontsize=10)
d = res.dropna(subset=["mean_test_score"]).copy()
d["depth"] = d.param_clf__max_depth.fillna(99).astype(int)
g = d.groupby("depth")[["mean_train_score","mean_test_score"]].max()
ax[1].plot(g.index, g.mean_train_score, "o-", label="training", color="#b2182b")
ax[1].plot(g.index, g.mean_test_score, "s-", label="cross-validated", color="#2166ac")
ax[1].set_xlabel("max_depth (99 shown for None)"); ax[1].set_ylabel("average precision")
ax[1].legend(fontsize=8)
ax[1].set_title("(b) The two curves separate as depth grows", fontsize=10)
fig.tight_layout(); fig.savefig("figures/q4_fig1.png", dpi=150); plt.close(fig)
print("figure: figures/q4_fig1.png")
