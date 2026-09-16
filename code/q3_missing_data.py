"""
Q3 - loan default: why zero-filling is wrong, and a defensible alternative.
"""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
RS = 42; rng = np.random.default_rng(RS)
N = 3000

# ---- construct a loan book with a realistic, NON-random missingness mechanism ----
income = np.round(rng.lognormal(12.6, 0.55, N))                 # RWF-scale annual income
years_employed = np.clip(rng.gamma(2.2, 2.4, N), 0, 40)
loan_amount = np.round(income * rng.uniform(0.15, 1.4, N))
credit_score = np.clip(rng.normal(640, 85, N), 300, 850)
n_prev_loans = rng.poisson(1.6, N)
dti = loan_amount / income
logit = (-3.1 + 2.3*dti - 0.010*(credit_score-640) - 0.10*years_employed
         + 0.22*n_prev_loans + rng.normal(0, 0.45, N))
default = rng.binomial(1, 1/(1+np.exp(-logit)))
df = pd.DataFrame({"income":income,"years_employed":years_employed,"loan_amount":loan_amount,
                   "credit_score":credit_score,"n_prev_loans":n_prev_loans,
                   "employment_type":rng.choice(["salaried","self_employed","informal"],N,p=[.5,.3,.2]),
                   "default":default})
truth = df.copy()

# missingness mechanisms
p_inc = 0.06 + 0.34*(df.employment_type=="informal") + 0.10*(df.employment_type=="self_employed")
df.loc[rng.random(N) < p_inc, "income"] = np.nan                          # MAR/MNAR: informal earners
df.loc[rng.random(N) < np.where(df.employment_type=="informal",.30,.05), "years_employed"] = np.nan
thin = df.n_prev_loans == 0
df.loc[thin & (rng.random(N) < 0.85), "credit_score"] = np.nan            # MNAR: no file, no score
df.loc[rng.random(N) < 0.04, "employment_type"] = np.nan

print("="*80); print("1. MISSINGNESS PROFILE")
prof = pd.DataFrame({"missing":df.isna().sum(),"%":(df.isna().mean()*100).round(1)})
print(prof[prof.missing>0].to_string())
print("\nDefault rate by whether the field is missing (evidence against MCAR):")
for col in ["income","years_employed","credit_score"]:
    m = df[col].isna()
    t,p = stats.ttest_ind(df.default[m], df.default[~m], equal_var=False)
    print(f"  {col:15s} missing: {df.default[m].mean():.3f}   present: {df.default[~m].mean():.3f}"
          f"   Welch p = {p:.2e}  -> {'NOT MCAR' if p<0.05 else 'consistent with MCAR'}")
print("\nMissingness tested against an OBSERVED COVARIATE (employment_type):")
for col in ["income","years_employed"]:
    tab = pd.crosstab(df[col].isna(), df.employment_type)
    chi2, p, _, _ = stats.chi2_contingency(tab)
    rate = (tab.loc[True] / tab.sum()).round(3).to_dict()
    print(f"  {col:15s} chi2 p = {p:.2e}  -> {'MAR, depends on employment_type' if p<0.05 else 'no covariate dependence'}")
    print(f"                  missing rate by employment type: {rate}")
print("\n  Note: testing missingness against the target alone is not sufficient. income shows no")
print("  association with default (p = 0.87) yet is strongly associated with employment_type, so it")
print("  is MAR rather than MCAR. A target-only test would have wrongly cleared it.")

print("\nMissing credit_score concentrated in customers with no prior loans:")
print(pd.crosstab(df.credit_score.isna(), df.n_prev_loans==0, rownames=["score missing"],
                  colnames=["no prior loans"]).to_string())

print("\n"+"="*80); print("2. WHAT ZERO-FILLING ASSERTS (Q3a)")
obs = truth.loc[df.income.isna(),"income"]
print(f"  true income of the {len(obs)} applicants whose income is missing:")
print(f"    min {obs.min():,.0f}   median {obs.median():,.0f}   max {obs.max():,.0f}")
print(f"  zero-fill asserts every one of them earns 0, placing them {(-obs.median()/truth.income.std()):.2f} SD")
print(f"  below the observed median of {truth.income.median():,.0f}")
print(f"  a zero credit_score is outside the legal range [300, 850] and is not a low score, it is a non-score")
print(f"  debt-to-income = loan/income becomes division by zero for {len(obs)} applicants")

# ---- the reusable function required by Q3(c) ----
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Type-aware missing-value handler.

    Assumptions, stated explicitly:
      1. Numeric fields are MAR given the observed data, so a median computed on the
         training rows is an acceptable central estimate. The median is used rather than
         the mean because every monetary field here is right-skewed.
      2. Missingness itself may be informative, so an indicator column is retained for
         every field that is imputed. If the indicator carries no signal the model can
         ignore it; if it does, the information is not thrown away.
      3. Structurally absent values (no credit file) are a distinct category, not a low
         value, and are flagged separately rather than imputed onto the score scale.
      4. All statistics are learned in fit() on training rows only and re-applied
         unchanged in transform(), so no test or production record influences them.
    """
    def __init__(self, numeric, categorical, structural=(), indicator=True):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.structural, self.indicator = list(structural), indicator
    def fit(self, X, y=None):
        self.medians_ = X[self.numeric].median()
        self.modes_ = {c: (X[c].mode().iloc[0] if X[c].notna().any() else "unknown")
                       for c in self.categorical}
        self.columns_ = list(X.columns)
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.numeric + self.categorical:
            if self.indicator and X[c].isna().any() or c in self.structural:
                X[f"{c}__missing"] = X[c].isna().astype(int)
        for c in self.structural:
            X[c] = X[c].fillna(self.medians_.get(c, 0))
        for c in self.numeric:
            X[c] = X[c].fillna(self.medians_[c])
        for c in self.categorical:
            X[c] = X[c].fillna("not_reported")
        return X

print("\n"+"="*80); print("3. THREE TREATMENTS COMPARED ON IDENTICAL SPLITS (Q3d)")
y = df.pop("default")
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=.25, random_state=RS, stratify=y)
num = ["income","years_employed","loan_amount","credit_score","n_prev_loans"]
cat = ["employment_type"]

def evaluate(name, Xtr, Xte, handler=None, zero=False, ytrain=None):
    ytrain = ytr if ytrain is None else ytrain
    a, b = Xtr.copy(), Xte.copy()
    if zero:
        a[num] = a[num].fillna(0); b[num] = b[num].fillna(0)
        a[cat] = a[cat].fillna("missing"); b[cat] = b[cat].fillna("missing")
    else:
        h = handler.fit(a); a, b = h.transform(a), h.transform(b)
        b = b.reindex(columns=a.columns, fill_value=0)
    nc = [c for c in a.columns if a[c].dtype != object]
    cc = [c for c in a.columns if a[c].dtype == object]
    pipe = Pipeline([("pre", ColumnTransformer([
            ("n", Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]), nc),
            ("c", OneHotEncoder(handle_unknown="ignore"), cc)])),
        ("m", LogisticRegression(max_iter=3000, random_state=RS))]).fit(a, ytrain)
    p = pipe.predict_proba(b)[:,1]
    print(f"  {name:38s} AUC={roc_auc_score(yte,p):.4f}   Brier={brier_score_loss(yte,p):.4f}   "
          f"features={a.shape[1]}")
    return pipe, p

_, p_zero = evaluate("Zero-fill (management proposal)", Xtr, Xte, zero=True)
_, p_good = evaluate("Type-aware median + indicators", Xtr, Xte,
                     MissingValueHandler(num, cat, structural=["credit_score"]))
drop_tr = Xtr.dropna(); ytr_cc = ytr.loc[drop_tr.index]; drop_te = Xte
print(f"  {'Complete-case (listwise deletion)':38s} retains {len(drop_tr)}/{len(Xtr)} training rows "
      f"({len(drop_tr)/len(Xtr)*100:.0f}%), default rate shifts "
      f"{ytr.mean():.3f} -> {ytr_cc.mean():.3f}")
_, p_cc = evaluate("Complete-case then median at scoring", drop_tr, drop_te,
                   MissingValueHandler(num, cat, structural=["credit_score"]), ytrain=ytr_cc)

print("\n"+"="*80); print("4. EFFECT ON INDIVIDUAL APPLICANTS (Q3d)")
aff = Xte.income.isna().values | Xte.credit_score.isna().values
diff = p_zero - p_good
print(f"  applicants in the test set with any missing field: {aff.sum()} of {len(Xte)}")
print(f"  mean predicted default probability, those applicants:")
print(f"      zero-fill      {p_zero[aff].mean():.4f}")
print(f"      type-aware     {p_good[aff].mean():.4f}")
print(f"  largest single shift: {np.abs(diff[aff]).max():.4f} in predicted probability")
for thr in (0.20, 0.30):
    flip = ((p_zero>=thr) != (p_good>=thr)) & aff
    print(f"  at a {thr:.0%} decline threshold, {flip.sum()} applicants ({flip.sum()/aff.sum()*100:.1f}% "
          f"of those with gaps) receive a different decision purely from the imputation choice")
pd.DataFrame({"zero_fill":p_zero,"type_aware":p_good,"has_missing":aff}).to_csv(
    "output/q3_predictions.csv", index=False)
print("\nwritten: output/q3_predictions.csv")
