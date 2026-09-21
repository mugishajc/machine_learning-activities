"""
Question Eight - reproducing the unbalanced clustering the scenario describes,
and showing what each remedy is worth.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
RS=42; rng=np.random.default_rng(RS); N=3000

# the scenario: one feature spanning 0 to ~20,000, another spanning 0 to 1
spend      = np.concatenate([rng.gamma(2.0, 900, int(N*.70)),
                             rng.gamma(6.0, 1200, int(N*.25)),
                             rng.gamma(9.0, 1800, N-int(N*.70)-int(N*.25))])
engagement = np.clip(rng.beta(2, 5, N), 0, 1)
tenure     = rng.gamma(3, 8, N)
noise      = rng.normal(0, 1, N)                       # weakly informative by construction
spend[rng.choice(N, 25, replace=False)] *= 9           # a few extreme customers
df = pd.DataFrame({"monthly_spend": spend, "engagement_rate": engagement,
                   "tenure_months": tenure, "survey_noise": noise})
print("="*80); print("1. THE DATA AS THE COMPANY SET IT UP")
print(df.describe().T[["mean","std","min","max"]].to_string(float_format=lambda v:f"{v:12.3f}"))
print(f"\n  spend range {df.monthly_spend.max():,.0f} against engagement range "
      f"{df.engagement_rate.max():.2f}: a ratio of about "
      f"{df.monthly_spend.max()/df.engagement_rate.max():,.0f} to 1")
print(f"  variance contributed by spend: "
      f"{df.monthly_spend.var()/df.var().sum():.6%} of the total across all four features")

def run(data, k, label):
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(data)
    sizes = pd.Series(km.labels_).value_counts(normalize=True).sort_values(ascending=False)
    sil = silhouette_score(data, km.labels_)
    print(f"  {label:44s} k={k}  largest cluster {sizes.iloc[0]:6.1%}  silhouette {sil:.4f}")
    return km, sizes, sil

print("\n"+"="*80); print("2. REPRODUCING THE FAILURE, THEN FIXING IT ONE STEP AT A TIME")
raw = df.values
run(raw, 3, "raw, unstandardised, all four features")
run(StandardScaler().fit_transform(df), 3, "standardised")
run(StandardScaler().fit_transform(df.drop(columns="survey_noise")), 3,
    "standardised, uninformative feature dropped")
q = df.monthly_spend.quantile(.99)
trimmed = df[df.monthly_spend <= q]
run(RobustScaler().fit_transform(trimmed.drop(columns="survey_noise")), 3,
    "robust-scaled, top 1% of spend trimmed")

print("\n"+"="*80); print("3. CHOOSING k: INERTIA AND SILHOUETTE DISAGREE, AS THE QUESTION SAYS")
Z = StandardScaler().fit_transform(df.drop(columns="survey_noise"))
rows=[]
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Z)
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(Z, km.labels_),
                 "largest cluster": pd.Series(km.labels_).value_counts(normalize=True).max()})
t = pd.DataFrame(rows)
t["inertia drop %"] = -t.inertia.pct_change()*100
print(t.to_string(index=False, float_format=lambda v: f"{v:12.4f}"))
print(f"\n  silhouette peaks at k={int(t.loc[t.silhouette.idxmax(),'k'])}; the largest single")
print(f"  improvement in inertia is at k={int(t.loc[t['inertia drop %'].idxmax(),'k'])}")
t.to_csv("output/q8_k_selection.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a=ax[0]
km_raw = KMeans(3, n_init=10, random_state=RS).fit(raw)
a.scatter(df.monthly_spend, df.engagement_rate, c=km_raw.labels_, s=6, cmap="viridis", alpha=.6)
a.set_xlabel("monthly spend"); a.set_ylabel("engagement rate")
a.set_title("(a) Raw features: spend alone decides every boundary", fontsize=10)
a=ax[1]
km_z = KMeans(3, n_init=10, random_state=RS).fit(Z)
a.scatter(df.monthly_spend, df.engagement_rate, c=km_z.labels_, s=6, cmap="viridis", alpha=.6)
a.set_xlabel("monthly spend"); a.set_ylabel("engagement rate")
a.set_title("(b) Standardised: both features contribute", fontsize=10)
a=ax[2]
a.plot(t.k, t.inertia, "o-", color="#2166ac"); a.set_xlabel("k"); a.set_ylabel("inertia", color="#2166ac")
a2=a.twinx(); a2.plot(t.k, t.silhouette, "s--", color="#b2182b"); a2.set_ylabel("silhouette", color="#b2182b")
a.set_title("(c) Elbow and silhouette need not agree", fontsize=10)
fig.tight_layout(); fig.savefig("figures/q8_fig1.png", dpi=150); plt.close(fig)
print("\nfigure: figures/q8_fig1.png")
