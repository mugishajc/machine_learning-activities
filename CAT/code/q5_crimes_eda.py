"""
Question Five - exploratory analysis of crimes.csv for resource allocation.

The agency's stated constraint, that association must not be read as causation,
is not a caveat to add at the end. It decides which variables may enter the
analysis at all, and that is where this starts.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
pd.set_option("display.width", 200)
RS = 42
d = pd.read_csv("data/crimes.csv", low_memory=False)
TARGET = "crime_rate"

print("="*88); print("1. DATA QUALITY")
print(f"  communities {len(d):,}   variables {d.shape[1]}")
print(f"  missing cells {d.isna().sum().sum():,}   duplicate rows {d.duplicated().sum()}")
print(f"  non-numeric: {list(d.select_dtypes('object').columns)}   states represented: {d.state.nunique()}")
y = d[TARGET]
print(f"\n  target {TARGET} (offences per 100,000):")
print(f"    mean {y.mean():.1f}  median {y.median():.1f}  sd {y.std():.1f}  "
      f"skew {y.skew():.2f}  range {y.min():.1f} to {y.max():.1f}")
q1, q3 = y.quantile([.25, .75]); iqr = q3-q1
outl = ((y < q1-1.5*iqr) | (y > q3+1.5*iqr))
print(f"    zero-crime communities: {(y==0).sum()}   "
      f"Tukey outliers: {outl.sum()} ({outl.mean():.1%})")
print(f"    right skew of {y.skew():.2f} means the mean is pulled above the median by a")
print(f"    minority of high-crime communities; a log scale is the honest axis for plotting.")

# ------------------------------------------------------------------ variable groups
crime_cols = ["murders","rapes","robberies","assaults","burglaries","larcenies","autoTheft","arsons",
              "murdPerPop","rapesPerPop","robbbPerPop","assaultPerPop","burglPerPop","larcPerPop",
              "autoTheftPerPop","arsonsPerPop","nonViolPerPop","ViolentCrimesPerPop"]
crime_cols = [c for c in crime_cols if c in d.columns]
num = d.select_dtypes("number").columns.tolist()
socio = [c for c in num if c not in crime_cols + [TARGET, "fold"]]
print(f"\n  {len(crime_cols)} columns are other crime measures; {len(socio)} are socioeconomic "
      f"or demographic")

print("\n"+"="*88); print("2. WHAT CORRELATES WITH CRIME, AND WHY THE RANKING MISLEADS")
allc = d[num].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
top = allc.head(10)
print("  top 10 correlates of crime_rate, unrestricted:")
for k, v in top.items():
    tag = "OTHER CRIME MEASURE" if k in crime_cols else "socioeconomic"
    print(f"    {k:24s} {v:+.4f}   {tag}")
n_tauto = sum(1 for k in top.index if k in crime_cols)
print(f"\n  {n_tauto} of the top 10 are themselves crime counts or rates. Ranking features by")
print(f"  correlation selects the target restated, not anything that explains it.")

print("\n  top socioeconomic and demographic correlates, crime measures excluded:")
sc = d[socio+[TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
for k, v in sc.head(10).items():
    p = stats.pearsonr(d[k], y)[1]
    print(f"    {k:24s} {v:+.4f}   p = {p:.2e}")

print("\n"+"="*88); print("3. CORRELATION MATRIX AND ITS LIMITS")
sel = sc.head(9).index.tolist()
cm = d[sel+[TARGET]].corr()
print(cm.round(2).to_string())
red = []
for i, a in enumerate(sel):
    for b in sel[i+1:]:
        if abs(cm.loc[a, b]) >= 0.75: red.append((a, b, cm.loc[a, b]))
print(f"\n  predictor pairs correlated at |r| >= 0.75: {len(red)}")
for a, b, v in red[:6]: print(f"    {a:24s} <-> {b:24s} r = {v:+.3f}")

# linear vs monotone: where Pearson understates
print("\n  Pearson against Spearman, where the two disagree most:")
diff = []
for c in socio:
    pr = stats.pearsonr(d[c], y)[0]; sp = stats.spearmanr(d[c], y)[0]
    diff.append({"variable": c, "Pearson": pr, "Spearman": sp, "gap": abs(sp)-abs(pr)})
dd = pd.DataFrame(diff).sort_values("gap", ascending=False).head(6)
print(dd.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print("  A gap means the relationship is monotone but not straight, so a correlation")
print("  coefficient understates it and a linear model would miss part of it.")

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
a = ax[0,0]
a.hist(y, bins=60, color="#4393c3", edgecolor="white", linewidth=.4)
a.axvline(y.mean(), color="#b2182b", ls="--", lw=1.4, label=f"mean {y.mean():.0f}")
a.axvline(y.median(), color="k", ls=":", lw=1.4, label=f"median {y.median():.0f}")
a.set_xlabel("crime rate per 100,000"); a.set_ylabel("communities"); a.legend(fontsize=8)
a.set_title(f"(a) Continuous and right-skewed (skew {y.skew():.2f}), so a histogram\n"
            f"with mean and median shown, not a bar chart", fontsize=9)
a = ax[0,1]
v = sc.index[0]
a.scatter(d[v], y, s=9, alpha=.35, edgecolor="none", color="#2166ac")
b_, a_ = np.polyfit(d[v], y, 1); xs = np.linspace(d[v].min(), d[v].max(), 50)
a.plot(xs, a_+b_*xs, "r-", lw=1.4)
lo = stats.spearmanr(d[v], y)[0]
a.set_xlabel(v); a.set_ylabel("crime rate per 100,000")
a.set_title(f"(b) Two continuous variables, so a scatter with a fitted line.\n"
            f"Pearson {sc.iloc[0]:+.3f}, Spearman {lo:+.3f}", fontsize=9)
a = ax[1,0]
st = d.groupby("state")[TARGET].agg(["median","size"])
st = st[st["size"] >= 25].sort_values("median")
keep = st.index.tolist()
# tick labels are set separately: boxplot(labels=...) was renamed in matplotlib 3.9
# and removed in 3.11, so passing it breaks on newer installations
a.boxplot([d.loc[d.state==s, TARGET].values for s in keep], showfliers=False)
a.set_xticks(range(1, len(keep)+1))
a.set_xticklabels(keep, rotation=90, fontsize=7)
a.set_ylabel("crime rate per 100,000")
a.set_title("(c) A categorical grouping against a continuous outcome, so box plots\n"
            "showing spread within each state, not just a mean", fontsize=9)
a = ax[1,1]
im = a.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(len(cm))); a.set_xticklabels(cm.columns, rotation=90, fontsize=7)
a.set_yticks(range(len(cm))); a.set_yticklabels(cm.columns, fontsize=7)
for i in range(len(cm)):
    for j in range(len(cm)):
        a.text(j, i, f"{cm.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6,
               color="white" if abs(cm.iloc[i,j])>.6 else "black")
a.set_title("(d) Many numeric variables at once, so a correlation heatmap", fontsize=9)
fig.colorbar(im, ax=a, shrink=.75)
fig.suptitle("Q5: visualisations chosen from the measurement level of each variable", y=1.00)
fig.tight_layout(); fig.savefig("figures/q5_fig1.png", dpi=150, bbox_inches="tight"); plt.close(fig)
cm.to_csv("output/q5_correlation.csv"); sc.head(20).to_csv("output/q5_socio_correlates.csv")
print("\nfigure: figures/q5_fig1.png")
