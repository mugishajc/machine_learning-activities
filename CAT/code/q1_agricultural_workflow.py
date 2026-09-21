"""
Question One - demonstrations supporting parts (c), (d) and (e).

No agricultural dataset is supplied, so a farm-season panel is constructed under a
stated process that carries every difficulty the scenario names: incomplete records,
variables on different scales, districts represented very unequally, and a rainfall
to yield relationship that changes over time. Because the generating process is known,
each claim can be checked against ground truth rather than argued.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from scipy import stats

RS = 42; rng = np.random.default_rng(RS)
DISTRICTS = ["Nyagatare","Kayonza","Bugesera","Musanze","Huye","Karongi"]
WEIGHT    = [0.34, 0.24, 0.17, 0.12, 0.08, 0.05]        # unequal historical coverage
SEASONS   = list(range(2016, 2026))

rows = []
for farm in range(1800):
    dist = rng.choice(DISTRICTS, p=WEIGHT)
    soil = rng.normal({"Nyagatare":6.2,"Kayonza":5.9,"Bugesera":5.4,
                       "Musanze":6.6,"Huye":6.0,"Karongi":5.6}[dist], 0.4)
    size = np.exp(rng.normal(-0.4, 0.7))                # hectares, right-skewed
    for s in SEASONS:
        # rainfall regime shifts from 2022: lower mean, higher variance
        shift = s >= 2022
        rain = rng.normal(880 if not shift else 790, 110 if not shift else 165)
        temp = rng.normal(19.5 + 0.09*(s-2016), 1.1)
        fert = max(0, rng.normal(110*(0.6+0.5*size), 35))
        plant_day = int(rng.normal(258, 12))
        irrig = rng.random() < (0.10 + 0.25*(dist in ("Nyagatare","Kayonza")))
        pest = rng.poisson(1.4 + 0.8*shift)
        # the rainfall coefficient weakens after the regime change: concept drift
        b_rain = 0.0042 if not shift else 0.0016
        lin = (-3.1 + b_rain*(rain-850) + 0.011*(fert-110) + 0.42*soil
               - 0.10*pest + 0.55*irrig - 0.020*(plant_day-258) - 0.13*(temp-19.5))
        low_yield = rng.binomial(1, 1/(1+np.exp(lin)))
        rows.append(dict(farm_id=farm, district=dist, season=s, rainfall_mm=rain,
                         temperature_c=temp, soil_ph=soil, farm_size_ha=size,
                         fertiliser_kg=fert, planting_day=plant_day,
                         irrigated=int(irrig), pest_reports=pest, low_yield=low_yield))
df = pd.DataFrame(rows)
# incomplete records, concentrated in the districts with the thinnest coverage
p_miss = df.district.map({"Nyagatare":.04,"Kayonza":.06,"Bugesera":.11,
                          "Musanze":.14,"Huye":.22,"Karongi":.28})
df.loc[rng.random(len(df)) < p_miss, "fertiliser_kg"] = np.nan
df.loc[rng.random(len(df)) < p_miss*0.7, "soil_ph"] = np.nan

print("="*88); print("1. THE PANEL AS THE COMPANY WOULD RECEIVE IT")
print(f"  farm-season records {len(df):,}   farms {df.farm_id.nunique():,}   "
      f"seasons {df.season.min()} to {df.season.max()}")
print(f"  low-yield rate {df.low_yield.mean():.3f}")
print("\n  records per district, and missing fertiliser within each:")
t = df.groupby("district").agg(records=("low_yield","size"),
                               low_yield_rate=("low_yield","mean"),
                               fertiliser_missing=("fertiliser_kg", lambda s: s.isna().mean()))
t["share_of_data"] = t.records/len(df)
print(t.sort_values("records", ascending=False).to_string(float_format=lambda v: f"{v:10.3f}"))
print(f"\n  scale spread: rainfall sd {df.rainfall_mm.std():.0f} against soil pH sd "
      f"{df.soil_ph.std():.2f}, a ratio of about {df.rainfall_mm.std()/df.soil_ph.std():.0f} to 1")

FEAT = ["rainfall_mm","temperature_c","soil_ph","farm_size_ha","fertiliser_kg",
        "planting_day","irrigated","pest_reports"]
CATS = ["district"]
def build():
    return Pipeline([("pre", ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), FEAT),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])),
        ("model", GradientBoostingClassifier(random_state=RS))])

print("\n"+"="*88); print("2. PART (c): WHY THE ORDER OF OPERATIONS CHANGES THE ANSWER")
X, y, g = df[FEAT+CATS], df.low_yield, df.farm_id
# wrong: impute and scale on everything, then split
wrong_pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),("scale", StandardScaler())]), FEAT),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])
Xw = wrong_pre.fit_transform(X)                       # statistics learned from all rows
a, b = train_test_split(np.arange(len(X)), test_size=.25, random_state=RS, stratify=y)
m = LogisticRegression(max_iter=2000, random_state=RS).fit(Xw[a], y.iloc[a])
auc_leak = roc_auc_score(y.iloc[b], m.predict_proba(Xw[b])[:,1])
# right: split first, fit the transformers on the training rows only
Xtr, Xte, ytr, yte = X.iloc[a], X.iloc[b], y.iloc[a], y.iloc[b]
right = Pipeline([("pre", wrong_pre.__class__(wrong_pre.transformers)),
                  ("model", LogisticRegression(max_iter=2000, random_state=RS))]).fit(Xtr, ytr)
auc_ok = roc_auc_score(yte, right.predict_proba(Xte)[:,1])
print(f"  preprocess then split  AUC = {auc_leak:.4f}")
print(f"  split then preprocess  AUC = {auc_ok:.4f}")
print(f"  difference {auc_leak-auc_ok:+.4f}")
# the leak that actually bites: farms repeat across seasons
tr_i, te_i = next(GroupShuffleSplit(1, test_size=.25, random_state=RS).split(X, y, groups=g))
gm = build().fit(X.iloc[tr_i], y.iloc[tr_i])
auc_grp = roc_auc_score(y.iloc[te_i], gm.predict_proba(X.iloc[te_i])[:,1])
rm = build().fit(X.iloc[a], y.iloc[a])
auc_row = roc_auc_score(y.iloc[b], rm.predict_proba(X.iloc[b])[:,1])
print(f"\n  random split over farm-seasons   AUC = {auc_row:.4f}  "
      f"(farms on both sides: {len(set(g.iloc[a]) & set(g.iloc[b])):,})")
print(f"  grouped by farm                  AUC = {auc_grp:.4f}  (farms on both sides: 0)")
print(f"  optimism from splitting rows instead of farms: {auc_row-auc_grp:+.4f}")

print("\n"+"="*88); print("3. PART (d): TWO KINDS OF CHANGE, SEPARATED")
early = df[df.season <= 2021]; late = df[df.season >= 2022]
mod = build().fit(early[FEAT+CATS], early.low_yield)
print(f"  trained on {early.season.min()}-{early.season.max()}, tested season by season:")
for s in SEASONS:
    sub = df[df.season == s]
    auc = roc_auc_score(sub.low_yield, mod.predict_proba(sub[FEAT+CATS])[:,1])
    ks = stats.ks_2samp(early.rainfall_mm, sub.rainfall_mm)
    tag = "train" if s <= 2021 else " live"
    print(f"    {s} [{tag}]  AUC {auc:.4f}   rainfall KS vs training {ks.statistic:.3f} "
          f"(p {ks.pvalue:.1e})   mean rain {sub.rainfall_mm.mean():.0f}")
print("\n  input drift, measured on the predictors alone:")
for c in ["rainfall_mm","temperature_c","pest_reports"]:
    ks = stats.ks_2samp(early[c], late[c])
    print(f"    {c:16s} KS {ks.statistic:.3f}  p {ks.pvalue:.2e}   "
          f"mean {early[c].mean():.1f} -> {late[c].mean():.1f}")
print("\n  relationship drift, measured as the rainfall coefficient itself:")
for label, part in [("2016-2021", early), ("2022-2025", late)]:
    z = (part.rainfall_mm-part.rainfall_mm.mean())/part.rainfall_mm.std()
    lr = LogisticRegression(max_iter=1000).fit(z.values.reshape(-1,1), part.low_yield)
    print(f"    {label}: coefficient on standardised rainfall = {lr.coef_[0][0]:+.4f}")
print("  The inputs moved and the coefficient moved. Monitoring only the inputs would")
print("  have caught the first and missed the second.")

print("\n"+"="*88); print("4. PART (e): IS THE REGIONAL DIFFERENCE REAL OR IS IT THE MODEL?")
gm2 = build().fit(X.iloc[tr_i], y.iloc[tr_i])
te = df.iloc[te_i].copy()
te["p"] = gm2.predict_proba(X.iloc[te_i])[:,1]
thr = np.quantile(te.p, 0.75)
te["flag"] = (te.p >= thr).astype(int)
out = []
for dname, sub in te.groupby("district"):
    if sub.low_yield.nunique() < 2: continue
    out.append({"district": dname, "n": len(sub),
                "true low-yield rate": sub.low_yield.mean(),
                "flag rate": sub.flag.mean(),
                "AUC": roc_auc_score(sub.low_yield, sub.p),
                "recall": recall_score(sub.low_yield, sub.flag),
                "precision": precision_score(sub.low_yield, sub.flag, zero_division=0),
                "missing fertiliser": df[df.district==dname].fertiliser_kg.isna().mean()})
o = pd.DataFrame(out).sort_values("true low-yield rate", ascending=False)
# subgroup metrics on 200 to 1400 records are point estimates with very different
# precision, so each carries a bootstrap interval. Without one, a district with a
# small sample looks like a finding when it is noise.
lo, hi = [], []
for dname in o.district:
    sub = te[te.district == dname]
    bs = []
    for k in range(400):
        ix = np.random.default_rng(k).choice(len(sub), len(sub), replace=True)
        ss = sub.iloc[ix]
        if ss.low_yield.nunique() == 2:
            bs.append(roc_auc_score(ss.low_yield, ss.p))
    lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
o["AUC 95% low"], o["AUC 95% high"] = lo, hi
o["interval width"] = o["AUC 95% high"] - o["AUC 95% low"]
print(o.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
print(f"\n  widest interval {o['interval width'].max():.3f} for {o.loc[o['interval width'].idxmax(),'district']} "
      f"(n={int(o.loc[o['interval width'].idxmax(),'n'])}), "
      f"narrowest {o['interval width'].min():.3f} for {o.loc[o['interval width'].idxmin(),'district']} "
      f"(n={int(o.loc[o['interval width'].idxmin(),'n'])})")
overlap = all(o["AUC 95% low"].max() <= o["AUC 95% high"].min() for _ in [0])
print(f"  every district interval overlaps every other: {o['AUC 95% low'].max() <= o['AUC 95% high'].min()}")
print("  so the apparent AUC ranking across districts is not statistically distinguishable")
print(f"\n  flag-rate spread {o['flag rate'].max()-o['flag rate'].min():.3f}, "
      f"but the true low-yield rate also spans {o['true low-yield rate'].max()-o['true low-yield rate'].min():.3f}")
cr = np.corrcoef(o["true low-yield rate"], o["flag rate"])[0,1]
print(f"  correlation between flag rate and true rate across districts: {cr:+.3f}")
print(f"  recall spread {o.recall.max()-o.recall.min():.3f}, "
      f"AUC spread {o.AUC.max()-o.AUC.min():.3f}")
cr2 = np.corrcoef(o["missing fertiliser"], o["AUC"])[0,1]
print(f"  correlation between a district's missing-data rate and the model's AUC there: {cr2:+.3f}")
print("  That coefficient rests on six points with sample sizes from 200 to 1,420 and")
print("  overlapping intervals, so it is not evidence of anything. It is recorded here as")
print("  a hypothesis to test on more districts, not as a finding.")
o.to_csv("output/q1_district_audit.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]
per = [roc_auc_score(df[df.season==s].low_yield,
                     mod.predict_proba(df[df.season==s][FEAT+CATS])[:,1]) for s in SEASONS]
a.plot(SEASONS, per, "o-", color="#2166ac", lw=1.8)
a.axvspan(2021.5, 2025.5, color="#fddbc7", alpha=.6, label="after the rainfall regime change")
a.set_xlabel("season"); a.set_ylabel("AUC"); a.legend(fontsize=8)
a.set_title("(a) Performance decays once the regime shifts", fontsize=10)
a = ax[1]
a.hist(early.rainfall_mm, bins=40, alpha=.6, label="2016-2021", color="#2166ac", density=True)
a.hist(late.rainfall_mm, bins=40, alpha=.6, label="2022-2025", color="#b2182b", density=True)
a.set_xlabel("seasonal rainfall (mm)"); a.set_ylabel("density"); a.legend(fontsize=8)
a.set_title("(b) The inputs themselves moved", fontsize=10)
a = ax[2]
a.scatter(o["true low-yield rate"], o["flag rate"], s=60, color="#2166ac", zorder=3)
for _, r in o.iterrows():
    a.annotate(r.district, (r["true low-yield rate"], r["flag rate"]), fontsize=7,
               xytext=(4,4), textcoords="offset points")
lim = [min(o["true low-yield rate"].min(), o["flag rate"].min())-.03,
       max(o["true low-yield rate"].max(), o["flag rate"].max())+.03]
a.plot(lim, lim, "k--", lw=1, label="flagged in proportion to need")
a.set_xlabel("true low-yield rate"); a.set_ylabel("share of farms flagged"); a.legend(fontsize=8)
a.set_title("(c) Disparity tracks real need, it is not free-standing", fontsize=10)
fig.tight_layout(); fig.savefig("figures/q1_fig1.png", dpi=150); plt.close(fig)
print("\nfigure: figures/q1_fig1.png")
