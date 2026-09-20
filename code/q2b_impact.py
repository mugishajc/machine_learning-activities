"""
Q2(a) - measuring how the preprocessing decisions change the model.

The 15-row frame in q2_recommender_prep.py shows the mechanics of each operation.
It is too small to show what any of them is worth. This builds a realistic log,
injects the four faults the question names, and prices each fix against a
downstream task: predicting whether a learner will complete a recommended course.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RS = 42; rng = np.random.default_rng(RS)
N_LEARNERS, N_COURSES = 500, 40

cats = ["Data Science", "AI/ML", "Databases", "Web Development", "Networking"]
courses = pd.DataFrame({
    "course_title": [f"Course {i:02d}" for i in range(N_COURSES)],
    "course_category": rng.choice(cats, N_COURSES),
    "course_length_min": rng.integers(40, 1500, N_COURSES)})
learners = pd.DataFrame({
    "learner_id": np.arange(N_LEARNERS),
    "diligence": rng.beta(2.2, 2.2, N_LEARNERS),          # unobserved trait
    "courses_taken_before": rng.poisson(3.0, N_LEARNERS)})

rows = []
for _, L in learners.iterrows():
    for c in rng.choice(N_COURSES, rng.integers(3, 12), replace=False):
        C = courses.iloc[c]
        engagement = np.clip(rng.normal(L.diligence, 0.22), 0.02, 1.6)
        rows.append({"learner_id": int(L.learner_id), "course_title": C.course_title,
                     "course_category": C.course_category,
                     "minutes_watched": float(engagement * C.course_length_min),
                     "course_length_min": int(C.course_length_min),
                     "courses_taken_before": int(L.courses_taken_before),
                     "diligence": L.diligence, "engagement": engagement})
df = pd.DataFrame(rows)
p_complete = 1/(1+np.exp(-(-1.5 + 3.4*df.engagement + 0.9*df.diligence
                           - 0.0004*df.course_length_min + rng.normal(0,.35,len(df)))))
df["completed"] = rng.binomial(1, p_complete)
# rating exists only when the learner engaged enough to form a view
df["rating"] = np.clip(rng.normal(2.0 + 2.6*df.engagement, 0.6), 1, 5).round(1)
df["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(
    rng.integers(0, 540, len(df)), unit="D")
df = df.drop(columns=["diligence", "engagement"])

clean = df.copy()                                   # ground truth, never shown to the model
# ---- inject the four faults the question names ----
dup = df.sample(frac=0.08, random_state=RS)
df = pd.concat([df, dup], ignore_index=True)                        # duplicated records
variants = {"Data Science": ["data science", "DATA SCIENCE", "Data  Science", "Data Science"],
            "AI/ML": ["ai/ml", "AI / ML", "AI/ML"], "Databases": ["databases", "Databases"],
            "Web Development": ["web development", "Web Development"],
            "Networking": ["networking", "Networking"]}
df["course_category"] = [rng.choice(variants[c]) for c in df.course_category]  # inconsistent labels
miss = rng.random(len(df)) < (0.55 - 0.40*(df.minutes_watched/df.course_length_min).clip(0,1))
df.loc[miss, "rating"] = np.nan                                     # MNAR: disengaged do not rate
print("="*82); print("1. THE LOG AS RECEIVED")
print(f"  interactions {len(df)}   learners {df.learner_id.nunique()}   courses {df.course_title.nunique()}")
print(f"  duplicate rows            : {df.duplicated().sum()}")
print(f"  distinct category strings : {df.course_category.nunique()} for {len(cats)} real categories")
print(f"  missing ratings           : {df.rating.isna().sum()} ({df.rating.isna().mean():.1%})")
print(f"  completion rate when rating present {df.completed[~df.rating.isna()].mean():.3f} "
      f"vs missing {df.completed[df.rating.isna()].mean():.3f}")
print(f"  scale spread: minutes sd {df.minutes_watched.std():.0f} vs rating sd {df.rating.std():.2f}")

cut = df.timestamp.quantile(0.75)
def evaluate(label, frame, harmonise, impute, indicator, features, scale):
    d = frame.copy()
    if harmonise:
        d["course_category"] = (d.course_category.str.strip().str.lower()
                                 .str.replace("/", " ", regex=False).str.split().str.join(" "))
    tr, te = d[d.timestamp <= cut].copy(), d[d.timestamp > cut].copy()
    if impute == "drop":
        tr = tr[tr.rating.notna()]
    num = ["minutes_watched", "courses_taken_before", "course_length_min", "rating"]
    for part in (tr, te):
        if indicator: part["rating_missing"] = part.rating.isna().astype(int)
        if features:
            med = tr.groupby("course_title").minutes_watched.median()
            part["engagement_ratio"] = part.minutes_watched / part.course_title.map(med).fillna(
                tr.minutes_watched.median())
    cols = num + (["rating_missing"] if indicator else []) + (["engagement_ratio"] if features else [])
    fill = "median" if impute in ("median", "drop") else "mean"
    steps = [("i", SimpleImputer(strategy=fill))] + ([("s", StandardScaler())] if scale else [])
    pipe = Pipeline([("pre", ColumnTransformer([
            ("n", Pipeline(steps), cols),
            ("c", OneHotEncoder(handle_unknown="ignore"), ["course_category"])])),
        ("m", LogisticRegression(max_iter=4000, random_state=RS))])
    pipe.fit(tr[cols+["course_category"]], tr.completed)
    auc = roc_auc_score(te.completed, pipe.predict_proba(te[cols+["course_category"]])[:,1])
    width = pipe.named_steps["pre"].transform(tr[cols+["course_category"]]).shape[1]
    print(f"  {label:44s} AUC={auc:.4f}  train rows={len(tr):5d}  width={width}")
    return auc, label

print("\n"+"="*82); print("2. WHAT EACH DECISION IS WORTH (temporal split, identical model)")
res = []
res.append(evaluate("Raw log, no cleaning at all", df, False, "mean", False, False, False))
res.append(evaluate("Drop rows with a missing rating", df, True, "drop", False, False, True))
res.append(evaluate("Mean-impute ratings, no indicator", df, True, "mean", False, False, True))
res.append(evaluate("Median-impute + missingness indicator", df, True, "median", True, False, True))
res.append(evaluate("Full pipeline (+ engagement_ratio)", df.drop_duplicates(), True, "median", True, True, True))
# Rows 3 and 4 above differ in two ways at once, so the indicator is isolated here by
# holding the imputation rule fixed and toggling only the indicator.
print("\n  isolated ablation, imputation held fixed:")
a,_ = evaluate("    median impute, no indicator", df, True, "median", False, False, True)
b,_ = evaluate("    median impute, with indicator", df, True, "median", True, False, True)
c,_ = evaluate("    mean impute, no indicator", df, True, "mean", False, False, True)
d,_ = evaluate("    mean impute, with indicator", df, True, "mean", True, False, True)
print(f"    indicator effect, median imputation fixed : {b-a:+.4f}")
print(f"    indicator effect, mean imputation fixed   : {d-c:+.4f}")
print(f"    median vs mean imputation, no indicator   : {a-c:+.4f}")

best = max(res)[0]
print(f"\n  cleaning the log is worth {best-res[0][0]:+.4f} AUC over using it raw")
print(f"  the missingness indicator, properly isolated, is worth {b-a:+.4f} AUC")
print(f"  dropping incomplete rows costs {res[3][0]-res[1][0]:+.4f} AUC against imputing them")

fig, ax = plt.subplots(figsize=(10, 4.6))
labels = [r[1] for r in res]; vals = [r[0] for r in res]
colors = ["#b2182b","#ef8a62","#f7f7f7","#67a9cf","#2166ac"]
b = ax.barh(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=.6)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0.5, max(vals)+0.03); ax.invert_yaxis(); ax.set_xlabel("AUC on the later time window")
for i, v in enumerate(vals): ax.text(v+0.002, i, f"{v:.4f}", va="center", fontsize=8)
ax.set_title("Q2 Fig 1: each preprocessing decision priced against the downstream model")
fig.tight_layout(); fig.savefig("figures/q2_fig1_impact.png", dpi=150); plt.close(fig)
pd.DataFrame({"strategy": labels, "auc": vals}).to_csv("output/q2b_impact.csv", index=False)
print("\nfigure: figures/q2_fig1_impact.png")
