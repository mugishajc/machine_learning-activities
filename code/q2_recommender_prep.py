"""
Q2(b)(c)(d) - preprocessing pipeline for the course-recommendation dataset.
A sample frame is constructed carrying every fault named in the question:
missing ratings, duplicate records, inconsistent category labels, and
numeric variables on very different scales.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
pd.set_option("display.width", 200)
RS = 42; rng = np.random.default_rng(RS)

raw = pd.DataFrame({
 "learner_id":[101,101,102,103,104,104,105,106,107,108,109,101,110,111,112],
 "course_title":["Intro to Python","Intro to Python","Data Viz","Intro to Python","Deep Learning",
                 "Deep Learning","Data Viz","SQL Basics","Deep Learning","SQL Basics","Intro to Python",
                 "Data Viz","SQL Basics","Deep Learning","Data Viz"],
 "course_category":["Data Science"," data science","DATA SCIENCE","Data  Science","AI/ML","ai/ml",
                    "Data Science","Databases","AI / ML","databases","Data Science","DATA SCIENCE",
                    "Databases","ai/ml","Data Science"],
 "rating":[4.5,4.5,np.nan,3.0,5.0,5.0,np.nan,2.5,4.0,3.5,4.0,4.0,np.nan,4.5,3.5],
 "completion_status":["completed","completed","dropped","completed","completed","completed",
                      "in_progress","dropped","completed","completed","completed","completed",
                      "in_progress","completed","dropped"],
 "timestamp":pd.to_datetime(["2026-01-05","2026-01-05","2026-01-11","2026-02-02","2026-02-14",
                             "2026-02-14","2026-03-01","2026-03-08","2026-03-19","2026-04-02",
                             "2026-04-15","2026-04-21","2026-05-03","2026-05-17","2026-06-01"]),
 "minutes_watched":[320,320,45,610,1450,1450,90,30,1180,240,505,275,60,1320,150],
 "courses_taken_before":[0,0,3,1,7,7,2,0,5,1,4,2,1,6,3],
})

print("="*80); print("BEFORE CLEANING")
print(raw.to_string(index=False))
print(f"\nrows={len(raw)}  exact duplicates={raw.duplicated().sum()}  "
      f"missing ratings={raw.rating.isna().sum()}  "
      f"distinct category strings={raw.course_category.nunique()}")
print("raw category labels:", sorted(raw.course_category.unique()))
print("\nScale disparity (raw units):")
print(raw[["rating","minutes_watched","courses_taken_before"]].describe().T[["mean","std","min","max"]]
      .to_string(float_format=lambda v: f"{v:10.2f}"))

df = raw.copy()

# 1. duplicates: exact, then business-key duplicates
df = df.drop_duplicates()
before_key = len(df)
df = df.sort_values("timestamp").drop_duplicates(subset=["learner_id","course_title"], keep="last")
print("\n"+"="*80); print("STEP 1  de-duplication")
print(f"  exact duplicates removed : {len(raw)-before_key}")
print(f"  repeat learner-course rows removed (kept most recent): {before_key-len(df)}")

# 2. category harmonisation
def norm(s):
    s = s.strip().lower().replace("/", " ").replace("-", " ")
    s = " ".join(s.split())
    return {"data science":"Data Science","ai ml":"AI/ML","databases":"Databases"}.get(s, s.title())
df["course_category"] = df.course_category.map(norm)
print("\n"+"="*80); print("STEP 2  category harmonisation")
print(f"  {raw.course_category.nunique()} raw strings -> {df.course_category.nunique()} canonical: "
      f"{sorted(df.course_category.unique())}")

# 3. missing ratings: indicator + group median, fitted on training rows only
df = df.sort_values("timestamp").reset_index(drop=True)
cut = df.timestamp.quantile(0.7)
train_mask = df.timestamp <= cut
df["rating_missing"] = df.rating.isna().astype(int)
med_by_cat = df.loc[train_mask].groupby("course_category").rating.median()
global_med = df.loc[train_mask].rating.median()
df["rating_filled"] = df.rating.fillna(df.course_category.map(med_by_cat)).fillna(global_med)
print("\n"+"="*80); print("STEP 3  missing ratings")
print(f"  cut-off for statistics: {cut.date()}  (train rows only, no future information)")
print(f"  category medians from training rows: {med_by_cat.round(2).to_dict()}")
print(f"  imputed {df.rating_missing.sum()} ratings; 'rating_missing' retained as a signal "
      "(non-response often means disengagement, not a neutral score)")

# 4. engineered features (Q2c)
df["engagement_ratio"] = (df.minutes_watched / df.groupby("course_title")
                          .minutes_watched.transform("median")).round(3)
df["is_completed"] = (df.completion_status == "completed").astype(int)
df = df.sort_values(["learner_id","timestamp"])
df["prior_completion_rate"] = (df.groupby("learner_id").is_completed
                               .transform(lambda s: s.shift().expanding().mean())).fillna(0).round(3)
df["experience_band"] = pd.cut(df.courses_taken_before, [-1,0,2,5,np.inf],
                               labels=["new","light","regular","heavy"])
print("\n"+"="*80); print("STEP 4  engineered features")
print("  engagement_ratio      = learner minutes / median minutes for that course")
print("  prior_completion_rate = share of the learner's EARLIER courses completed (shift() before expanding())")
print("  experience_band       = banded courses_taken_before")

# 5. scaling, fitted on training rows only
num = ["rating_filled","minutes_watched","courses_taken_before","engagement_ratio"]
sc = StandardScaler().fit(df.loc[train_mask, num])
df[[c+"_z" for c in num]] = sc.transform(df[num])
print("\n"+"="*80); print("STEP 5  scaling (StandardScaler fitted on training rows only)")
print(pd.DataFrame({"variable":num,"train mean":sc.mean_.round(2),"train scale":sc.scale_.round(2)})
      .to_string(index=False))

print("\n"+"="*80); print("AFTER CLEANING")
cols = ["learner_id","course_title","course_category","rating","rating_filled","rating_missing",
        "engagement_ratio","prior_completion_rate","experience_band"]
print(df[cols].to_string(index=False))
model_cols=[c for c in cols if c!="rating"]
print(f"\nrows {len(raw)} -> {len(df)}   missing values in modelling columns: {df[model_cols].isna().sum().sum()}"
      f"   (raw rating column retained with {int(df.rating.isna().sum())} NaN for audit)")

# Q2(d) temporal integrity check
print("\n"+"="*80); print("Q2(d)  LEAKAGE CHECK")
tr, te = df[df.timestamp<=cut], df[df.timestamp>cut]
print(f"  train window {tr.timestamp.min().date()} .. {tr.timestamp.max().date()}  (n={len(tr)})")
print(f"  test  window {te.timestamp.min().date()} .. {te.timestamp.max().date()}  (n={len(te)})")
print(f"  overlap: {'NONE' if tr.timestamp.max() < te.timestamp.min() else 'PRESENT - leakage'}")
bad = df.groupby("learner_id").apply(
    lambda g: (g.prior_completion_rate.notna() &
               (g.index.to_series().rank()==1) & (g.prior_completion_rate!=0)).any(), include_groups=False)
print(f"  first-ever record for any learner carrying non-zero history: {bool(bad.any())} (must be False)")
df.to_csv("output/q2_cleaned.csv", index=False)
print("\nwritten: output/q2_cleaned.csv")
