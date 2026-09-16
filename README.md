# MIT91207 Machine Learning: Assignment 1

**Mugisha Jean Claude**, registration number **26016815**.
MSc Information Technology, Level 9, Trimester 4. University of Kigali.
Lecturer: Dr Gustave Udahemuka. Submission date: 17 September 2026.

Eight questions covering problem formulation, preprocessing, missing data, feature
engineering, end-to-end workflow design, feature selection, regularised regression and
exploratory survey analysis.

## Contents

| Path | Contents |
|---|---|
| `report/` | Full written report, Markdown source |
| `MIT91207_Assignment1_Mugisha_Jean_Claude_26016815.docx` | Submission document |
| `code/` | One runnable script per question |
| `data/` | The two supplied datasets |
| `figures/` | 11 generated figures |
| `output/` | Console logs and result tables for every question |

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Seed is fixed at 42 throughout, so every figure and table in the report regenerates
identically. Console output is written to `output/*_console.txt`.

## Datasets

`data/synthetic_ETA.csv` holds 1,500 ride-hailing journeys with 15 variables and is used
in Question 7. `data/uscrime.txt` holds 47 US states with 16 variables, where `Crime` is
offences per 100,000 population, and is used in Question 6. Questions 2, 3, 4, 5 and 8
supply no dataset, so each script constructs a sample frame carrying the faults the
question describes, under a generating process written out in the script. That choice is
deliberate: it makes the ground truth known, so a proposed treatment can be shown to
recover the right answer.

## Selected findings

| Question | Finding |
|---|---|
| 3 | Zero-filling missing values costs 9.2 AUC points and changes the lending decision for 33% of applicants with data gaps at a 20% threshold |
| 4 | Encoding nominal variables as ordinal integers costs 8.9 AUC points under logistic regression but only 1.6 under a random forest |
| 5 | Optimism from a patient-level split grows with records per patient, measured at +0.006 to +0.012 AUC across 12 seeds |
| 6 | `Po1` and `Po2` correlate at 0.994 and OLS assigns them opposite signs, +566.87 and −302.69, which ridge corrects to +124.47 and +104.60 |
| 7 | OLS, ridge and lasso are indistinguishable at n=1,200. The one-standard-error rule removes 7 of 15 predictors for 0.13 minutes of RMSE |
