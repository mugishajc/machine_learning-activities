# MIT91207 Machine Learning: Assignment 1

Eight questions covering problem formulation, preprocessing, missing data, feature
engineering, workflow design, feature selection, regularised regression and survey
analysis. University of Kigali, MSc Information Technology, Level 9, Trimester 4.

## Contents

| Path | Contents |
|---|---|
| `MIT91207_Assignment1_Mugisha_Jean_Claude_26016815.docx` | Submission document |
| `report/` | Report source |
| `code/` | One runnable script per question |
| `data/` | The two supplied datasets |
| `figures/` | Generated figures |
| `output/` | Console logs and result tables |

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

The seed is fixed at 42, so every figure and table in the report regenerates identically.

## Datasets

`data/synthetic_ETA.csv` holds 1,500 ride-hailing journeys and is used in Question 7.
`data/uscrime.txt` holds 47 US states, where `Crime` is offences per 100,000 population,
and is used in Question 6.

Questions 2, 3, 4, 5 and 8 supply no dataset. Each script builds a sample frame carrying
the faults its question describes, under a generating process written out in the code.
That keeps the ground truth known, so a proposed treatment can be shown to recover the
right answer.

<br>

<img src="assets/mugisha-jean-claude.jpg" width="130" alt="Mugisha Jean Claude">

**Mugisha Jean Claude**
Registration number 26016815
