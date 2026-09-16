#!/usr/bin/env bash
# Regenerates every result, table and figure in the report.
set -e
PY="${PYTHON:-python3}"
mkdir -p output figures
for s in q2_recommender_prep q3_missing_data q4_encoding q5_readmission \
         q5b_leakage_scaling q6_crime q7_regression q7b_complexity q8_survey_eda; do
  echo "=== running $s ==="
  "$PY" "code/$s.py" 2>&1 | tee "output/${s%%_*}_console.txt"
done
echo "done. figures in figures/, tables and logs in output/"
