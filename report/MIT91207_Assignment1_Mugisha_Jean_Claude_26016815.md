# MIT91207 Machine Learning: Assignment 1

**Name:** Mugisha Jean Claude
**Registration number:** 26016815
**Programme:** MSc Information Technology, Level 9, Trimester 4
**Module:** MIT91207 Machine Learning
**Lecturer:** Dr Gustave Udahemuka
**Submission date:** 17 September 2026

## Note on reproducibility

Every number, table and figure in this report was produced by the scripts in `code/`
and can be regenerated in full. Nothing is quoted from memory or from a textbook.

| Item | Detail |
|---|---|
| Python | 3.9.6, virtual environment `mluok` |
| Libraries | pandas 2.3.3, scikit-learn 1.6.1, statsmodels, scipy, matplotlib |
| Random seed | 42 throughout, so all results are reproducible |
| Q6 data | `data/uscrime.txt`, 47 states, 16 variables, no missing values |
| Q7 data | `data/synthetic_ETA.csv`, 1,500 journeys, 15 variables, no missing values |
| Other questions | Data generated under a stated process, since no dataset was supplied |

To reproduce: `pip install -r requirements.txt` then `bash run_all.sh`. Console
output for each question is preserved in `output/*_console.txt`.

Where a question supplies no dataset (Questions 2, 3, 4, 5 and 8), a sample frame is
constructed that carries exactly the faults the question describes. The generating
process is written out in the script, so the conclusions can be checked against a
known ground truth. This is deliberate: it is the only way to show that a proposed
treatment recovers the right answer, because with real data the right answer is unknown.

# Question 1: Student early-warning system (12 marks)

## 1(a) Formulating the problem (5 marks)

The university has not asked for a prediction. It has asked for a way to decide which
students an advisory team should contact next week. That distinction governs everything
that follows, because the quantity being optimised is the quality of a ranking under a
fixed capacity constraint, not the accuracy of a label.

The target variable. "Poor academic outcome" is not yet a variable. It has to be
made one, and the choice is a policy decision and not a technical one. Three
defensible definitions are available, and they are not equivalent:

| Definition | Type | What it supports |
|---|---|---|
| Final module mark | Continuous, 0 to 100 | How far below the line a student is likely to fall |
| Pass or fail at the 50% threshold | Binary | Whether to intervene at all |
| Progression outcome at end of year | Binary, longer horizon | Retention planning |

The prediction horizon is part of the problem definition. A model that predicts the
final mark using the final coursework score is useless, because by the time that score
exists the outcome is already settled. The system must be specified as: given only the
information observable at week *t*, predict the outcome at end of module. This makes the
feature set a function of *t* and forces attendance, learning-management-system
engagement and early assessment marks to be truncated at the prediction date. Section
2(d) of this report treats the same problem in a recommendation setting.

My recommendation is to treat it as both, with the regression as the primary model.
The underlying quantity, the final mark, is continuous and the dichotomy at 50% is an
administrative convention applied to it. Fitting the continuous outcome and then
thresholding preserves information that a direct classifier discards. A student
predicted at 49% and a student predicted at 18% are both classified "at risk", but they
require entirely different interventions, and the classifier cannot tell them apart. The
regression output also supports a conversation the classifier cannot support, namely what
mark the student needs on the remaining assessment to recover.

The classification framing is still required, for two reasons. The decision itself is
binary, so a threshold has to exist somewhere. And a calibrated probability of failure is
the correct input to a capacity-constrained triage rule, because it can be combined with
the cost of a missed intervention to produce a defensible cut-off. The practical design
is therefore a regression model on the mark, plus a calibrated classifier used for
ranking, with the two reconciled at the point of decision.

What the university is really buying is a ranking. With, say, four advisors and 900
students, only the top forty or so cases will ever be contacted. A model with modest
overall accuracy but a well-ordered top decile is more useful than an accurate model that
mixes the top decile, and this is why the evaluation in 1(c) and in Question 5 is framed
around ranking quality and subgroup behaviour, not raw accuracy.

## 1(b) Two modelling approaches (5 marks)

Classification approach: penalised logistic regression, moving to gradient boosting if
the data justifies it.

Logistic regression is the right starting point here, not a fallback. It produces a
probability that is directly interpretable as a risk score, the coefficients can be shown
to an academic board that will reasonably ask why a particular student was flagged, and
with L2 penalisation it behaves stably when attendance and engagement measures are
correlated, which they invariably are. Question 7 of this report demonstrates that
penalisation mainly buys coefficient stability more than accuracy, and that is what is wanted when the output has to be defended.

How the prediction supports the decision: the model returns P(fail) for every enrolled
student each week. Advisors receive a ranked list with the probability attached, banded
into three tiers. Above 0.70, contact this week. Between 0.40 and 0.70, monitor and
contact if the trend worsens. Below 0.40, no action. The bands are set from the advisory
capacity and the relative cost of the two errors, not from the default 0.50 cut-off,
which corresponds to no particular institutional preference. Because the probability is
calibrated, "seventy per cent" can be checked against the observed failure rate of
students given that score, and the board can audit that claim.

Regression approach: gradient-boosted regression trees on the final mark.

The relationship between engagement and attainment is not linear. Attendance rising from
30% to 50% matters a great deal, and from 80% to 95% matters very little. A boosted tree
model captures that saturation without anyone having to specify it, and it handles the
interaction between prior grades and current engagement that a linear model would need to
be told about explicitly. Ridge regression is the appropriate comparator, and on a
first-year cohort with few features it will often match the boosted model, a result worth knowing before committing to the more complex option.

How the prediction supports the decision: the model returns a predicted mark with an
interval. The actionable quantity is the predicted distance from the pass mark. A student
predicted at 47 plus or minus 6 is a different case from one predicted at 25 plus or minus
6, even though both are below the line, and the intervals let an advisor distinguish a
confident prediction from an uncertain one. The intervals matter more than the point
estimate for a borderline student, because that is where the intervention decision is
genuinely difficult.

Why run both. They fail differently. If the classifier flags a student the regression
places at 62, that disagreement is itself a signal, usually that the student's pattern is
unusual relative to the training cohort. Routing disagreements to human review is a
cheap and effective safeguard.

## 1(c) Two factors beyond predictive accuracy (2 marks)

Subgroup performance, and the asymmetry of the two errors. Aggregate accuracy is an
average, and averages conceal exactly the failures that matter here. Question 5 of this
report measures this directly on a comparable problem: an overall AUC of 0.731 was
composed of subgroup AUCs ranging from 0.695 to 0.763, and subgroup recall ranged from
0.275 to 0.529. A 25 percentage point spread in recall means the system finds half the
at-risk students in one group and a quarter in another, while reporting a single
respectable headline number. In this university setting the groups that matter are likely
to be part-time students, students entering through non-standard qualifications, and
students whose engagement is off-campus and therefore under-recorded by the
learning-management system. The two errors are also not symmetric. A false negative is a
student who fails without ever being contacted. A false positive costs an advisor twenty
minutes. Accuracy weights these equally, an indefensible weighting, so the model must be
evaluated on recall within each subgroup at the operating threshold, and the threshold
itself must be justified against that asymmetry.

The gap between prediction and action, and the feedback loop the system creates. The
model finds association, not cause. Low learning-management-system engagement predicts
failure, but it is largely a symptom. A student in financial difficulty, working night
shifts, will show low engagement, and an intervention aimed at raising their click rate
addresses the indicator and not the cause. The features must therefore be
interrogated for whether they are actionable at all before the system is allowed to drive
interventions, a point developed at length in Question 6(d). There is also a loop that
most deployments overlook: once interventions begin, a successful intervention makes the
prediction wrong, so the model's apparent accuracy degrades precisely when it is working.
Without recording which students were contacted and treating that as a variable, the
retraining cycle will progressively unlearn the signal that made the system useful. In
addition, a student labelled "at risk" may be treated differently by staff and may
internalise the label, so the prediction can contribute to the outcome it forecasts.
Under Rwanda's Law No. 058/2021 relating to the protection of personal data and privacy,
profiling students in this way also carries transparency and lawful-basis obligations
that need to be settled before deployment, not after.

# Question 2: Course-recommendation preprocessing (13 marks)

## 2(a) Preprocessing pipeline design (5 marks)

The four faults the question lists are not equally serious, and treating them as a
uniform cleaning checklist is the main risk. Ordering matters, because several of the
operations interact.

Order of operations, and why.

1. **Duplicates first.** Deduplicating after computing statistics would mean the
   statistics were computed on inflated counts. Two kinds exist here and they need
   different treatment. Exact repeated rows are ingestion artefacts and can be dropped
   outright. Repeated learner-course pairs with different timestamps are not errors at
   all, they are re-enrolments, and dropping them blindly destroys real behaviour. In the
   sample frame this distinction removed 2 rows as re-enrolments while retaining the most
   recent record for each pair.
2. **Category harmonisation second,** because grouped statistics computed later depend on
   the grouping key being correct. `"Data Science"`, `" data science"`, `"DATA SCIENCE"`
   and `"Data  Science"` are one category recorded four ways. Left alone they become four
   one-hot columns, each with a quarter of the support, and the model sees four rare
   categories instead of one common one. In the sample frame, 9 distinct raw strings
   collapse to 3 genuine categories.
3. **Missing ratings third.** The critical judgement is that a missing rating is not a
   missing number, it is an absent behaviour. Learners who abandon a course rarely rate
   it. Imputing the mean assigns an average opinion to people who were too indifferent to
   express one, which biases the recommender towards courses that are quietly disliked.
   The treatment is to impute a central value so the row remains usable, and to retain a
   binary `rating_missing` indicator so the model can learn what the absence itself
   means. Question 3 develops this argument with measured consequences.
4. **Scaling last, and fitted on training rows only.** `minutes_watched` spans 30 to
   1,450 while `rating` spans 2.5 to 5.0. Any method operating on distance, which
   includes k-nearest-neighbour recommenders, matrix factorisation with regularisation,
   and anything penalised, will be dominated by the wider variable purely because of its
   units.

Effect on the downstream model. Skipping harmonisation fragments the category signal
and weakens any content-based component. Mean-imputing ratings without an indicator
introduces a systematic bias towards recommending material that learners quietly
abandoned. Fitting the scaler on all rows leaks the test distribution into training and
produces an optimistic offline estimate that will not survive contact with live traffic.
Deduplicating carelessly removes genuine repeat engagement, one of the strongest signals a recommender has.

On what not to do. Rows with missing ratings should not be dropped. In the sample
frame that would discard 3 of 15 records, and the discarded rows are systematically the
disengaged learners, that is, the population the recommender most needs to model.

### What each decision is actually worth

The fifteen-row frame in 2(b) shows the mechanics. It is far too small to show what any
of the operations is worth, and a pipeline justified only by argument is not justified.
`code/q2b_impact.py` builds a realistic log, 3,727 interactions across 500 learners and
40 courses, injects the four faults the question names, and prices each fix against a
downstream task: predicting whether a learner completes a recommended course, trained on
the first three quarters of the period and tested on the last.

```
duplicate rows            : 71
distinct category strings : 13 for 5 real categories
missing ratings           : 1286 (34.5%)
completion rate when rating present 0.596 vs missing 0.491
```

| Treatment | Test AUC | Training rows |
|---|---|---|
| Raw log, no cleaning at all | 0.7554 | 2,795 |
| Drop rows with a missing rating | 0.7460 | 1,822 |
| Mean-impute ratings, no indicator | 0.7572 | 2,795 |
| Median-impute plus missingness indicator | 0.7561 | 2,795 |
| Full pipeline with `engagement_ratio` | **0.7597** | 2,741 |

![Preprocessing impact](figures/q2_fig1_impact.png)

Three findings, one of which corrects the argument above.

Deleting incomplete rows is the expensive mistake, costing 0.0101 AUC and 35% of the
training data at once. The rows removed are not a random sample: their completion rate is
0.491 against 0.596 for rows with a rating, so deletion strips out precisely the
disengaged learners the recommender exists to identify. This is the one decision in the
pipeline with a large and unambiguous price.

Cleaning the log end to end is worth 0.0043 AUC. Useful, and smaller than the confidence
with which cleaning is usually recommended.

The missingness indicator earns nothing here. Rows three and four of the table differ in
two respects at once, so the indicator is isolated separately by holding the imputation
rule fixed and toggling only the indicator: it is worth −0.0011 under median imputation
and −0.0011 under mean imputation, while median against mean imputation moves the result
by 0.0000. The effect is therefore attributable to the indicator and to nothing else, and
the honest reading is that my argument in step 3 above is stated too generally. The indicator captures why a
rating is absent, and in this log the reason is low engagement. But `minutes_watched` is
already in the feature set and measures that same engagement directly: the indicator
correlates −0.251 with it and is close to a duplicate. Question 3 is the contrasting case,
where the identical device formed part of a treatment worth 9.2 AUC points, because a
missing credit score signals a thin file and no other field in the loan record reveals
that. A missingness indicator is insurance against selection that nothing else observes.
Where another feature already observes it, the indicator is redundant, and it should be
retained on those grounds and not as a reflex.

## 2(b) Implementation (5 marks)

The full script is `code/q2_recommender_prep.py`. The core operations:

```python
# 1. two kinds of duplicate, treated differently
df = df.drop_duplicates()                                   # ingestion artefacts
df = (df.sort_values("timestamp")
        .drop_duplicates(subset=["learner_id", "course_title"], keep="last"))

# 2. category harmonisation: case, whitespace, separators
def norm(s):
    s = s.strip().lower().replace("/", " ").replace("-", " ")
    s = " ".join(s.split())
    return {"data science": "Data Science", "ai ml": "AI/ML",
            "databases": "Databases"}.get(s, s.title())
df["course_category"] = df.course_category.map(norm)

# 3. missing ratings: indicator retained, statistics from training rows only
cut = df.timestamp.quantile(0.7)
train_mask = df.timestamp <= cut
df["rating_missing"] = df.rating.isna().astype(int)
med_by_cat = df.loc[train_mask].groupby("course_category").rating.median()
df["rating_filled"] = (df.rating.fillna(df.course_category.map(med_by_cat))
                                .fillna(df.loc[train_mask].rating.median()))

# 4. scaling fitted on training rows, applied to all
sc = StandardScaler().fit(df.loc[train_mask, num])
df[[c + "_z" for c in num]] = sc.transform(df[num])
```

Measured result:

```
rows=15  exact duplicates=0  missing ratings=3  distinct category strings=9
raw category labels: [' data science', 'AI / ML', 'AI/ML', 'DATA SCIENCE',
                      'Data  Science', 'Data Science', 'Databases', 'ai/ml', 'databases']

STEP 1  repeat learner-course rows removed (kept most recent): 2
STEP 2  9 raw strings -> 3 canonical: ['AI/ML', 'Data Science', 'Databases']
STEP 3  category medians from training rows: {'AI/ML': 4.5, 'Data Science': 4.0,
                                              'Databases': 3.0}
STEP 5  StandardScaler fitted on training rows only
            variable  train mean  train scale
       rating_filled        3.83         0.71
     minutes_watched      496.67       479.92
```

The scale disparity is worth stating numerically, since it is the reason step 5 exists:
before scaling, `minutes_watched` has a standard deviation of 536.34 against 0.77 for
`rating`, a ratio of roughly 700 to 1.

## 2(c) Two engineered features (2 marks)

`engagement_ratio` = learner's minutes on a course divided by the median minutes for
that course. Raw watch time confounds two different things, how engaged the learner is
and how long the course happens to be. A 45-minute course watched in full and a
1,400-minute course abandoned after 45 minutes produce the same raw value and mean
opposite things. Normalising by the course median separates them. For the recommendation
problem this matters because the quantity of interest is whether this learner found this
material worth their time relative to how other learners treated it, a completion signal that does not depend on a rating being given. In the sample frame the
feature ranges from 0.375 to 4.0, cleanly separating skimming from deep engagement.

`prior_completion_rate` = the proportion of the learner's *earlier* courses that they
completed. Learners differ systematically in follow-through, and that disposition is
one of the strongest available predictors of whether a recommendation will be acted on.
Recommending an advanced twelve-week course to a learner who has abandoned four of five
enrolments is a poor recommendation even if the topic matches perfectly. The
implementation detail is the entire point: it uses `shift()` before `expanding().mean()`,
so a record never contributes to its own feature value. Without the shift this is a
textbook leak, since the current course's completion status would be encoded in the
feature used to predict it.

## 2(d) Preventing information from the future (1 mark)

Split on time, not at random, and fit every statistic on the past side only.

Concretely, four things are required. Records are ordered by timestamp and cut at a date,
so training ends before validation begins. Every fitted quantity, meaning imputation
medians, scaler means and scales, and category vocabularies, is learned on the training
window and applied unchanged afterwards. Any per-learner aggregate is computed with an
expanding window over strictly earlier records, as `shift()` enforces above.
And features that could only be known after the fact, such as final course rating or
total watch time, are excluded from any model that scores a learner mid-course.

The check is mechanical and is run by the script:

```
train window 2026-01-05 .. 2026-04-15  (n=9)
test  window 2026-04-21 .. 2026-06-01  (n=4)
overlap: NONE
first-ever record for any learner carrying non-zero history: False (must be False)
```

The second assertion is the one that catches real bugs. Any learner's first record must
carry a prior-completion rate of zero, because at that moment they have no history. If it
does not, an aggregate has been computed over the full series and the future has entered
the training data.

# Question 3: Missing data in loan default prediction (13 marks)

The analysis uses a 3,000-record loan book generated under a known process, so that the
true values behind the gaps are available for comparison. The missingness mechanisms are
deliberately realistic: income is under-reported by informally employed applicants, and
credit scores are absent for applicants with no borrowing history. Script:
`code/q3_missing_data.py`.

```
                 missing     %
income               506  16.9
years_employed       255   8.5
credit_score         492  16.4
employment_type      122   4.1
```

## 3(a) Critical evaluation of zero-filling (5 marks)

The proposal is wrong, and it is worth being precise about why, because "zero is a bad
imputation" is not an argument. Zero is a legitimate value for some financial variables
and a nonsensical one for others, and the distinction turns on what zero means on each
variable's own scale.

Zero is a real, meaningful value on some of these fields and not on others. A loan
balance of zero means no outstanding debt. A missed-payment count of zero means a clean
record. On those fields, zero-filling does not merely add noise, it writes a specific and
usually favourable claim into the data. Applied to `n_prev_loans`, it converts "we do not
know their borrowing history" into "they have never borrowed", a materially different statement to a credit committee.

On other fields zero is outside the admissible range entirely. A credit score is
defined on 300 to 850. Zero is not a low score, it is not a score. Placing 492 applicants
at zero creates a cluster 300 points below the legal minimum, roughly 3.5 standard
deviations outside the observed distribution, and any model will read that as
overwhelming evidence of default risk. The measured consequence appears in 3(d).

The distortion reaches the geometry the model depends on. In the generated book the applicants
with missing income truly earn between 64,895 and 1,390,624, with a median of 302,620,
slightly above the observed median of 296,520. These are not poor applicants. They are
mostly self-employed and informally employed people who did not supply a payslip.
Zero-filling moves them 1.50 standard deviations below the median of the observed
distribution, inverting their true position.

Derived variables are corrupted silently. Debt-to-income is the single most
important ratio in consumer lending, and `loan_amount / income` becomes a division by
zero for all 506 affected applicants. Depending on the implementation this yields
infinity, a NaN that propagates, or a silently clipped maximum. Any of the three makes
those applicants look like the worst credits in the portfolio.

It confounds two populations that must stay separate. After zero-filling, an
applicant with a genuinely low income and an applicant who declined to state their income
are the same record. These are different credit risks and, under most regulatory regimes,
attract different treatment. The information distinguishing them has been erased rather
than modelled.

The damage is also not spread evenly. Because the missingness is not random, it falls systematically. The script tests
this directly. Testing missingness against the outcome alone is insufficient and would
have cleared two of the three fields:

```
Default rate by whether the field is missing:
  income          missing: 0.247   present: 0.251   Welch p = 8.66e-01  -> consistent with MCAR
  years_employed  missing: 0.204   present: 0.254   Welch p = 5.94e-02  -> consistent with MCAR
  credit_score    missing: 0.183   present: 0.263   Welch p = 4.45e-05  -> NOT MCAR

Missingness tested against an OBSERVED COVARIATE (employment_type):
  income          chi2 p = 4.83e-77  -> MAR, depends on employment_type
                  missing rate: {'informal': 0.417, 'salaried': 0.068, 'self_employed': 0.177}
```

Income is missing for 41.7% of informally employed applicants against 6.8% of salaried
applicants. A target-only test returns p = 0.87 and would have declared it
missing-completely-at-random. Because the missingness concentrates in one employment
class, zero-filling imposes its damage almost entirely on informal-sector applicants, who
in the Rwandan market are the majority of the addressable population. This is the
mechanism by which a naive imputation choice becomes a systematic lending bias.

## 3(b) A defensible strategy (3 marks)

The strategy must distinguish variables by type and, more importantly, by *why* the value
is absent. The two questions are separate and both have to be answered.

First, classify the mechanism. Missing-completely-at-random means the gap is unrelated
to anything, and almost nothing in a loan book qualifies. Missing-at-random means the gap
depends on other observed variables, as the chi-square result above establishes
for income: it depends on employment type, and it is recorded. Conditional imputation is
valid here. Missing-not-at-random means the gap depends on the unobserved value itself, and
credit score is the clear case. The cross-tabulation shows all 492 absent scores belong to
applicants with no prior loans:

```
no prior loans  False  True
score missing
False            2415     93
True                0     492
```

This is not a gap to be filled. It is a structural fact about thin-file customers, and
imputing any score onto that population fabricates a credit history that does not exist.

Second, apply a treatment matched to the type and the mechanism.

| Variable class | Treatment | Reasoning |
|---|---|---|
| Skewed monetary (income, loan amount) | Median from training rows, plus missing indicator | Median resists the right skew that makes the mean misleading on income data |
| Continuous non-monetary (years employed) | Median or conditional median within employment type, plus indicator | MAR given employment type, so conditioning recovers more than a global statistic |
| Structural absence (credit score) | Explicit `no_credit_file` flag, value held separately from the score scale | The applicant has no score; a thin file is a distinct underwriting category |
| Categorical (employment type) | `not_reported` as its own level, never the mode | Non-response is itself informative; imputing the mode invents an employment status |
| Target | Never imputed | Rows without a known outcome are excluded from supervised training |

Third, retain the indicator in every case. If non-response carries no information the
model will assign the indicator a near-zero coefficient and nothing is lost. If it does
carry information, and here it plainly does, that information is preserved instead of
being averaged away. This is the cheapest insurance available in the whole pipeline.

Fourth, learn every statistic on training rows only, and persist it. A median computed
over the full dataset leaks the test distribution; a median recomputed at scoring time
makes today's decision depend on who else applied today, and that is indefensible to an applicant and to a regulator.

Listwise deletion should be rejected explicitly. The measurement in 3(d) shows it retains
only 62% of training rows and shifts the observed default rate from 0.250 to 0.276, because
the discarded records are not a random sample.

## 3(c) A pipeline-ready function (3 marks)

Implemented as a scikit-learn transformer so it composes inside a `Pipeline` and inherits
the fit/transform separation that prevents leakage by construction. Full source in
`code/q3_missing_data.py`.

```python
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Type-aware missing-value handler.

    Assumptions, stated explicitly:
      1. Numeric fields are MAR given the observed data, so a median computed on the
         training rows is an acceptable central estimate. The median is used in place of the mean because every monetary field here is right-skewed.
      2. Missingness itself may be informative, so an indicator column is retained for
         every field that is imputed. If the indicator carries no signal the model can
         ignore it; if it does, the information is not thrown away.
      3. Structurally absent values (no credit file) are a distinct category, not a low
         value, and are flagged separately instead of being imputed onto the score scale.
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
```

Assumptions and their limits. The median is a point estimate, so single imputation of
this kind understates uncertainty: the model treats an imputed income as being known as
precisely as a reported one. Multiple imputation addresses that and should be preferred
where the downstream use requires calibrated intervals and not a ranking. The
transformer also assumes the missingness mechanism is stable between training and
production, an assumption that has to be monitored and not believed, since a
change in application-form design can alter it overnight.

## 3(d) Consequences for the model and for applicants (2 marks)

Three treatments, identical splits, identical model, so the only difference is the
handling of gaps:

| Treatment | Test AUC | Brier score | Note |
|---|---|---|---|
| Zero-fill, as proposed by management | 0.6659 | 0.1772 | |
| Type-aware median with indicators | **0.7575** | **0.1586** | |
| Complete-case deletion | 0.7524 | 0.1611 | Retains 1,397 of 2,250 rows; default rate shifts 0.250 to 0.276 |

Model performance. Zero-filling costs 9.2 points of AUC, from 0.758 to 0.666. On a
discrimination scale where 0.5 is a coin toss, that is roughly a third of all the signal
the model had. The Brier score, which measures the accuracy of the probabilities rather
than the ranking, worsens by 12%, so the predicted probabilities are not merely ordered
worse, they are numerically wrong. Complete-case deletion very nearly matches the proper
treatment on these metrics but does so having discarded 38% of the training data, and its
retained sample has a default rate 2.6 percentage points above the true one. That bias
does not appear in the AUC and would surface later as a miscalibrated portfolio.

Decisions about individual applicants. This is where the abstract metric becomes a
person. Of 750 test applicants, 228 have at least one missing field.

```
mean predicted default probability, those applicants:
    zero-fill      0.2883
    type-aware     0.2373
largest single shift: 0.3342 in predicted probability
at a 20% decline threshold, 75 applicants (32.9% of those with gaps) receive a
    different decision purely from the imputation choice
at a 30% decline threshold, 31 applicants (13.6%) receive a different decision
```

One applicant's assessed probability of default moves by 33 percentage points depending
on a preprocessing default that nobody would think to document. At a 20% decline
threshold, a third of applicants with any data gap are decided differently. None of these
people changed. Their circumstances did not change. A line of code changed.

Because the gaps concentrate in informally employed applicants, as established in 3(a),
those reversals are not spread evenly across the applicant pool. They fall on a specific
economic class, which converts a technical shortcut into a fair-lending exposure. That is
the argument to put to the management team: the proposal is not a simplification that
costs a little accuracy, it is an undocumented change in credit policy.

# Question 4: Feature engineering for subscription renewal (12 marks)

Measured on 4,000 generated customer records with a known outcome process, so that the
cost of each encoding choice can be quantified, not asserted. Script:
`code/q4_encoding.py`.

## 4(a) Pipeline design (3 marks)

The variables divide into three groups, and the division is by measurement level rather
than by storage type. This is the judgement the question is testing, because
`subscription_type` and `region` are both stored as strings and must be treated
differently.

| Group | Variables | Transformation | Justification |
|---|---|---|---|
| Numeric, skewed | `monthly_spend` | Log, then standardise | Log-normal by construction; a linear model fitted on the raw scale is dominated by the upper tail |
| Numeric, roughly symmetric | `age`, `n_prev_transactions`, `account_duration_days` | Standardise | Puts all coefficients on a comparable scale and makes any penalty act evenly |
| Ordinal | `subscription_type` (basic, standard, premium) | Ordinal encoding with the order stated explicitly | A genuine ranking exists, and one coefficient captures it |
| Nominal | `region`, `payment_method`, `customer_status` | One-hot | No ordering exists, so none may be imposed |

On scaling. Standardisation in preference to min-max, because min-max is defined by the
observed minimum and maximum and is therefore hostage to a single outlier; a customer
with an unusually long account compresses everyone else into a narrow band.
Standardisation also matters for any penalised model, since a penalty applied to
coefficients on different scales penalises variables unequally for reasons that have
nothing to do with their importance.

An empirical note that qualifies the usual advice: in this dataset, omitting scaling cost
almost nothing in discrimination, 0.7302 against 0.7310 AUC. What it did cost was
convergence. The unscaled model exhausted 2,000 L-BFGS iterations without converging on
every cross-validation fold. Scaling here is a numerical-conditioning requirement rather
than an accuracy requirement, and that is the honest statement of its value.

Ordering. Fit every transformation on training data only, and compose the whole
sequence inside a single `ColumnTransformer` so that the same objects can be persisted
and reused at scoring time. Section 4(d) shows what the alternative costs.

## 4(b) Consequences of each encoding choice (4 marks)

Measured by five-fold stratified cross-validated AUC, varying only the encoding:

| Strategy | Model | CV AUC | Width |
|---|---|---|---|
| One-hot nominal, ordinal tier | Logistic regression | **0.7310** | 14 |
| Ordinal codes forced onto nominal | Logistic regression | 0.6424 | 8 |
| One-hot everything, tier order discarded | Logistic regression | 0.7305 | 15 |
| Correct encoding, no scaling | Logistic regression | 0.7302 | 14 |
| Ordinal codes forced onto nominal | Random forest | 0.6908 | 8 |
| One-hot nominal | Random forest | 0.7069 | 17 |

One-hot encoding. Appropriate whenever the categories have no order, which covers
`region`, `payment_method` and `customer_status`. It imposes no false relationship, and
each category gets its own coefficient. The costs are real but manageable at this
cardinality: the design matrix widens from 8 to 14 columns, and with a high-cardinality
field such as a customer identifier or a postcode it would widen without limit while each
resulting column carried almost no data. It is the wrong choice when cardinality is high
relative to sample size, where target encoding with out-of-fold statistics or a grouped
"other" category is preferable. Notably, one-hot encoding the ordinal tier cost only
0.0005 AUC. Discarding a genuine ordering is nearly free when the variable has three
levels, because the model simply spends two coefficients where one would have done. That
is the mild error.

Ordinal encoding. Correct for `subscription_type`, where basic, standard and premium
form a real progression, provided the order is stated explicitly rather than left to
alphabetical accident. Applied to nominal variables it is the severe error, costing
0.0886 AUC, or 12% of the model's discrimination. The reason is visible in the
generating process:

```
  region    alphabetical code   true effect on renewal
  Kigali                    0                      0.9
Northern                    1                     -0.3
Southern                    2                     -0.5
 Eastern                    3                      0.2
 Western                    4                     -0.4
  correlation between the integer code and the real effect: -0.572
```

Encoding these regions as 0 to 4 asserts that Kigali is less than Northern is less than
Southern, and that the gap between each pair is identical. Neither claim is true. The
model fits one coefficient to a sequence carrying no meaning, and the sign of the
correlation between code and effect is arbitrary, being an artefact of the alphabet.

Inappropriate numerical encoding, and why the model matters. The same mistake under a
random forest costs 0.0161 AUC against 0.0886. A tree splits on thresholds and can
isolate any individual integer code through repeated splits, so it recovers most of the
structure a linear model cannot. This is the part usually stated too simply. Encoding is
not right or wrong in isolation, it is right or wrong relative to the model that consumes
it. Ordinal codes on nominal variables are close to fatal for linear and distance-based
methods and merely inefficient for tree ensembles. Even so, the forest still pays for it,
so the defensible default is to encode correctly and not to rely on the model to
compensate.

## 4(c) Implementation (3 marks)

```python
pre = ColumnTransformer([
    ("num",  Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("scale",  StandardScaler())]), num),
    ("ord",  Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                       ("encode", OrdinalEncoder(categories=[tiers],
                                                 handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), ordinal),
    ("nom",  Pipeline([("impute", SimpleImputer(strategy="constant",
                                                fill_value="not_reported")),
                       ("encode", OneHotEncoder(handle_unknown="ignore"))]), nominal),
])
full = Pipeline([("pre", pre),
                 ("model", LogisticRegression(max_iter=2000, random_state=42))]).fit(X, y)
```

Two details carry the weight. `categories=[tiers]` states the ordering explicitly, so it
cannot be reassigned alphabetically when a future dataset happens to sort differently.
`handle_unknown` is set on both encoders, so an unrecognised category at scoring time
produces a defined value instead of an exception in production.

## 4(d) Consistency at deployment (2 marks)

Persist the fitted pipeline and load it at scoring time. Never rebuild it. The
transformation is not code, it is code plus the parameters learned from training data:
the medians, the means and scales, and the category vocabularies. Re-deriving those from
whatever data happens to be present at scoring time silently changes the transformation.

```
saved artefact reproduces training-time output exactly: True
training feature width frozen at 17 columns
refitting the transformer on a later batch instead of loading it shifts the same
    five customers by up to 0.0924 in predicted probability
```

Refitting instead of loading moved identical customers by up to 9.2 percentage points of
predicted renewal probability. Nothing about those customers changed. This is train/serve
skew, and it is silent, since both pipelines run without error and produce plausible
numbers.

A specific trap worth naming, because it is a default that looks safe. Combining
`drop="first"` with `handle_unknown="ignore"` is a common pairing and the two options are
incompatible:

```
reference level 'Eastern' -> [[0, 0, 0, 0]]
unseen level 'Musanze'    -> [[0, 0, 0, 0]]
```

The dropped reference category and a genuinely unrecognised category produce an identical
all-zero vector. A customer from a region the model has never seen is therefore scored as
though they were from the reference region, silently and with no error raised. The remedy
is to keep all levels when unknown categories are possible in production, or to add an
explicit `unknown` category during training so the two cases stay distinguishable. This
also surfaced during Question 7's small-sample experiments, where subsamples that happened
to contain no `highway` journeys folded that category into the reference level.

Beyond the artefact. Pin library versions, since encoder behaviour changes between
releases. Validate the schema of each incoming batch against the training schema and
fail loudly on a mismatch instead of coercing. Monitor the input distributions, because
a pipeline that is applied perfectly consistently to data that has drifted is still
producing the wrong answer, just reproducibly.

# Question 5: Hospital readmission prediction (13 marks)

Measured on 3,558 admissions belonging to 1,200 patients, of whom 78% have more than one
admission. That structure is the crux of the question. Scripts:
`code/q5_readmission.py` and `code/q5b_leakage_scaling.py`.

## 5(a) End-to-end workflow (4 marks)

1. Define the label before touching the data. "Readmitted within a specified period"
is under-specified, and the specification changes the problem. Thirty days is the usual
clinical and reimbursement standard. Planned readmissions, for instance a scheduled second
stage of a procedure, must be excluded, or the model will learn to predict the surgical
calendar rather than clinical deterioration. Transfers between facilities must not count
as discharge followed by readmission. This step is placed first because every later
decision depends on it, and relabelling after modelling invalidates the work.

2. Fix the prediction point. The model must run at discharge, so only information
available at discharge may be used. Any post-discharge field, such as a follow-up
appointment attendance flag, is unusable no matter how predictive it looks. This is the
single most common source of a model that performs superbly offline and fails in
deployment.

3. Acquire and link, at the patient level. Admissions are linked to a stable patient
identifier before anything else, because that identifier is what makes the partitioning in
5(b) possible. Without it, group-aware splitting cannot be done and the evaluation cannot
be trusted.

4. Prepare. Diagnoses and procedures are high-cardinality codes and should be grouped
into clinically coherent categories instead of one-hot encoded raw. Length of stay is
right-skewed and log-transformed. Missing clinical values follow the type-aware strategy
of Question 3, with the specific caution that in clinical data an absent test result
usually means the clinician saw no reason to order it, and that is information about the patient's presentation and must be preserved as an indicator.

5. Partition, grouped by patient. Detailed in 5(b) and implemented in 5(d).

6. Train a baseline first. Penalised logistic regression, because it is calibrated,
auditable, and gives a floor that any more complex model must clear. Gradient boosting
follows. A clinical model that cannot be explained to the clinicians expected to act on it
will not be used, so interpretability is a functional requirement, not a nicety.

7. Evaluate on the ranking, not on accuracy. With a 34% readmission rate, a model
predicting "never readmitted" scores 66% accuracy and is worthless. Discrimination and
calibration are the relevant properties, together with the subgroup analysis in 5(c).

8. Set the threshold from capacity. If the transitional-care team can follow up
twenty patients a week, the threshold is the one that yields twenty flags a week. In the
measured run the top quartile of predicted risk was flagged, giving recall 0.438 at
precision 0.590, meaning roughly six of every ten flagged patients were genuinely
readmitted.

9. Deploy behind the persisted pipeline of Question 4(d), with input-schema validation
and a shadow period in which predictions are recorded but not acted upon, so that live
performance can be compared with the offline estimate before anyone relies on it.

10. Monitor and schedule retraining. Clinical practice, coding standards and case mix
all drift. Monitoring covers input distributions, calibration and subgroup performance,
not just aggregate discrimination. As in Question 1(c), acting on predictions changes the
outcomes being predicted, so the intervention flag must be recorded as a variable or
retraining will progressively erase the signal.

## 5(b) Partitioning strategy (3 marks)

The strategy: split by patient, not by admission, and hold the test set back for a
single final evaluation. Because 78% of patients contribute more than one admission, a
partition drawn over rows places the same individual on both sides of the split. The model
can then recognise the patient instead of the clinical pattern, and the test score
measures memorisation as well as skill.

Measured on a single split:

| Split | Test AUC | Patients on both sides |
|---|---|---|
| Random over admissions | 0.7455 | 581 |
| Grouped by patient | 0.7299 | 0 |
| Temporal, first 75% of days | 0.7579 | 561 |

The careless split reports 0.0156 AUC more than the grouped one. **A single split is a
noisy instrument, and that difference alone does not establish anything**, which is why
the comparison was repeated across twelve seeds while sweeping the number of admissions
recorded per patient:

| Admissions per patient | Random AUC | Grouped AUC | Inflation | 95% CI | p |
|---|---|---|---|---|---|
| 1 | 0.7159 | 0.7319 | −0.0160 | [−0.0474, +0.0153] | 0.338 |
| 2 | 0.7469 | 0.7515 | −0.0046 | [−0.0137, +0.0045] | 0.345 |
| 4 | 0.7476 | 0.7353 | +0.0123 | [+0.0015, +0.0232] | 0.048 |
| 8 | 0.7636 | 0.7574 | +0.0063 | [−0.0005, +0.0131] | 0.099 |
| 16 | 0.7745 | 0.7666 | +0.0079 | [+0.0017, +0.0140] | 0.030 |

![Leakage scaling with records per patient](figures/q5_fig2_leakage.png)

The honest reading: with one admission per patient the two procedures are identical by
construction and the difference is noise, as expected. Once patients repeat, the optimism
becomes positive and statistically detectable, but in this setting it is modest, on the
order of 0.006 to 0.012 AUC. It is not the dramatic collapse the textbook version of this
warning implies.

That modest size is itself the useful finding, and it has a mechanism. Optimism from group
leakage scales with how much of the outcome is patient-specific and with how many records
each patient contributes. Here the patient effect is observable through a recorded chronic
risk score, so a model trained on other patients recovers most of it legitimately, and
little is left to memorise. In a setting where the patient effect is large and not
captured by any recorded feature, the same mistake can inflate the estimate by far more.
The practical conclusion is unchanged: group by patient regardless, because grouping costs
nothing and the optimism it prevents is unbounded in the worst case.

Why careless partitioning flatters a model, in general terms. Three mechanisms, only
the first of which is specific to this dataset. Group leakage, where records from one
entity span the split, so the model is tested on entities it has already seen. Temporal
leakage, where training data postdates test data, so the model is given knowledge of a
period it is supposedly forecasting. And preprocessing leakage, where scalers, imputers or
feature-selection steps are fitted on the full dataset, so the test distribution informs
the transformation. The third is the most common in practice and the least visible, since
the code looks correct.

A further caution specific to this dataset: the cross-validated comparison gave
StratifiedKFold 0.7267 and StratifiedGroupKFold 0.7289, with the grouped estimate
marginally *higher*. Had only that comparison been run, the conclusion would have been
that grouping does not matter. Single comparisons at this effect size are not informative,
and the repeated experiment is what settles it.

## 5(c) Uneven performance across patient groups (3 marks)

The situation is normal and not exceptional, and the aggregate number is what
concealed it. An overall AUC of 0.7314 decomposes as:

| Subgroup | n | Base rate | AUC | Recall | Precision | Flag rate |
|---|---|---|---|---|---|---|
| insurance = mutuelle | 430 | 0.344 | 0.749 | 0.439 | 0.644 | 0.235 |
| insurance = none | 98 | 0.347 | 0.701 | 0.529 | 0.529 | 0.347 |
| insurance = private | 135 | 0.311 | 0.695 | 0.357 | 0.484 | 0.230 |
| sex = F | 347 | 0.354 | 0.763 | 0.455 | 0.675 | 0.239 |
| sex = M | 316 | 0.320 | 0.699 | 0.416 | 0.506 | 0.263 |
| age 18-45 | 152 | 0.263 | 0.721 | 0.275 | 0.524 | 0.138 |
| age 46-65 | 300 | 0.347 | 0.736 | 0.433 | 0.616 | 0.243 |
| age 66+ | 211 | 0.379 | 0.717 | 0.525 | 0.583 | 0.341 |

![Split comparison and subgroup performance](figures/q5_fig1.png)

The widest AUC gap is 0.068 and the widest recall gap is 0.254. The recall gap is the one
with clinical consequences: the system identifies 52.9% of at-risk uninsured patients and
27.5% of at-risk patients aged 18 to 45. Both groups are being served by the same headline
number.

Diagnosis before remedy. Three causes are worth separating, because they call for
different responses. Sample size, since the uninsured group has 98 test records and its
subgroup estimates carry wide intervals that should be reported with confidence bounds and not as point values. Genuinely different relationships, as is the case here by construction, since the uninsured group's readmission risk was generated with an
additional term, so one global model is fitting a compromise that suits neither group.
And differential measurement, where a group with less complete records supplies the model
with less to work with, the most likely explanation for the private-insurance
group being modelled worst despite not being the smallest.

What should be added before deployment.

Report subgroup metrics as standard, with confidence intervals, and treat the minimum
across subgroups as the headline figure in place of the average. Add calibration by
subgroup, since a model can rank well within a group while being systematically
over- or under-confident about it, and a miscalibrated risk score sends the wrong absolute
number to a clinician. Measure equal-opportunity difference, meaning the spread in
true-positive rate at the operating threshold, the 0.254 recall gap above and is
the fairness criterion that matches the harm at stake, namely a readmission that was
preventable and was not prevented. Consider group-specific thresholds so that each group
is flagged at a rate proportionate to its risk, noting that this is a policy decision with
legal implications and not a modelling decision to be taken quietly. Evaluate against a
clinical baseline such as the LACE index, because a model that underperforms existing
practice in a subgroup should not be deployed for that subgroup. And run a prospective
silent period before the model influences any care decision.

Underlying all of it: the question of whether the model is good enough cannot be
answered in aggregate, because no patient experiences the aggregate. A patient is treated
by the subgroup-specific behaviour of the system.

## 5(d) Partition implementation (3 marks)

```python
from sklearn.model_selection import GroupShuffleSplit

# 1. hold out 20% of PATIENTS as the test set, touched once at the very end
rest, test = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
                  .split(X, y, groups=patient_id))

# 2. split the remainder into train and validation, again by patient
#    0.25 of the remaining 80% gives a 60/20/20 overall partition
tr, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
               .split(X.iloc[rest], y.iloc[rest], groups=patient_id.iloc[rest]))
train_idx, val_idx = rest[tr], rest[val]

# 3. verify, do not assume
for a, b in [("train","validation"), ("train","test"), ("validation","test")]:
    assert len(set(patient_id.iloc[parts[a]]) & set(patient_id.iloc[parts[b]])) == 0
```

Result:

```
subset        admissions  patients   share  readmit rate
train               2195       720   61.7%         0.350
validation           700       240   19.7%         0.314
test                 663       240   18.6%         0.338
patient overlap between subsets: train/validation=0, train/test=0, validation/test=0
```

The shares are approximate and not exact because patients contribute different numbers
of admissions, and the split is over patients. That is the correct behaviour and the
alternative would defeat the purpose. The assertion at step 3 is not decoration: it is the
only thing standing between a correct split and a silent bug, and it costs one line. For
model selection rather than a single split, `StratifiedGroupKFold` provides the same
protection while preserving the class balance in each fold.

# Question 6: US Crime dataset (13 marks)

47 states, 15 predictors, no missing values, with `Crime` measured as offences per 100,000
population. The ratio of 3.1 observations per predictor governs the entire analysis and is
stated first because it disqualifies several otherwise reasonable approaches.
Script: `code/q6_crime.py`.

## 6(a) Feature-selection strategy (3 marks)

Step 1: begin from the constraint, not from the data. With n = 47 and p = 15, ordinary
least squares fitted on all predictors has 31 residual degrees of freedom. The measured
consequence is severe: cross-validated R² for the full model is **0.098**, against an
in-sample fit that appears far better. The model is memorising states. Any strategy that
does not reduce dimensionality is not viable here, and this is a property of the dataset and not a preference.

Step 2: remove redundancy on structural grounds, before looking at the target. `Po1`
and `Po2` are police expenditure in 1960 and 1959 and correlate at **r = 0.994**. They are
one variable recorded twice. Dropping one is a decision about the data's construction, not
a data-driven selection, so it does not consume any of the statistical budget and does not
risk selection bias.

Step 3: apply the three selection families and compare them honestly under
cross-validation. Filter methods rank by univariate association and are fast but blind
to interaction, which matters here because Section 6(b) shows two variables whose
importance only appears once others are held constant. Wrapper methods such as recursive
feature elimination search subsets against model performance and capture interaction, at
the cost of being prone to overfitting the selection itself at this sample size. Embedded
methods such as Lasso select during fitting and are the most efficient use of a small
sample.

Step 4: transform where the variable's own scale calls for it. `Pop` has skew 1.98 and
`NW` 1.47, both right-skewed, so a log transformation is appropriate. `Prob` is a
probability and a logit transformation is natural. `So` is already binary. The target
itself has skew 1.125, which argues for modelling log(Crime) if the residuals show
heteroscedasticity.

Step 5: consider combining instead of discarding. `Wealth` and `Ineq` correlate at
−0.884 and both describe the same economic structure. A composite index, or principal
components, retains the information that dropping one would discard.

Measured comparison, five-fold cross-validation repeated ten times:

| Strategy | k | CV RMSE | CV R² |
|---|---|---|---|
| All 15 predictors | 15 | 296.84 | 0.098 |
| Filter, abs(r) with Crime ≥ 0.30 | 6 | 317.39 | −0.052 |
| Drop redundant (Po2, U1, Wealth) | 12 | 280.95 | 0.196 |
| Embedded, LassoCV (α = 11.39) | 11 | 250.04 | 0.327 |
| Wrapper, RFE with CV | 4 | 255.32 | **0.334** |

Two findings deserve emphasis. The filter method performs **worse than using everything**,
with a negative R² meaning it predicts less well than the sample mean. Selecting on
univariate correlation discarded variables whose contribution is conditional, the failure mode Section 6(b) demonstrates. And the wrapper method reaches the
best R² using only four predictors, `Ed`, `Po1`, `Po2`, `Ineq`, though its selection of
both `Po1` and `Po2` shows that recursive elimination does not by itself resolve
collinearity and still needs the structural judgement from step 2.

Recommended strategy: drop `Po2` structurally, log-transform `Pop` and `NW`, then use
Lasso with cross-validated penalty for selection, and report the result alongside a
domain-chosen model, not in place of it. Selection should be nested inside the
cross-validation loop, otherwise the reported performance is optimistic for the same
reason discussed in Question 5(b).

## 6(b) Exploratory analysis and correlation matrix (6 marks)

```python
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
```

![Correlation matrix](figures/q6_fig1_corr.png)

Association with the target.

| Variable | Meaning | r with Crime | p |
|---|---|---|---|
| Po1 | Police spending 1960 | **+0.688** | <0.001 |
| Po2 | Police spending 1959 | +0.667 | <0.001 |
| Wealth | Median assets | +0.441 | 0.002 |
| Prob | Probability of imprisonment | **−0.427** | 0.003 |
| Pop | Population | +0.338 | 0.020 |
| Ed | Mean schooling | +0.323 | 0.027 |
| M.F | Males per 100 females | +0.214 | 0.149 |
| Ineq | Income inequality | −0.179 | 0.229 |
| NW | Non-white population | +0.033 | 0.828 |

Six of fifteen predictors reach conventional significance. Note that with fifteen
simultaneous tests at α = 0.05, roughly one false positive is expected by chance alone,
so the marginal results around p = 0.02 to 0.03 should not be over-read.

![Bivariate relationships](figures/q6_fig2_scatter.png)

Relationships that look useful. Police expenditure is the strongest single correlate.
The probability of imprisonment is the strongest negative one, consistent with a
deterrence reading. Wealth and education both associate positively with crime, a result that puzzles until one notices they also correlate with urbanisation and with
reporting quality.

Why correlation alone is not sufficient evidence for selecting features. Four distinct
reasons, each measured on this dataset instead of asserted.

*First, reverse causation.* `Po1` has the strongest correlation with `Crime` in the
dataset. The natural reading, that police spending causes crime, is absurd. The plausible
mechanism runs the other way: states experiencing high crime allocate more money to
policing. A correlation coefficient is symmetric and contains no information about
direction. It cannot distinguish "police cause crime" from "crime causes police", and
nothing in the matrix resolves it.

*Second, confounding.* Both `Wealth` and `Ed` associate positively with crime, which
inverts the usual expectation. Both also correlate with urbanisation, and urban areas have
more reported property crime and better reporting infrastructure. The bivariate
correlation absorbs the confounder's effect and attributes it to the variable being
examined.

*Third, and most damaging for feature selection, suppression.* A variable can appear
irrelevant on its own and become strongly significant once others are held constant:

```
    Ineq    simple r=-0.179 (p=0.229)   controlled beta=  281.65 (p=0.0010)
    M       simple r=-0.089 (p=0.550)   controlled beta=  129.99 (p=0.0164)
    U2      simple r= 0.177 (p=0.233)   controlled beta=   14.35 (p=0.7350)
```

Income inequality has a **negative, non-significant** simple correlation with crime. After
controlling for police spending and wealth, its coefficient is **positive, large and
highly significant** (p = 0.001). Both the sign and the significance reverse. A filter that
screened at abs(r) ≥ 0.30 would have discarded `Ineq` and `M` before modelling began, and that is why that strategy produced a negative cross-validated R² in 6(a). The
converse also appears: `U2` looks mildly promising alone and is clearly irrelevant once
conditioned.

*Fourth, correlation measures only monotone linear association.* A U-shaped relationship
returns r near zero. Pearson correlation also responds strongly to outliers, and with 47
points a single unusual state can move a coefficient materially.

The effect works in the other direction too. `Po1`'s simple correlation of 0.688 rises to
a partial correlation of **0.773** once `Ineq`, `Ed` and `Prob` are held constant, and its
standardised coefficient increases from 263.10 to 340.72. The bivariate figure understated
it.

## 6(c) The problem caused by correlated predictors (2 marks)

The problem is variance inflation, which destabilises coefficients and makes them
uninterpretable, while leaving overall predictive accuracy largely intact. This last
clause matters: the model still predicts, so the failure is invisible unless the
coefficients are examined.

```
predictor       VIF   severity
      Po2    113.56     severe
      Po1    104.66     severe
   Wealth     10.53     severe
     Ineq      8.64   moderate
       U1      6.06   moderate
```

![Collinearity](figures/q6_fig3_collinearity.png)

A VIF of 113 means the variance of that coefficient is 113 times what it would be under
orthogonal predictors, so its standard error is inflated roughly elevenfold. The
consequence is visible in the fitted model:

| Predictor | OLS coefficient | Ridge coefficient |
|---|---|---|
| Po1 | **+566.87** | +124.47 |
| Po2 | **−302.69** | +104.60 |
| Ineq | +278.94 | +94.64 |
| Ed | +208.43 | +74.16 |

`Po1` and `Po2` correlate at 0.994 and measure the same thing one year apart. Ordinary
least squares assigns them **coefficients of opposite sign**, +566.87 and −302.69. Read
literally, the model claims police spending in 1960 raises crime sharply while spending in
1959 lowers it. That is not a finding, it is the arithmetic of a near-singular design
matrix, which can only identify their sum and distributes it arbitrarily between them. A
small change in the sample would redistribute it differently. The same instability was
measured directly in Question 7, where the collinear predictors flipped sign in 26% to 41%
of bootstrap refits.

Strategies, in order of preference for this dataset.

Remove one of the pair on structural grounds. `Po1` and `Po2` are the same measurement in
consecutive years, so retaining `Po1` loses nothing. This is the first move because it
requires no statistical machinery and is defensible to a non-technical audience.

Apply ridge regression, which keeps all predictors and shrinks the variance at the cost of
a little bias. Measured here at α = 10.91, ridge reduces `Po1` from 566.87 to 124.47 and
corrects `Po2` from −302.69 to +104.60, so the two now agree, as they must. Cross-validated
RMSE is 282.04 against 296.84 for full OLS, so stability was gained without losing
accuracy. One caution on reading the shrinkage table: for `So`, whose OLS coefficient is
−1.80 and therefore near zero, the percentage change is arithmetically extreme and
meaningless.

Combine into components. Five principal components capture 80% of the variance and eight
capture 95%, with principal-component regression on eight components reaching CV RMSE
272.95, better than full OLS. The cost is interpretability, since a component
mixing wealth, education and inequality has no policy meaning, and this dataset exists to
inform policy.

## 6(d) Recommendations to government (2 marks)

What the analysis supports. Three variables are consistently retained across
independent selection methods and are worth treating as planning indicators: police
expenditure (`Po1`), income inequality (`Ineq`) and education (`Ed`). Recursive feature
elimination selected these plus `Po2` and reached the best cross-validated fit of any
subset tested. They can be used to identify which areas are likely to record higher crime,
and therefore where to position resources.

What the analysis does not support, stated plainly. Every relationship reported here
is an association measured across 47 states at one point in time. None of it is causal
evidence, and three specific cautions follow.

*Do not read the police coefficient as a policy lever.* Police expenditure is the
strongest correlate of crime and the association is positive. Acting on it directly would
imply cutting police budgets to reduce crime, which inverts the likely direction of
causation. Resources follow crime. This variable is useful as an indicator of where crime
already is and is unusable as an instrument for reducing it.

*Do not read the null results as evidence of no effect.* Unemployment shows no significant
simple correlation, yet `Ineq` demonstrates that a variable can be invisible alone and
important once conditioned. Absence of correlation at n = 47 is weak evidence of absence.

*Do not extrapolate.* These are 47 US states from a single historical period. The
relationships are not transportable to Rwandan districts, where the reporting
infrastructure, the age structure and the economic base all differ. Applying these
coefficients elsewhere would be an error of a different order from the statistical ones
above.

The distinction, stated for the record. Statistical association means two quantities
move together in this sample and offers no guarantee that changing one changes the other.
Causal interpretation requires either an experiment or a design that credibly rules out
confounding and reverse causation, and this dataset supports neither. The practical
consequence is that the selected variables are legitimate for *targeting* resources and
illegitimate for *justifying* interventions.

What would be needed to move from association to causation. Panel data tracking the
same states over time, so that within-state changes can be examined and time-invariant
state characteristics differenced out. A natural experiment or policy discontinuity, such
as a funding formula that allocates policing by an arbitrary population threshold. A
pre-registered analysis plan, given that fifteen predictors offer ample scope for
selecting a congenial result. And an explicit causal diagram stating which variables are
assumed to be confounders and which are on the causal path, since conditioning on a
mediator introduces bias instead of removing it.

# Question 7: YEGO ETA prediction (14 marks)

1,500 journeys, 13 candidate predictors, no missing values and no duplicate rows. Target
`eta_minutes` has mean 38.13 and standard deviation 13.54. Scripts:
`code/q7_regression.py` and `code/q7b_complexity.py`.

## 7(a) Experimental procedure (3 marks)

Data preparation. Drop `observation_id`, which identifies records and carries no
information about journeys. One-hot encode `road_type` with `arterial` as the reference
level. Standardise every numeric predictor, which is not optional here: ridge and lasso
penalise the sum of squared or absolute coefficients, so on unstandardised data a variable
measured in small units attracts a large coefficient and is penalised more heavily for
reasons of measurement scale alone. Standardising also makes the coefficients directly
comparable, each expressing minutes of ETA per one standard deviation of the predictor.

Splitting. Hold out 20% (300 journeys) as a test set that is touched exactly once, at
the end. Within the remaining 1,200, use ten-fold cross-validation for all model
selection. The penalty parameter is chosen on cross-validation folds only. Using the test
set to choose α would make the reported test error an optimistic estimate of a quantity
that no longer means anything.

Training. OLS has no hyperparameter. Ridge and lasso are searched over 61 values of α
on a logarithmic grid from 10⁻³ to 10³, wide enough that the chosen value is interior and not at a boundary. Every transformation sits inside a `Pipeline` so the scaler is
refitted within each cross-validation fold, preventing the preprocessing leakage described
in Question 5(b).

Evaluation. RMSE as the primary measure, since it is in minutes and penalises the
large errors that damage a ride-hailing arrival promise most. MAE alongside it, being less
sensitive to a handful of extreme journeys. R² for the share of variance explained. Beyond
these, and central to the argument in 7(c): the number of retained predictors, the
stability of coefficients under resampling, and the gap between training and test error.

A step most comparisons omit. Because the three models are expected to perform
similarly on a sample this size, the procedure includes a bootstrap analysis of
coefficient stability and a sweep of training-set size. Without these, the comparison
reduces to three nearly identical numbers and no basis for choosing between them.

## 7(b) Implementation and comparison (6 marks)

```python
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)])
cv = KFold(n_splits=10, shuffle=True, random_state=42)
alphas = np.logspace(-3, 3, 61)

for name, est, grid in [("Linear Regression (OLS)", LinearRegression(), None),
                        ("Ridge Regression", Ridge(), {"model__alpha": alphas}),
                        ("Lasso Regression", Lasso(max_iter=50000), {"model__alpha": alphas})]:
    pipe = Pipeline([("pre", pre), ("model", est)])
    best = (GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv)
            .fit(X_tr, y_tr).best_estimator_) if grid else pipe.fit(X_tr, y_tr)
```

Multicollinearity, as advertised in the data description.

```
                predictor      VIF          Strongest predictor-predictor correlations
      journey_distance_km    10.86          journey_distance_km <-> distance_proxy  0.928
            traffic_index     7.38          traffic_proxy       <-> traffic_index   0.897
           distance_proxy     7.22          previous_journey    <-> journey_distance 0.835
previous_journey_time_min     5.25          traffic_index       <-> average_speed   0.770
            traffic_proxy     5.20
```

![Correlation matrix](figures/q7_fig1_corr.png)

Results.

| Model | α | CV RMSE | Train RMSE | Test RMSE | Test MAE | Train R² | Test R² |
|---|---|---|---|---|---|---|---|
| Linear Regression (OLS) | – | 5.7119 | 5.6518 | 6.2366 | 4.3801 | 0.8223 | 0.7999 |
| Ridge Regression | 1.000 | 5.7117 | 5.6519 | **6.2353** | **4.3775** | 0.8223 | **0.8000** |
| Lasso Regression | 0.004 | 5.7117 | 5.6519 | 6.2355 | 4.3781 | 0.8223 | 0.8000 |

![Predicted versus actual](figures/q7_fig2_pred.png)

The three models are indistinguishable. Test RMSE spans 6.2353 to 6.2366, a range of
0.0013 minutes, or eight hundredths of a second. Reporting Ridge as "the winner" on
that margin would be meaningless. The gap between training RMSE (5.65) and test RMSE
(6.24) is about 10%, modest and consistent across all three, so none is
materially overfitting.

Standardised coefficients, in minutes per one standard deviation.

| Predictor | OLS | Ridge | Lasso |
|---|---|---|---|
| journey_distance_km | 9.537 | 9.442 | 9.484 |
| traffic_index | 6.811 | 6.757 | 6.756 |
| road_type_highway | −4.112 | −4.071 | −4.066 |
| road_type_express | −3.034 | −3.011 | −3.007 |
| weather_score | 1.829 | 1.825 | 1.825 |
| road_type_local | 1.476 | 1.488 | 1.478 |
| distance_proxy | **−0.932** | −0.873 | −0.880 |
| previous_journey_time_min | 0.907 | 0.946 | 0.911 |
| road_slope_pct | 0.758 | 0.758 | 0.755 |
| average_speed_kmh | **0.313** | 0.301 | 0.295 |
| traffic_proxy | **−0.229** | −0.201 | −0.190 |
| random_noise_feature | −0.139 | −0.138 | −0.134 |
| time_of_day_hours | 0.079 | 0.076 | 0.072 |
| passenger_rating | −0.022 | −0.022 | −0.018 |

![Coefficient shrinkage across the three models](figures/q7_fig3_coef.png)

Three coefficients have signs that contradict their own simple correlations with the
target. `distance_proxy` correlates +0.662 with ETA and receives a negative coefficient.
`traffic_proxy` correlates +0.431 and receives a negative coefficient. `average_speed_kmh`
correlates −0.380 and receives a positive one. These are the sign reversals predicted by
collinearity in Question 6(c), appearing here in a second dataset.

Lasso eliminated nothing at the cross-validation optimum. The selected α of 0.004 is
so small that all 15 coefficients remain non-zero, including `random_noise_feature`, which
the data description states has no relationship with ETA. At n = 1,200 with 15 predictors,
cross-validation finds it cannot improve prediction by shrinking, so it does not.

![Lasso paths](figures/q7_fig4_path.png)

The coefficient paths show why the exercise still worked as designed: `passenger_rating`,
`time_of_day_hours`, `random_noise_feature` and `driver_experience_years` are the first to
reach zero as α rises. Regularisation does order the predictors correctly by usefulness.
It simply is not asked to discard any of them at the penalty that minimises prediction
error.

The one-standard-error rule, which is where regularisation earns its place. Selecting
the largest α whose cross-validated error is within one standard error of the minimum:

```
best alpha            = 0.0039   CV RMSE = 5.7117 (SE 0.1536)
1-SE threshold        = 5.8653
largest alpha within  = 0.2868   CV RMSE = 5.8563
```

| Model | α | Predictors kept | Test RMSE | Test R² |
|---|---|---|---|---|
| OLS, all predictors | – | 15 | 6.2366 | 0.7999 |
| Lasso, CV-optimal | 0.0039 | 15 | 6.2355 | 0.8000 |
| **Lasso, 1-SE rule** | 0.2868 | **8** | 6.3680 | 0.7914 |

The 1-SE model retains `journey_distance_km` (8.184), `traffic_index` (6.015),
`weather_score` (1.594), the three road-type indicators, `previous_journey_time_min`
(1.213) and `road_slope_pct` (0.468). It discards exactly the seven predictors that should
be discarded: `random_noise_feature`, `passenger_rating`, `driver_experience_years`,
`time_of_day_hours`, both redundant proxies, and `average_speed_kmh`. Nearly half the
predictors removed costs 0.13 minutes of RMSE, or eight seconds.

Coefficient stability, 300 bootstrap refits at n = 200.

| Predictor | OLS sd | Ridge sd | Reduction | OLS sign flips |
|---|---|---|---|---|
| journey_distance_km | 1.645 | 0.864 | 47.5% | 0.0% |
| traffic_index | 1.209 | 0.758 | 37.3% | 0.0% |
| distance_proxy | 1.180 | 0.768 | 35.0% | **26.3%** |
| traffic_proxy | 0.989 | 0.701 | 29.1% | **41.3%** |
| previous_journey_time_min | 0.950 | 0.702 | 26.1% | 17.7% |
| average_speed_kmh | 0.620 | 0.540 | 12.9% | **27.7%** |
| passenger_rating | 0.448 | 0.416 | 7.1% | 44.0% |
| random_noise_feature | 0.427 | 0.403 | 5.8% | 39.0% |

Ridge reduces coefficient variance by 26% to 47% on exactly the collinear predictors and
by only 3% to 7% on the independent ones, the behaviour the penalty is designed to produce. `traffic_proxy` changes sign in 41.3% of refits under OLS: its estimated
direction of effect is close to a coin toss. A model whose coefficients are quoted to a
product team must not behave that way.

![Complexity and stability](figures/q7_fig5_complexity.png)

Where regularisation actually matters, by training-set size.

| n_train | OLS | Ridge | Lasso | OLS − Ridge |
|---|---|---|---|---|
| 30 | 8.520 | 7.688 | **7.544** | +0.832 |
| 50 | 7.344 | 7.182 | **6.972** | +0.162 |
| 80 | 6.833 | 6.847 | 6.693 | −0.014 |
| 200 | **6.421** | 6.462 | 6.475 | −0.042 |
| 1200 | 6.237 | **6.234** | 6.381 | +0.003 |

This explains the tie. At 30 training journeys OLS is 0.83 minutes worse than ridge, an
11% degradation. The advantage disappears by roughly 80 observations and is absent
thereafter. With 1,200 journeys and 15 predictors, OLS is already well-conditioned enough
that shrinkage has nothing left to correct.

### Inference and regression assumptions

RMSE measures how far predictions fall from the truth. It says nothing about whether the
linear form is right, whether a coefficient is distinguishable from zero, or how precisely
each effect is estimated. Those are separate questions, and on a question about linear
regression they need answering. `code/q7c_diagnostics.py` fits the same specification
through statsmodels to obtain them.

| Predictor | Coefficient | Std error | t | p | 95% CI |
|---|---|---|---|---|---|
| journey_distance_km | 9.5371 | 0.5436 | 17.55 | <0.001 | 8.47 to 10.60 |
| traffic_index | 6.8112 | 0.4476 | 15.22 | <0.001 | 5.93 to 7.69 |
| weather_score | 1.8287 | 0.1703 | 10.74 | <0.001 | 1.49 to 2.16 |
| road_type_highway | −4.1117 | 0.5621 | −7.32 | <0.001 | −5.21 to −3.01 |
| road_type_express | −3.0342 | 0.4635 | −6.55 | <0.001 | −3.94 to −2.12 |
| road_slope_pct | 0.7583 | 0.1655 | 4.58 | <0.001 | 0.43 to 1.08 |
| road_type_local | 1.4756 | 0.4014 | 3.68 | 0.0002 | 0.69 to 2.26 |
| previous_journey_time_min | 0.9069 | 0.3727 | 2.43 | 0.015 | 0.18 to 1.64 |
| distance_proxy | −0.9317 | 0.4448 | −2.09 | 0.036 | −1.80 to −0.06 |
| average_speed_kmh | 0.3130 | 0.2600 | 1.20 | 0.229 | −0.20 to 0.82 |
| driver_experience_years | 0.1539 | 0.1656 | 0.93 | 0.353 | −0.17 to 0.48 |
| random_noise_feature | −0.1389 | 0.1650 | −0.84 | 0.400 | −0.46 to 0.18 |
| traffic_proxy | −0.2294 | 0.3725 | −0.62 | 0.538 | −0.96 to 0.50 |
| time_of_day_hours | 0.0789 | 0.1698 | 0.47 | 0.642 | −0.25 to 0.41 |
| passenger_rating | −0.0217 | 0.1652 | −0.13 | 0.896 | −0.35 to 0.30 |

Nine predictors are significant at 5% and six are not. The six are exactly those the data
description marks as uninformative or redundant, `random_noise_feature` among them, and
their confidence intervals all straddle zero.

The cross-check is the useful part. The one-standard-error lasso above kept eight
predictors on purely predictive grounds, with no hypothesis test anywhere in the
procedure. All eight are significant here. The only disagreement is `distance_proxy`,
significant at p = 0.036 and dropped by the lasso. Under HC3 robust standard errors, which
the heteroscedasticity finding below requires, that verdict flips and `distance_proxy`
becomes non-significant too. Applied correctly, the two procedures agree completely, which
is stronger support for the eight-predictor model than either gives on its own.

Model level: R² 0.8223, adjusted R² 0.8200, F = 365.2 (p < 0.001), residual standard error
5.69 minutes. The condition number of the design matrix is 7.3, comfortably below the
usual threshold of 30. That is worth setting against the VIF of 10.86 reported above.
Variance inflation is a pairwise diagnostic and flags the proxy pairs correctly, while the
condition number measures how near-singular the matrix is as a whole. They disagree
because the collinearity is confined to specific pairs and does not extend to the full
matrix, which is also why OLS stayed stable enough to tie with ridge.

| Assumption | Test | p | Verdict |
|---|---|---|---|
| Linear functional form | Ramsey RESET | 0.505 | No evidence of a missing non-linear term |
| Constant variance | Breusch-Pagan | 0.016 | Heteroscedastic |
| Normal residuals | Jarque-Bera | <0.001 | Non-normal, skew 0.639 |
| Independence | Durbin-Watson | 1.970 | No autocorrelation |

![Regression diagnostics](figures/q7_fig6_diagnostics.png)

Two assumptions fail, and neither invalidates the model for its purpose. Ordinary least
squares stays unbiased under heteroscedasticity; it is the standard errors that become
unreliable, which is why the coefficient table should be read with the HC3 correction.
Non-normal residuals matter for small-sample inference, and at n = 1,200 the central limit
theorem covers the coefficient tests, but it does mean a prediction interval quoted to a
passenger would be wrong in the tails. The right-skew of 0.639 says the model
underestimates a minority of long journeys more than it overestimates short ones, and that
is the error pattern a ride-hailing operator would care about most.

Influence: 55 of 1,200 observations exceed the conventional 4/n threshold on Cook's
distance, which is 4.6% and close to chance. The largest is 0.0616, far below 1.0, so no
individual journey drives the fit.

## 7(c) Which model to deploy (2 marks)

Recommendation: lasso at the one-standard-error penalty, giving an eight-predictor
model. If all fields must be retained for other reasons, ridge rather than OLS.

Choosing on test RMSE alone is not possible, since the three differ by 0.0013 minutes.
Four other considerations decide it.

Predictors are an operating cost, not a free parameter. Each retained feature is a
field that must be collected, validated, stored and monitored in production. The 1-SE
model needs eight inputs instead of fifteen. The 0.13-minute accuracy cost, against a
typical journey of 38 minutes, is an error of one third of one per cent, and no passenger
distinguishes a 38.0-minute estimate from a 38.1-minute one. Halving the feature pipeline
for that is a straightforward trade.

Two of the discarded predictors were redundant by construction. `traffic_proxy` and
`distance_proxy` exist to duplicate `traffic_index` and `journey_distance_km` at r = 0.897
and r = 0.928. Removing them eliminates the multicollinearity rather than merely damping
it, which is the preference established in Question 6(c).

One discarded predictor should never have been a candidate. `average_speed_kmh` is
described as the estimated average speed during the journey. A journey's average speed is
not knowable at the moment the ETA is quoted, which is before the journey starts. Whatever
its statistical contribution, it is unavailable at prediction time, and including it would
produce a model that validates well offline and cannot be served. The 1-SE model removes
it on statistical grounds; it should have been removed on design grounds. This is the
Question 5(a) discipline of fixing the prediction point applied to a specific field.

Stability is a deployment property. The bootstrap analysis shows OLS coefficients on
the collinear predictors flipping sign in up to 41% of resamples. A model retrained weekly
on fresh journey data would present a different and sometimes contradictory account of
what drives ETA each week. Ridge reduces that variance by up to 47%, and the 1-SE lasso
removes the unstable predictors altogether.

The honest qualification. Ridge is the better answer if YEGO expects to retrain on
small city-level or corridor-level subsets, where the learning curve shows OLS losing 0.83
minutes at n = 30. The recommendation is conditional on the operating regime, and with
1,500 journeys per model the 1-SE lasso wins on simplicity at negligible cost.

## 7(d) Reducing overfitting risk (3 marks)

1. Nest model selection inside cross-validation, and keep a test set that is used once.
The α values here were chosen on ten-fold cross-validation within the training set alone,
and the 300-journey test set was scored a single time. Where selection and evaluation share
data, the reported error is a selection artefact. For a fully unbiased estimate of the
whole procedure, including the choice of α, nested cross-validation is required: an inner
loop selecting α and an outer loop estimating error. The gap between training and test RMSE
here, 5.65 against 6.24, is the quantity to watch, and a widening gap on retraining is the
first symptom of overfitting.

2. Prefer the simpler model when the difference is within noise, using the 1-SE rule.
This is the measure with the largest effect in this study. Cross-validated error has a
standard error of 0.1536, and the difference between the best and the eight-predictor model
is 0.1446, and that is inside it. The two models are statistically indistinguishable, and one
uses seven fewer inputs. Selecting the minimum of a cross-validation curve systematically
chooses a model slightly too complex, because the minimum of a noisy curve sits below the
true minimum.

3. Validate the way the model will be used, not the way the data is stored. Random
splitting assumes journeys are exchangeable. They are not. Journeys cluster by driver, by
corridor and by time of day, and a random split places journeys from the same driver on the
same corridor in both training and test, which is the group leakage quantified in Question
5(b). A production estimate requires splitting by time, training on earlier weeks and
testing on later ones, and by driver or corridor. The reported R² of 0.80 should be
regarded as an upper bound until that is done.

4. Audit features for availability at prediction time. As noted in 7(c),
`average_speed_kmh` cannot be known before the journey. A feature that is unavailable at
serving time produces a validated model that cannot be deployed, and it is not detectable
by any amount of cross-validation, because the leak is in the data's definition and not in the split.

5. Monitor after deployment. Kigali traffic in six months is not Kigali traffic today.
Track the distribution of each input and the realised prediction error against the offline
estimate, and retrain on a schedule instead of waiting for complaints.

# Question 8: Product-launch survey analysis (12 marks)

600 unique responses after cleaning, with stated willingness to purchase at 36.3%.
Script: `code/q8_survey_eda.py`.

## 8(a) Exploratory strategy (3 marks)

The brief is explicit that management wants evidence for a decision rather than a summary
of responses. That shapes the analysis: every step should reduce uncertainty about whether
to launch, at what price, and to whom.

Establish what the sample can and cannot support, first. With 600 responses, the 95%
margin of error on a single proportion is ±3.8 percentage points. Any difference smaller
than that is not a finding. Stating this before analysis prevents the familiar pattern of
reporting a 2-point gap between subgroups as though it were real.

Data quality, before any interpretation. Twelve duplicate submissions were present and
removed. `satisfaction_current` is missing for 4.3% of respondents and
`expected_price_band` for 3.0%. Non-response is checked for informativeness and not assumed neutral: willingness among those who skipped the satisfaction question is 0.308
against 0.366 for those who answered, a gap in the direction one would expect from
disengagement, though the subgroup of 26 is too small to conclude from.

Distributions before relationships. Each variable's measurement level determines what
may be done with it. `expected_price_band` is ordinal and its ordering must be preserved,
as in Question 4. `occupation` and `preferred_feature` are nominal. `satisfaction_current`
is a five-point scale, so medians and rank tests are safer than means, although both are
reported.

Relationships selected by decision relevance. Only three or four comparisons actually
bear on the launch: does prior experience with similar products predict willingness, does
willingness vary with expected price, does dissatisfaction with the current solution
predict willingness, and do feature preferences differ by segment. Each is paired with an
appropriate statistical test, so that a visible pattern is separated from a real one.

Segments last, because segmentation is only meaningful once the drivers are known.

## 8(b) Visualisations (4 marks)

Each chart is chosen from the measurement levels of the two variables involved, and each
carries the test statistic that justifies reading it.

![Survey relationships](figures/q8_fig1_survey.png)

(a) Prior use against willingness. Nominal by binary, so a grouped bar of proportions.
A pie chart would be wrong, since the comparison is between two rates rather than parts of
a whole.

```
willingness: prior users 45.5% vs non-users 28.6%
chi-square p = 2.82e-05, Cramer V = 0.171
```

(b) Expected price band against willingness. Ordinal by binary, so an ordered bar with a
trend test. The category order carries meaning and must not be sorted by frequency.

```
                       n  willing %
<5k                  177       27.7
5-15k                187       37.4
15-30k               112       38.4
30-50k                70       42.9
>50k                  36       58.3
Spearman rho = +0.144 (p = 5.10e-04)
```

(c) Current satisfaction against willingness. Ordinal by binary, so a box plot. It
shows the median, the spread and the overlap, where a bar of means would hide all three.

```
median satisfaction, willing = 2.0 vs not willing = 4.0  (Mann-Whitney p = 5.45e-08)
mean satisfaction,   willing = 2.68 vs not willing = 3.35
```

(d) Preferred feature by age group. Nominal by nominal, so a heatmap of column
percentages. Column rather than row percentages, because the question is what each age
group wants, not how each feature's supporters are distributed.

```
age_group          18-24  25-34  35-44  45-54   55+
local language      32.1   24.4   17.4   29.8  22.2
mobile money        30.0   37.3   27.5   22.6  28.9
offline mode        17.9    9.8   28.3   27.4  20.0
chi-square p = 5.13e-04, Cramer V = 0.131
```

## 8(c) Insights bearing on the launch (3 marks)

Insight 1: the addressable market is the dissatisfied, not the unfamiliar. Willingness
among prior users of similar products is 45.5% against 28.6% among non-users, a 16.9-point
gap at p < 0.001. Dissatisfaction points the same way, with median satisfaction of 2 among
willing buyers against 4 among unwilling ones (p < 0.001). The two together describe a
specific buyer: someone who has used a competing product and found it wanting. The
implication for the launch is that positioning should be comparative and migration-focused,
addressing people who already understand the category, instead of educational messaging
aimed at first-time users. It also identifies the competitor's dissatisfied base as the
primary acquisition channel.

Insight 2: price is not the binding constraint, and the pricing assumption is probably
wrong. Willingness *rises* with the price band the respondent already expects to pay,
from 27.7% in the under-5,000 band to 58.3% above 50,000, with Spearman ρ = +0.144 at
p < 0.001. The usual reading of a price question, that a lower price widens the market, is
contradicted here. Respondents anchored to low prices are the least likely to buy at any
price, which is consistent with them placing little value on the category. The implication
is that a low-price launch would attract the least willing segment while surrendering
margin on the most willing one. The caution attached to this is that the highest band
contains only 36 respondents, so the 58.3% figure carries a wide interval and the direction
of the trend is better supported than its magnitude.

Supporting finding on the feature roadmap. Preferences differ significantly by age
(p < 0.001), but Cramér's V of 0.131 indicates a weak association. Mobile money is the top
or joint-top preference in every age group, ranging from 22.6% to 37.3%. The practical
conclusion is that mobile money is a universal requirement, not a segment feature,
while offline mode shows a genuine split, at 9.8% among 25 to 34 year olds and 28.3% among
35 to 44 year olds, and is a candidate for later differentiation rather than for the first
release.

Segment concentration.

| Segment | n | Willing | Share of sample | Share of all willing |
|---|---|---|---|---|
| Experienced, higher budget | 100 | 54.0% | 16.7% | 24.8% |
| Experienced, lower budget | 175 | 40.6% | 29.2% | 32.6% |
| New, higher budget | 126 | 32.5% | 21.0% | 18.8% |
| New, lower budget | 199 | 26.1% | 33.2% | 23.9% |

Experienced buyers with a higher budget are 16.7% of respondents and 24.8% of everyone
willing to buy. The two experienced segments together are 45.8% of the sample and 57.4% of
willing buyers. A launch targeting them addresses well over half the demand with under half
the outreach.

## 8(d) Limitation and further evidence (2 marks)

The limitation: stated willingness to purchase is not purchase. Every finding here
rests on what 600 people said they would do about a product that does not exist and that
they have not seen priced. Three distinct biases operate, and they do not cancel. Hypothetical
bias, where survey willingness routinely overstates purchase by a large factor, so the
36.3% headline is an upper bound and not a forecast. Acquiescence and social-desirability
effects, where respondents give the answer the questioner appears to want. And selection,
since the analysis can say nothing about who declined to take the survey, and non-respondents
are plausibly the least interested. There is a further structural point: because the sample
is dominated by people already familiar with the category, and familiarity is the strongest
predictor of willingness, the survey may be measuring the enthusiasm of an existing market instead of the size of a new one.

Additional evidence to obtain before committing the budget. In rough order of cost:

A discrete-choice or conjoint exercise, in which respondents choose between specified
bundles at specified prices instead of reporting an abstract willingness. This produces
price-elasticity estimates that the current ordinal price question cannot, and the ordering
effect found in 8(c) needs exactly that test before it is trusted as a pricing rule.

A pre-launch commitment test: a landing page with real pricing, measuring sign-ups or
pre-orders. A small financial or registration commitment separates intent from politeness
far more reliably than any survey instrument, and it can be run in days.

A limited pilot in one segment, preferably the experienced higher-budget group, measuring
actual conversion, activation and retention. Retention is the number that determines whether
the product is viable, and no survey can supply it.

Market sizing from independent data, since 600 respondents can establish the *proportion*
willing but not the absolute size of the addressable market.

Competitive and channel analysis, because the finding that the buyer is a dissatisfied
competitor customer makes the competitor's switching costs and response the decisive
commercial variable, and the survey says nothing about either.

Recommendation to management: treat the survey as having identified who to sell to and
what to build first, and as having established nothing reliable about how many will buy or
at what price. Fund the commitment test and the pilot before the full launch.

# Appendix: Repository contents

| Path | Contents |
|---|---|
| `data/synthetic_ETA.csv` | Question 7 dataset, 1,500 journeys |
| `data/uscrime.txt` | Question 6 dataset, 47 states |
| `code/q2_recommender_prep.py` | Q2 preprocessing pipeline and leakage checks |
| `code/q2b_impact.py` | Q2 measured impact of each preprocessing decision |
| `code/q3_missing_data.py` | Q3 missingness diagnostics, `MissingValueHandler`, comparison |
| `code/q4_encoding.py` | Q4 encoding experiments and deployment artefact |
| `code/q5_readmission.py` | Q5 workflow, partitioning, subgroup analysis |
| `code/q5b_leakage_scaling.py` | Q5 repeated-seed leakage measurement |
| `code/q6_crime.py` | Q6 EDA, correlation, VIF, feature selection |
| `code/q7_regression.py` | Q7 OLS, Ridge, Lasso comparison |
| `code/q7b_complexity.py` | Q7 1-SE rule, bootstrap stability, learning curve |
| `code/q7c_diagnostics.py` | Q7 coefficient inference and assumption tests |
| `code/q8_survey_eda.py` | Q8 survey EDA and significance tests |
| `figures/` | 11 generated figures |
| `output/` | Console logs and result tables for every question |
| `run_all.sh` | Regenerates every result from scratch |

All results in this report were produced by these scripts at seed 42 and can be
reproduced by running `bash run_all.sh`.
