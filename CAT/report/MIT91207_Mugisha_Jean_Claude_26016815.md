# MIT91207 Machine Learning

Name: Mugisha Jean Claude
Registration number: 26016815
Programme: MSc Information Technology, Level 9
Module: MIT91207 Machine Learning
Lecturer: Dr Gustave Udahemuka

Section A is compulsory. From Section B I have answered Question Four and
Question Five, the two permitted.

## How to read this

Every figure quoted below is produced by a script in `code/` and executed in this
notebook. Nothing is asserted that is not measured. Where a result contradicts the
argument I was about to make, the result is reported and the argument is changed.

| | |
|---|---|
| Python | 3.9.6, seed 42 throughout |
| Libraries | pandas, numpy, scikit-learn, statsmodels, scipy, matplotlib |
| Question One | No dataset supplied, so a farm-season panel is generated under a stated process |
| Question Four | `data/loan.csv`, 20,000 applicants, 34 variables |
| Question Five | `data/crimes.csv`, 2,215 communities, 144 variables |

# SECTION A

# Question One: agricultural low-yield early warning

## 1(a) Evaluating the proposal to deploy on predictive performance alone

The proposal fails on a definitional point before any of the practical objections
land. "Highest overall performance" is not a property a model has. It is a number
produced by a choice of metric, a choice of validation scheme and a choice of test
sample, and every one of those choices is contestable. A model that tops the table
under one protocol can sit mid-table under another. So the proposal is not merely
risky; as stated it does not specify a decision rule at all.

Beyond that, four factors need settling before anything is deployed.

### Validation protocol decides the ranking, and this data breaks the usual one

Farms recur across seasons. A holdout drawn at random over farm-season rows puts the
same farm on both sides of the split, so the model is rewarded for recognising farms
it has already been fitted to. The demonstration in 1(c) measures this on the
generated panel: a random split reports AUC 0.7105 against 0.6912 for a split grouped
by farm, an optimism of 0.0193 with 1,694 of 1,800 farms appearing on both sides.
Whichever model "wins" under the careless protocol may simply be the model that
memorises hardest. The ranking has to be re-derived under a grouped and
forward-in-time protocol before it means anything.

### The operating point matters more than the ranking

Field officers cannot visit every farm. If the advisory capacity is, say, the top
quartile of risk, then what matters is precision and recall at that cut-off, not the
area under a curve that averages over thresholds the company will never use. The two
errors are also priced very differently. A missed low-yield farm costs a smallholder
a season of income. A false alarm costs an officer a morning. Any metric that weights
these equally, which includes accuracy and to a lesser degree AUC, encodes a
preference the company has not stated and would not endorse if asked.

### Coverage is unequal, so an average conceals who the model fails

In the generated panel the district shares run from 34.0% for Nyagatare down to 4.6%
for Karongi, and missing fertiliser records run the other way, from 4.3% to 28.6%.
A single headline number is dominated by the districts that contributed most of the
data. The subgroup audit in 1(e) is the check that has to pass, and the intervals in
that table are wide enough that several apparent differences are not differences at
all.

### The model will be obsolete on a schedule, and nobody has set the schedule

The scenario states that weather patterns have changed. Section 1(d) shows what that
does: performance holds between AUC 0.735 and 0.767 across the six seasons the model
was fitted on, then drops to between 0.654 and 0.686 once the rainfall regime shifts.
Deploying "immediately" without a retraining trigger, a monitoring plan and an owner
is deploying an asset that decays silently.

Two further considerations sit outside the modelling but decide whether the system is
acceptable.

Predictions here allocate scarce inputs to smallholders, so a farm flagged as low risk
may be denied fertiliser or pest support it would have received under the previous
process. That is a consequential decision about a household's income, and it needs a
stated appeal route and a human decision-maker who can override the model. Field
officers also need to know why a farm was flagged, because "the model says so" is not
an instruction anyone can act on. A ranked list with the two or three drivers attached
is usable; a bare probability is not.

Finally, the model predicts association, not effect. Farms with low fertiliser use
have lower yields, but that does not establish that supplying fertiliser to a flagged
farm raises its yield by the amount the model implies. Establishing that requires a
trial, and the company should run one on a subset before scaling the intervention.

## 1(b) Formulating the problem

The decision the company is making is a triage decision under a capacity constraint:
which farms does a field officer visit before planting. That framing, and not the data, should determine the target.

### As classification

Target: `low_yield`, a binary indicator that the farm's yield for the coming season
falls below a stated threshold, for example the 25th percentile of yield for that crop
and district in recent seasons. Note that the threshold has to be defined relative to
a district and a crop, because an absolute tonnes-per-hectare cut-off would flag
entire low-potential districts every season and tell the company nothing it did not
already know.

A calibrated probability from this model supports the decision directly. Farms are
ranked, the top N by risk are assigned to officers where N is the visit capacity, and
the probability itself can be banded into act, monitor and no-action tiers. Because
the output is calibrated, "0.7" can be audited against the observed rate among farms
given that score, which is what makes the number defensible to a manager.

### As regression

Target: `yield_kg_per_ha`, the continuous yield for the coming season.

This supports a different and complementary decision. The classifier says whether to
intervene; the regression says how much is at stake. A farm predicted at 15% below its
district threshold and a farm predicted at 60% below are both "at risk", and the
classifier cannot distinguish them, yet they justify very different responses. The
continuous prediction also allows the company to estimate the tonnage at risk across a
district, which is what an operations manager needs for planning fertiliser volumes instead of officer visits.

### The recommendation

Fit the regression on the continuous yield and derive the classification by thresholding it, instead of fitting a separate classifier. The underlying quantity is
continuous and the binary label is an administrative convention applied to it, so
modelling the convention discards information. Running both is reasonable in practice,
and disagreement between them is a useful signal that a farm is atypical and deserves
human review.

One caveat that applies to both framings: the prediction must be made at a point in
the season when intervention is still possible. A model that uses rainfall over the
whole season predicts a yield that has already happened. The feature set is therefore a
function of the decision date, and everything observed after it is unusable no matter
how predictive.
## 1(c) A leakage-safe workflow, and why the order matters

Leakage is not one failure. It is at least four, they enter at different points, and
the order of operations is what keeps each of them out. The workflow below is stated
in the order it must run, with the reason each step sits where it does.

Step 1. Fix the decision date and discard everything observed after it. For a
pre-planting warning the model may use soil, farm size, seed variety, irrigation
status, prior-season yields and the long-run climate normals for the location. It may
not use realised rainfall for the season being predicted, harvest-time pest counts, or
the yield of the neighbouring plot recorded at harvest. This is first because no later
step can repair it. A feature that does not exist at decision time produces a model
that validates well and cannot be served.

Step 2. Split by farm and by time, before touching the data. Farms recur, so the
split is grouped on `farm_id`. Seasons progress, so the test period follows the
training period. Splitting first is what makes every subsequent statistic honest.

Step 3. Fit imputation on the training rows only. Median fertiliser is computed on
training farms and applied unchanged to the test rows. A missingness indicator is
retained, because in this panel the gaps are not random: they run from 4.3% in
Nyagatare to 28.6% in Karongi, so absence encodes district and extension-service
coverage.

Step 4. Fit scaling on the training rows only. Rainfall has a standard deviation
of 141 against 0.53 for soil pH, a ratio of roughly 265 to 1. Any distance-based or
penalised method is otherwise dominated by rainfall for reasons of measurement units.

Step 5. Encode categorical variables with the training vocabulary. District and
seed variety are nominal. An unseen category at scoring time must map to a defined
value rather than raise, and it must be distinguishable from a legitimate reference
level.

Step 6. Select features inside the cross-validation loop, not before it. Ranking
predictors on the full dataset and then cross-validating the winners reports the error
of the selection, not of the procedure.

Step 7. Evaluate once on the held-out period.

### What the measurement actually shows

The textbook warning is that steps 3 and 4 must follow step 2. On this panel that
warning is worth nothing:

```
preprocess then split  AUC = 0.7119
split then preprocess  AUC = 0.7119
difference            +0.0000
```

A median over 18,000 rows and a standard deviation over 18,000 rows barely move when
4,500 of them are removed, so the leak exists but is too small to measure. That is a
property of these statistics and this sample size, not a general result. Replace the
median with a target encoder, or run the same test on 200 rows, and the gap opens
immediately.

The leak that does bite is the one the textbook mentions less:

```
random split over farm-seasons   AUC = 0.7105   farms on both sides: 1,694
grouped by farm                  AUC = 0.6912   farms on both sides: 0
optimism                        +0.0193
```

Nearly every farm appears on both sides of a random split, and the reported score is
0.0193 higher as a result. The practical lesson is that the ordering rule people
recite is the cheap one, and the grouping rule people forget is the expensive one.

## 1(d) Diagnosing and responding to decay

The two kinds of change have to be separated because they call for different responses,
and only one of them is visible without labels.

Covariate shift is a change in the distribution of the inputs. The relationship
between predictors and yield is intact; the model is simply being asked about farms
unlike those it was fitted on. It is detectable immediately, from unlabelled data.

Concept drift is a change in the relationship itself. The same rainfall now implies
a different yield. It is invisible in the inputs and only becomes measurable once
outcomes arrive, which in agriculture means a season later.

Both are present in the panel, and separating them is the point:

```
input drift, predictors only
  rainfall_mm      KS 0.308  p 0.00e+00    mean 877.4 -> 787.1
  temperature_c    KS 0.173  p 2.01e-113   mean  19.7 ->  20.2
  pest_reports     KS 0.242  p 2.62e-222   mean   1.4 ->   2.2

relationship drift, the coefficient itself
  2016-2021: coefficient on standardised rainfall = -0.4262
  2022-2025: coefficient on standardised rainfall = -0.2206
```

Rainfall fell by about 90 mm and the coefficient on it roughly halved. Those are two
separate failures happening together. A monitoring system watching only input
distributions would have raised an alarm on the first and stayed silent on the second,
while concluding, wrongly, that it had characterised the problem.

The consequence for performance:

```
2016-2021 (fitted period)   AUC 0.735 to 0.767
2022-2025 (after the shift) AUC 0.654 to 0.686
```

### Monitoring and response

Monitoring runs at three levels, because each catches something the others miss.

At the input level, a Kolmogorov-Smirnov or population-stability check per predictor,
per season, against the training distribution. This is cheap, needs no labels, and
fires immediately. The thresholds should be calibrated on the historical
season-to-season variation instead of a textbook constant, since agricultural
inputs vary substantially between years without anything being wrong.

At the prediction level, the distribution of predicted risk and the flag rate per
district. A model whose flagged share jumps without a corresponding change in inputs
is behaving differently for a reason worth finding.

At the outcome level, AUC, calibration and precision or recall at the operating
threshold, computed per season as harvest data lands. This is the only level that
detects concept drift, and it is necessarily a season behind.

The response ladder should be agreed in advance so that nobody is negotiating under
pressure. Recalibration first, which re-fits the mapping from score to probability and
is cheap. Refitting on a rolling window next, the correct response to concept
drift and implies discarding the oldest seasons instead of accumulating them.
Re-specification last, if the drivers themselves have changed, for example if
irrigation has become the dominant factor in a way the original feature set did not
anticipate.

Two design choices reduce exposure before any of this is needed. Weighting recent
seasons more heavily in training accepts a little variance for a large reduction in
staleness. And holding out the most recent season as the test set at every refit means
the reported performance is always an estimate of next season and not of the past.

One caution that applies to the whole scheme: once field officers act on the
predictions, successful interventions prevent the outcomes the model predicts, and
measured performance falls even though the system is working. Unless the intervention
is recorded as a variable, retraining will progressively unlearn the signal that made
the system useful.

## 1(e) Is the regional difference real, or is it the model?

The question cannot be answered by looking at the flag rates, because a difference in
flag rates is exactly what a correct model produces when the underlying rates differ.
The investigation has to separate three candidate explanations.

First, establish the ground truth rate per district. On the held-out farms:

```
 district    n  true rate  flag rate    AUC   recall  precision  missing fert.  AUC 95% CI
     Huye  410      0.688      0.254  0.700    0.323      0.875          0.235  0.650-0.749
 Bugesera  720      0.685      0.444  0.688    0.529      0.816          0.108  0.646-0.729
  Kayonza 1040      0.636      0.238  0.669    0.309      0.823          0.062  0.641-0.699
  Karongi  200      0.635      0.355  0.722    0.472      0.845          0.286  0.643-0.790
Nyagatare 1420      0.597      0.186  0.690    0.245      0.788          0.043  0.665-0.721
  Musanze  710      0.573      0.166  0.688    0.226      0.780          0.130  0.650-0.726
```

The flag rate spans 0.278 while the true low-yield rate spans 0.115, and the two
correlate at +0.712 across districts. Most of the disparity therefore tracks real
differences in need. The model is not inventing the gap. It is, however, amplifying
it, and the amplification is the part that needs justification, and not the gap itself.

Second, check whether the model works equally well where it is applied. AUC ranges
from 0.669 to 0.722, a spread of 0.053, and recall ranges from 0.226 to 0.529, a spread
of 0.303. The recall spread is the one with consequences: in Bugesera the system finds
53% of the farms that go on to have poor yields, in Musanze 23%.

Third, and this is the step most audits skip, ask whether any of it is
distinguishable from noise. Every district interval in the table above overlaps every
other. Karongi's AUC of 0.722 is the highest in the table and rests on 200 farms, with
an interval of 0.643 to 0.790 that is 0.147 wide; Nyagatare's interval is 0.056 wide on
1,420 farms. Ranking districts by a point estimate would produce a league table that
reshuffles on the next sample. There is a tempting pattern in the data, a correlation
of +0.865 between a district's missing-data rate and the model's AUC there, and it
rests on six points with overlapping intervals. It is a hypothesis to test on more
districts, not a finding to report.

### What the company should require before the model allocates support

Report every subgroup metric with an interval, and treat the worst credible value, and not the point estimate, as the number that must clear the bar.

Separate the two fairness questions, because they have different answers here.
Calibration asks whether a score of 0.7 means the same thing in Karongi as in
Nyagatare; if it does, the differing flag rates are defensible. Equal opportunity asks
whether a genuinely at-risk farm has the same chance of being found regardless of
district; the recall spread of 0.303 says it does not, and that is the finding that
should block deployment in its current form.

Test the mechanism and not only the outcome. Missing fertiliser records run from
4.3% to 28.6% across districts, and if the model performs worse where records are
thinner, the remedy is better data collection in those districts, not a change to the
model. Refitting a per-district model, or adding district interactions, will paper over
a measurement problem and entrench it.

Set the threshold per district if the policy is to serve each district in proportion
to its need. This is a policy decision with legal implications, not a modelling
decision, and it should be taken by someone empowered to take it.

Finally, run the system in parallel with existing practice for a season before it
controls any allocation, and compare what it would have done against what officers
actually did. If the model only reproduces the existing allocation, it has added
nothing; if it diverges, the divergences are precisely the cases worth reviewing by
hand.
# SECTION B

# Question Four: loan amount prediction

Worked on `data/loan.csv`: 20,000 applicants, 34 variables, no missing cells.
Target `LoanAmount`, mean 24,883 and standard deviation 13,427.

## 4(a) Experimental procedure

Before anything is fitted, audit the predictors for availability. This step is not
usually listed in a comparison protocol and on this dataset it turns out to be the only
step that matters. Each candidate must be checked against one question: could this have
been observed before the loan amount was decided? Section 4(c) shows what the audit
finds.

Data preparation. Separate numeric from categorical. Standardise the numeric
predictors, mandatory here, not optional: ridge and lasso penalise
coefficient magnitude, so on raw scales a variable measured in thousands is penalised
differently from one measured in units, for reasons of measurement rather than
importance. One-hot encode the five categorical variables with a training-set
vocabulary and a defined behaviour for unseen levels.

Splitting. Twenty-five per cent of applicants are set aside and scored once, at the
end. Everything else happens inside the other 75%, where ten folds carry the model
selection. Alpha never sees the held-out quarter. Tune on it and the number it later
reports has become a selection statistic, measuring the search instead of the model.

Training. OLS has no hyperparameter. Ridge and lasso are searched over 61 values
of alpha on a log grid from 0.01 to 10,000, wide enough that the selected value is
interior and not pinned at a boundary. Every transformation sits inside a pipeline
so that the scaler is re-fitted within each fold instead of once on all training data.

Evaluation. RMSE as the primary measure, in the units of the target, because it
penalises the large errors that matter most when sizing credit. MAE alongside it,
being less sensitive to a few extreme applicants. R-squared for the share of variance
explained, the measure that exposes the central finding here. Beyond
accuracy: the number of retained predictors, and the stability of coefficients, since
a model whose coefficients change sign between refits cannot be explained to a credit
committee.

## 4(b) Multicollinearity, and how ridge and lasso respond

Ordinary least squares solves for the coefficient vector that minimises squared error,
and the solution involves inverting the matrix of predictor cross-products. When two
predictors are nearly linear combinations of one another that matrix approaches
singularity, and the consequence is a variance problem and not a bias problem. The
estimates stay unbiased. Their standard errors inflate, sometimes enormously, because
the data cannot identify how to divide a shared effect between two variables that move
together. Only their sum is determined; the split between them is arbitrary and is
settled by noise.

Three practical symptoms follow. Coefficients take implausible magnitudes and
sometimes signs opposite to the simple correlation. Small changes in the sample produce
large changes in the estimates. And the fitted values remain perfectly reasonable, so
the failure is invisible unless the coefficients are inspected, which is why a model
can predict adequately while telling a false story about its drivers.

This dataset carries the problem by construction:

```
predictor                        VIF
MonthlyIncome                  49.84       AnnualIncome  <-> MonthlyIncome   r = 0.990
AnnualIncome                   49.84       Age           <-> Experience      r = 0.983
NetWorth                       40.48       NetWorth      <-> TotalAssets     r = 0.979
TotalAssets                    39.80
Experience                     29.76
Age                            29.65
```

A variance inflation factor of 49.84 means the variance of that coefficient is roughly
fifty times what it would be under orthogonal predictors, so its standard error is
about seven times larger. Annual and monthly income are the same quantity on two
scales; age and experience are the same life-course fact recorded twice.

Ridge adds a penalty on the sum of squared coefficients. This makes the inverted
matrix well-conditioned again by adding a constant to its diagonal, which is why the
variance falls. The estimator becomes biased, and that is the deliberate trade: a small
bias in exchange for a large variance reduction. With two collinear predictors, ridge
splits the shared effect between them rather than choosing, so both survive with
roughly half the coefficient each. Nothing reaches exactly zero.

Lasso penalises the sum of absolute coefficients. The geometry of that constraint
has corners on the axes, so the optimum frequently lands where a coefficient is exactly
zero. With two collinear predictors, lasso tends to keep one and delete the other,
and which one it keeps can be unstable between samples when the two are close
substitutes.

The distinction measured on this data, at a common penalty:

```
coefficients set exactly to zero:   OLS 0    Ridge 0    Lasso 32 of 39
sum of absolute coefficients:       OLS 11,591    Ridge 9,379    Lasso 456
```

Ridge shrank the total coefficient mass by about a fifth while retaining every
predictor. Lasso cut it by 96% and removed 32 of 39. Ridge is the right tool when the
correlated predictors are all genuinely relevant and an interpretable subset is not
required; lasso is the right tool when a smaller model is wanted and one of each
correlated pair can be surrendered.

## 4(c) Implementation and comparison

The audit from 4(a) comes first, and it changes the whole question.

`MonthlyLoanPayment` is the standard amortisation formula applied to the target:

```
MonthlyLoanPayment reconstructed from LoanAmount
  max absolute error 0.000004     correlation 1.00000000

TotalDebtToIncomeRatio = (MonthlyDebtPayments + MonthlyLoanPayment) / MonthlyIncome
  mean absolute difference 0.00000000
```

Both reproduce to machine precision. Neither can exist before the loan amount is
decided, so neither is a predictor of it. `InterestRate`, `BaseInterestRate` and
`RiskScore` are set as part of the same pricing decision and are excluded on the same
grounds.

Fitting all three models on both feature sets:

| Feature set | Model | alpha | Test RMSE | Test MAE | Test R² | Non-zero coefs |
|---|---|---|---|---|---|---|
| With post-decision variables | OLS | | 0.0000 | 0.0000 | **1.0000** | 3 |
| With post-decision variables | Ridge | 0.01 | 0.2103 | 0.1433 | 1.0000 | 44 |
| With post-decision variables | Lasso | 0.398 | 3.4369 | 2.1372 | 1.0000 | 7 |
| Pre-decision only | OLS | | 13,020.63 | 9,779.41 | **−0.0043** | 39 |
| Pre-decision only | Ridge | 10,000 | 12,996.43 | 9,757.30 | −0.0006 | 39 |
| Pre-decision only | Lasso | 199.53 | 12,995.01 | 9,754.04 | −0.0004 | 1 |

The top half is not a model. OLS reaches a test RMSE of exactly zero using three
coefficients because it has rediscovered the amortisation identity and is solving it
backwards. Any analyst who reported R-squared of 1.0 on a credit problem and stopped
there has confirmed only that they did not read the data dictionary.

The bottom half is the honest answer, and it is that the applicant characteristics in
this dataset carry no information about the loan amount requested. An R-squared at or
below zero means the fitted model predicts worse than the training mean. The RMSE of
roughly 13,000 sits at the standard deviation of the target, which is what a model that
has learned nothing produces. Lasso at its cross-validated penalty retains a single
coefficient, which is the correct response: given no signal, the penalty that minimises
prediction error removes almost everything.

Comparing the three on the honest feature set: lasso is marginally best at RMSE
12,995.01, ridge next at 12,996.43, OLS last at 13,020.63. The spread is 26 units on a
target measured in tens of thousands, and reporting a winner on that basis would be
noise-chasing. The regularised models edge ahead only because shrinking coefficients
towards zero is the right move when the coefficients should be zero.

What should be reported to the company. Not a model. The finding is that
`LoanAmount` in this extract is unpredictable from applicant attributes, and that every
variable which appears to predict it is computed from it. Three responses follow:
confirm whether the extract is missing the pre-application fields that would actually
carry signal, such as the amount the applicant requested on the form; reconsider
whether loan amount is the right target, since default risk or approval probability may
be both predictable and more useful; and treat any existing model trained on this
extract as invalid, because it will have been fitted on the leakage and will fail
completely on live applications where the amortisation fields do not yet exist.

## 4(d) Selecting the regularisation parameter

Use k-fold cross-validation on the training set only, with the penalty chosen inside
the loop, and keep the test set for a single final measurement. Ten folds is a
reasonable default at 15,000 training rows.

The mechanism by which this reduces overfitting is worth stating precisely. Alpha
controls the bias-variance trade: too small and the model tracks noise in the training
sample, too large and it cannot represent real structure. Cross-validation estimates
out-of-sample error at each candidate alpha by repeatedly fitting on part of the
training data and scoring on the part held back, so the selected alpha is the one that
generalises best across resamples rather than the one that fits best in sample. Because
no fold's validation rows contribute to the fit that scores them, the curve being
minimised is an estimate of generalisation error and not of training error.

Two refinements matter at this level. Selecting the exact minimum of a noisy
cross-validation curve systematically picks a model slightly too complex, since the
minimum of a noisy curve sits below the true minimum; the one-standard-error rule,
which takes the largest alpha whose error is within one standard error of the best,
gives a simpler model at negligible cost. And if the reported test error is to be an
unbiased estimate of the whole procedure including the choice of alpha, the selection
must be nested inside an outer cross-validation loop instead of being run once.

For this dataset there is a prior step that outranks both. No validation strategy
protects against a predictor computed from the target: cross-validation would have
returned an excellent score at every alpha, because the leak is present in every fold.
Leakage is a data-understanding problem, and no amount of resampling detects it.
# Question Five: crime rates and resource allocation

Worked on `data/crimes.csv`: 2,215 communities across 48 states, 144 variables, no
missing cells and no duplicate rows. Target `crime_rate`, offences per 100,000.

## 5(a) Exploratory strategy

The agency has stated the constraint that decides this analysis: association must not
be read as causation. That is not a caveat for the conclusion, it is a filter on what
enters the analysis at all, and it shapes every step below.

Establish what the sample can support. 2,215 communities across 48 states, with
states contributing very unequally. Any state-level claim rests on however many
communities that state supplied, and several supplied fewer than ten.

Audit quality before interpreting anything. No missing cells and no duplicates
here, unusual enough to be worth confirming rather than assuming, since a dataset with
no missing values has often had them filled in upstream by someone whose choices are
not documented.

Separate the variables by what they are, not by their correlation with the target.
Of 142 numeric columns, 16 are other crime measures: counts and rates for murder, rape,
robbery, assault, burglary, larceny, auto theft and arson. These are the same
phenomenon as the target, not explanations of it, and they belong in a different
category from the 124 socioeconomic and demographic variables. Section 5(c) shows what
happens when that distinction is not made.

Examine distributions before relationships. `crime_rate` has a skew of 2.24, a
median of 374 against a mean of 568, and a range from 0 to 4,877. That shape decides
the plotting axis, the summary statistic and whether a linear model needs a
transformed target.

Treat outliers as findings before treating them as errors. 164 communities, 7.4% of the
sample, sit outside the Tukey fence, and one reports zero crime. In a crime dataset a
high-value tail is expected and not anomalous, so these should be investigated
before any are removed, and removing them would delete the communities the agency most
needs to understand.

Then relationships, and only between variables that could inform a decision.

## 5(b) Visualisations

Each chart below is chosen from the measurement level of the variables involved, which
is the justification the question asks for.

![Q5 visualisations](figures/q5_fig1.png)

(a) One continuous, heavily skewed variable, so a histogram with both centre
measures marked. A bar chart would be wrong, since the variable is continuous and not categorical, and reporting the mean alone would mislead: the mean of 568 sits
above the median of 374 because a minority of high-crime communities pulls it up. The
chart shows the skew of 2.24 directly and makes the 7.4% outlier tail visible.

(b) Two continuous variables, so a scatter with a fitted line. `PctKidsBornNeverMar`
against `crime_rate`, Pearson +0.687 and Spearman +0.667. The scatter shows what a
coefficient cannot: the spread around the line, whether the relationship is linear
throughout, and where the outliers sit relative to it. A bar chart of binned averages
would hide all three.

(c) A categorical grouping against a continuous outcome, so box plots by state.
States with at least 25 communities are shown, ordered by median. Box plots in preference to a bar chart of means, because the question is how much variation exists within each
state, and the answer is that within-state spread is large relative to between-state
differences. That itself is the finding: state is a weak summary of a community's
crime rate, and allocating by state average would misallocate within every state.

(d) Many numeric variables at once, so a correlation heatmap. Discussed in 5(c).

## 5(c) Correlation matrix and its limits

Ranking all 142 numeric variables by correlation with `crime_rate` produces this top ten:

```
  assaultPerPop          +0.8521   another crime measure
  robbbPerPop            +0.7896   another crime measure
  PctKidsBornNeverMar    +0.6867   socioeconomic
  PctKids2Par            -0.6841   socioeconomic
  burglPerPop            +0.6765   another crime measure
  PctFam2Par             -0.6498   socioeconomic
  racePctWhite           -0.6472   socioeconomic
  murdPerPop             +0.6359   another crime measure
  PctTeen2Par            -0.6170   socioeconomic
  PctYoungKids2Par       -0.6119   socioeconomic
```

Four of the top ten are other crime rates. An automatic filter that kept the
strongest correlates would select assault, robbery, burglary and murder rates as
predictors of the crime rate. That model would score superbly and be useless, because
it predicts crime from crime. Worse, for the agency's purpose those variables are not
available in advance: knowing this year's assault rate does not help allocate next
year's resources, and if it did, the agency would not need a model. This is the same
failure as Question Four in a different costume, and it is the first limitation of
correlation-based selection: the coefficient cannot tell a cause from a restatement.

Excluding the crime measures, the correlation matrix among the leading socioeconomic
variables shows the second limitation:

```
                     PctKidsBornNeverMar  PctKids2Par  PctFam2Par  racePctWhite  crime_rate
PctKidsBornNeverMar                 1.00        -0.87       -0.85         -0.80        0.69
PctKids2Par                        -0.87         1.00        0.99          0.71       -0.68
PctFam2Par                         -0.85         0.99        1.00          0.65       -0.65
racePctWhite                       -0.80         0.71        0.65          1.00       -0.65
```

Fifteen pairs among the top nine correlate at 0.75 or above, and `PctKids2Par` with
`PctFam2Par` reach 0.99. These are not nine findings. They are a small number of
underlying factors measured repeatedly. Selecting the top nine by correlation would
build a model with the multicollinearity problems set out in Question Four, and would
report nine drivers where the data supports perhaps two or three.

The third limitation is that correlation measures only straight-line association:

```
                       Pearson  Spearman     gap
  NumKidsBornNeverMar    0.214     0.696   0.482
  NumUnderPov            0.217     0.637   0.421
  NumStreet              0.136     0.439   0.303
  HousVacant             0.247     0.497   0.251
```

`NumKidsBornNeverMar` has a Pearson correlation of 0.214, which a filter with a 0.3
threshold would discard, and a Spearman correlation of 0.696, which is among the
strongest in the dataset. The relationship is monotone but far from linear, and these
are count variables that scale with community size. A linear filter discards them; a
rank-based one keeps them. Which variables "matter" therefore depends on which
coefficient is used, and neither is the right answer on its own.

The fourth limitation is that correlation is symmetric and carries no direction. It
cannot distinguish poverty causing crime from crime causing disinvestment, and nothing
in the matrix resolves which way the arrow points.

## 5(d) Recommendation to the government

Take the strongest socioeconomic correlate, `PctKidsBornNeverMar` at +0.687
(p < 10⁻³⁰⁰). The recommendation has to use that without asserting what it does not
establish.

What the evidence supports. Communities with higher values on this measure have
substantially higher recorded crime rates, and the association is among the strongest
in a dataset of 2,215 communities. That makes the variable usable as an indicator
for targeting: it helps identify where recorded crime is likely to be high, which is
a legitimate basis for deciding where to direct services and analytical attention.

What it does not support. It does not establish that family structure causes
crime, and three alternatives are at least as consistent with the data. The variable
may be a proxy for concentrated poverty, which the correlation matrix supports: it
correlates at 0.81 with `racepctblack` and strongly with several income measures, so
it is partly measuring the economic and historical position of a community rather than
its households. Causation may run the other way, since high-crime environments affect
family stability and household formation. And both may be consequences of a third
factor such as long-term disinvestment, in which case intervening on this variable
changes nothing.

Therefore the recommendation is: use the variable to decide *where* to look, and
do not use it to decide *what to do*. A policy that targeted family structure on this
evidence would be acting on an unproven mechanism, and could stigmatise the communities
it is meant to help.

One point the agency should be told explicitly. `racePctWhite` correlates at −0.647
and `racepctblack` at +0.591. These are among the strongest correlates available, and
they must not enter a resource-allocation model. Racial composition is not a cause of
crime; it is a marker of where segregation, disinvestment and differential enforcement
have concentrated. A model using it would encode historical enforcement patterns as
future predictions and direct policing by race, unlawful in most jurisdictions
and self-confirming in all of them: more policing produces more recorded crime, which
justifies more policing. That feedback loop is the clearest illustration in this
dataset that a strong correlation can be exactly the wrong thing to act on.

Additional evidence that would strengthen the decision. The single most valuable
addition is longitudinal data on the same communities, which this cross-section
cannot substitute for. Observing the same community over time allows within-community
change to be examined, differences out every fixed characteristic of a place, and
begins to address direction: does a change in the socioeconomic measure precede a
change in crime, or follow it?

Three further sources would each address a distinct weakness. A natural experiment or
policy discontinuity, such as an eligibility threshold that assigns a programme to some
communities and not others on an arbitrary cut-off, supports a causal estimate that no
observational correlation can. Independent measurement of crime, such as victimisation
surveys, addresses the fact that recorded crime measures both crime and enforcement
intensity, the confound underlying the race correlation above. And evaluation
evidence from comparable interventions elsewhere indicates whether acting on this
factor has ever produced the intended effect, the question the agency
ultimately needs answered.
