## 1(c) A leakage-safe workflow, and why the order matters (10 marks)

Leakage is not one failure. It is at least four, they enter at different points, and
the order of operations is what keeps each of them out. The workflow below is stated
in the order it must run, with the reason each step sits where it does.

**Step 1. Fix the decision date and discard everything observed after it.** For a
pre-planting warning the model may use soil, farm size, seed variety, irrigation
status, prior-season yields and the long-run climate normals for the location. It may
not use realised rainfall for the season being predicted, harvest-time pest counts, or
the yield of the neighbouring plot recorded at harvest. This is first because no later
step can repair it. A feature that does not exist at decision time produces a model
that validates well and cannot be served.

**Step 2. Split by farm and by time, before touching the data.** Farms recur, so the
split is grouped on `farm_id`. Seasons progress, so the test period follows the
training period. Splitting first is what makes every subsequent statistic honest.

**Step 3. Fit imputation on the training rows only.** Median fertiliser is computed on
training farms and applied unchanged to the test rows. A missingness indicator is
retained, because in this panel the gaps are not random: they run from 4.3% in
Nyagatare to 28.6% in Karongi, so absence encodes district and extension-service
coverage.

**Step 4. Fit scaling on the training rows only.** Rainfall has a standard deviation
of 141 against 0.53 for soil pH, a ratio of roughly 265 to 1. Any distance-based or
penalised method is otherwise dominated by rainfall for reasons of measurement units.

**Step 5. Encode categorical variables with the training vocabulary.** District and
seed variety are nominal. An unseen category at scoring time must map to a defined
value rather than raise, and it must be distinguishable from a legitimate reference
level.

**Step 6. Select features inside the cross-validation loop, not before it.** Ranking
predictors on the full dataset and then cross-validating the winners reports the error
of the selection, not of the procedure.

**Step 7. Evaluate once on the held-out period.**

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

## 1(d) Diagnosing and responding to decay (10 marks)

The two kinds of change have to be separated because they call for different responses,
and only one of them is visible without labels.

**Covariate shift** is a change in the distribution of the inputs. The relationship
between predictors and yield is intact; the model is simply being asked about farms
unlike those it was fitted on. It is detectable immediately, from unlabelled data.

**Concept drift** is a change in the relationship itself. The same rainfall now implies
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
season-to-season variation rather than set to a textbook constant, since agricultural
inputs vary substantially between years without anything being wrong.

At the prediction level, the distribution of predicted risk and the flag rate per
district. A model whose flagged share jumps without a corresponding change in inputs
is behaving differently for a reason worth finding.

At the outcome level, AUC, calibration and precision or recall at the operating
threshold, computed per season as harvest data lands. This is the only level that
detects concept drift, and it is necessarily a season behind.

The response ladder should be agreed in advance so that nobody is negotiating under
pressure. Recalibration first, which re-fits the mapping from score to probability and
is cheap. Refitting on a rolling window next, which is the correct response to concept
drift and implies discarding the oldest seasons rather than accumulating them.
Re-specification last, if the drivers themselves have changed, for example if
irrigation has become the dominant factor in a way the original feature set did not
anticipate.

Two design choices reduce exposure before any of this is needed. Weighting recent
seasons more heavily in training accepts a little variance for a large reduction in
staleness. And holding out the most recent season as the test set at every refit means
the reported performance is always an estimate of next season rather than of the past.

One caution that applies to the whole scheme: once field officers act on the
predictions, successful interventions prevent the outcomes the model predicts, and
measured performance falls even though the system is working. Unless the intervention
is recorded as a variable, retraining will progressively unlearn the signal that made
the system useful.

## 1(e) Is the regional difference real, or is it the model? (10 marks)

The question cannot be answered by looking at the flag rates, because a difference in
flag rates is exactly what a correct model produces when the underlying rates differ.
The investigation has to separate three candidate explanations.

**First, establish the ground truth rate per district.** On the held-out farms:

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
it, and the amplification is the part that needs justification rather than the gap.

**Second, check whether the model works equally well where it is applied.** AUC ranges
from 0.669 to 0.722, a spread of 0.053, and recall ranges from 0.226 to 0.529, a spread
of 0.303. The recall spread is the one with consequences: in Bugesera the system finds
53% of the farms that go on to have poor yields, in Musanze 23%.

**Third, and this is the step most audits skip, ask whether any of it is
distinguishable from noise.** Every district interval in the table above overlaps every
other. Karongi's AUC of 0.722 is the highest in the table and rests on 200 farms, with
an interval of 0.643 to 0.790 that is 0.147 wide; Nyagatare's interval is 0.056 wide on
1,420 farms. Ranking districts by a point estimate would produce a league table that
reshuffles on the next sample. There is a tempting pattern in the data, a correlation
of +0.865 between a district's missing-data rate and the model's AUC there, and it
rests on six points with overlapping intervals. It is a hypothesis to test on more
districts, not a finding to report.

### What the company should require before the model allocates support

Report every subgroup metric with an interval, and treat the worst credible value
rather than the point estimate as the number that must clear the bar.

Separate the two fairness questions, because they have different answers here.
Calibration asks whether a score of 0.7 means the same thing in Karongi as in
Nyagatare; if it does, the differing flag rates are defensible. Equal opportunity asks
whether a genuinely at-risk farm has the same chance of being found regardless of
district; the recall spread of 0.303 says it does not, and that is the finding that
should block deployment in its current form.

Test the mechanism rather than only the outcome. Missing fertiliser records run from
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
