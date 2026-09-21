# MIT91207 Machine Learning: Assignment 2

Name: Mugisha Jean Claude
Registration number: 26016815
Programme: MSc Information Technology, Level 9
Module: MIT91207 Machine Learning
Lecturer: Dr Gustave Udahemuka


# Question One: a decision tree that memorises

## 1(a) The gap between 99.6% and 59.4%

The model has overfitted, and the configuration makes it close to inevitable.

The evidence is the size and direction of the gap. A 40.2 percentage point fall from
training to validation is not sampling noise; with a validation set of any reasonable
size the standard error on an accuracy estimate is a couple of points at most, so a gap
of this magnitude is structural. The direction rules out the alternatives: a model that
is too simple fails on both sets together, and a model with a data problem such as
label noise usually underperforms on both. Only a model that has fitted the training
sample specifically can do nearly perfectly on it and barely better than guessing
elsewhere.

The configuration supplies the mechanism. With `max_depth = None` the tree grows until
every leaf is pure or cannot be split further, and with `min_samples_leaf = 1` a leaf
containing a single customer is permitted. Together these place no limit on complexity
at all, so the tree keeps partitioning until it has isolated individual training
records. Training accuracy of 99.6% is therefore not a sign that the model learned
something; it is close to what this configuration produces by construction on almost
any dataset.

The validation figure of 59.4% is the honest estimate of what the model knows. For a
churn problem, where the majority class typically sits around 70% to 75%, that number
is worth comparing against a classifier that always predicts "stays". If the majority
class exceeds 59.4%, the tree is performing worse than a model with no inputs, which
would be the strongest possible evidence that the depth has bought nothing.

## 1(b) Bias and variance here

Bias is the error from a model that cannot represent the underlying relationship, the
part that persists no matter how much data is supplied. Variance is the error from
sensitivity to the particular training sample, the part that changes when the sample
changes. Total expected error decomposes into these two plus irreducible noise, and the
two trade against each other as complexity moves.

This model has high variance and low bias.

Low bias follows from the training accuracy. A hypothesis class that reaches 99.6% on
the training data plainly has the capacity to represent the relationship; nothing is
being systematically missed. High variance follows from the gap. The decision boundary
has been fitted to the noise in this particular sample, so a different draw from the
same population would produce a substantially different tree and substantially
different predictions.

A useful way to see it: fit the same configuration on two disjoint halves of the
training data and compare the resulting trees. Under high variance they will differ in
their root split, not merely in their leaves. Under high bias they would agree with each
other and both be wrong.

## 1(c) What the two settings do to complexity

`max_depth` caps the number of successive splits on any path from root to leaf. Setting
it to `None` removes the cap entirely, so the depth is limited only by the data. Since
each additional level can double the number of leaves, the number of regions the tree
carves out can grow exponentially with depth, and each region is fitted from fewer and
fewer observations.

`min_samples_leaf` sets the smallest number of training observations a leaf may contain.
At 1, a split is allowed even when it isolates a single customer, so the tree can
manufacture a rule that describes exactly one person. Any prediction resting on one
observation carries the full variance of that observation.

The two interact rather than acting separately. A depth cap alone still permits tiny
leaves within the permitted depth, and a leaf-size floor alone still permits a very deep
tree of narrow splits. Left at `None` and `1` they remove both constraints at once,
which is why the model can drive training error almost to zero.

## 1(d) Overfitting, underfitting, and what to change

Overfitting is learning the training sample rather than the population: low training
error, high error on unseen data, the two separated by a large gap. Underfitting is
failing to learn either: high training error and high validation error, with the two
close together. The diagnostic is not the level of either number alone but the size of
the gap alongside them.

This model is overfitting, on the evidence in 1(a).

Two configuration changes would help, and they attack complexity from different
directions.

Limit the depth. Setting `max_depth` to a small value, searched over a grid instead of guessed, forces the tree to spend its splits on the partitions that separate the classes
best, and prevents the long chains of narrow splits that fit noise. This is the blunter
instrument and usually the more effective one.

Raise the leaf floor. Setting `min_samples_leaf` to something like 20 or 50 requires
every prediction to rest on a group rather than an individual, which is what makes leaf
estimates stable. `min_samples_split` acts similarly one level up, and
`ccp_alpha` offers a more principled alternative: cost-complexity pruning grows the
tree fully and then removes the branches whose contribution does not justify their
complexity, with the trade governed by a single parameter that cross-validation can
select.

Both values should be chosen by cross-validated search on the training data, for the
reason set out in Question Three: tuning against a single fixed validation set and then
reporting that set's score overstates what the model will do on new customers.

# Question Two: parametric against non-parametric

## 2(a) Reading the two sets of scores

| Model | Training F1 | Validation F1 | Gap |
|---|---|---|---|
| Linear SVM | 0.79 | 0.76 | 0.03 |
| Decision Tree | 0.99 | 0.63 | 0.36 |

A parametric model commits in advance to a fixed functional form with a fixed number of
parameters, and learning consists of estimating those parameters. A linear SVM assumes
the classes can be separated by a hyperplane, so whatever the sample size, it estimates
one weight per feature plus an intercept. The size of the model does not grow with the
data.

A non-parametric model does not fix the form in advance, and its effective number of
parameters grows with the sample. A decision tree decides how many splits to make and
where, so the structure is read off the data rather than imposed on it. On a large
sample the tree can become very large indeed.

That structural difference is what produces their inductive biases, meaning the
assumptions each makes when generalising beyond what it has seen. The SVM's bias is
strong and explicit: it believes the boundary is linear, and it will impose that belief
even where it is wrong, which is why it cannot chase noise but also cannot capture a
genuinely curved boundary. The tree's bias is weak and different in kind: it assumes the
boundary is made of axis-parallel rectangles, and because it can use arbitrarily many of
them, it can approximate almost any shape, including the shape of the noise. Weak bias
buys flexibility and pays for it in variance.

The decision tree shows the stronger evidence of overfitting, by a wide margin. Its gap
of 0.36 is twelve times the SVM's 0.03, and the pattern is the signature: near-perfect
training performance with validation performance far below it. The SVM's small gap says
its capacity is well matched to the problem; it is not memorising, because its form does
not permit memorising.

The practical conclusion is the one that the table makes uncomfortable for the tree. The
SVM is the better model on this evidence, with a validation F1 of 0.76 against 0.63, and
the tree's superior training score is not an argument in its favour but the reason for
its worse validation score. Two qualifications are worth stating. Single-split estimates
carry uncertainty, so the comparison should be repeated under cross-validation before
the SVM is declared the winner. And the tree has not been tuned: the comparison is
between a well-matched linear model and an unconstrained tree, so the fair contest is
against a tree whose complexity has been controlled by the parameters in 2(b).

## 2(b) Three hyperparameters for the grid

max_depth. Controls the maximum number of splits along any root-to-leaf path.
Lowering it stops the tree early, so the leaves cover broader regions and each is
estimated from more observations; raising it lets the tree keep subdividing. Because the
number of leaves can double with each level, this parameter governs complexity more
directly than any other, and constraining it is usually the single most effective
intervention against a 0.36 gap. Sensible grid: 3, 5, 7, 10, 15, None.

min_samples_leaf. Controls the minimum observations permitted in a terminal node. A
split that would leave fewer than this on either side is rejected, so raising it prunes the tree upward from the leaves and prevents leaves built on one or two transactions. The
effect on generalisation is direct: a leaf's predicted probability is estimated from the
observations it contains, so a floor of 50 gives an estimate with roughly a seventh of
the standard error of one based on a single observation. For fraud detection specifically
the floor must be set with the class imbalance in mind, since too high a value can
eliminate the small pure-fraud leaves that carry the signal. Sensible grid: 1, 5, 20, 50, 100.

min_samples_split. Controls the minimum observations a node must hold before it is
eligible to be split at all. It acts one level above `min_samples_leaf`, halting growth
before a split is attempted rather than rejecting the result afterwards, which prunes
whole branches instead of trimming their ends. Raising it produces a shallower tree with
fewer, better-populated decision regions and reduces variance for the same reason.
Sensible grid: 2, 10, 50, 100.

A fourth worth including is `ccp_alpha`, the cost-complexity pruning parameter, which
differs from the three above in that it prunes after growth instead of constraining it during growth. That lets the tree discover useful deep structure first and remove only
the branches whose accuracy gain does not justify their complexity, which often
outperforms pre-pruning at the same effective size.

These parameters interact, so they should be searched jointly over a grid rather than
tuned one at a time, and scored on F1 or average precision rather than accuracy, for the
reason set out in Question Four.

# Question Three: validation done properly

## 3(a) Fifty configurations against one fixed validation set

The practitioner has produced a number that cannot be trusted, and the reason is
selection, not modelling.

Each of the 50 validation scores is an estimate of true performance carrying random
error from the particular composition of that validation set. Taking the maximum of 50
such estimates does not return the best model's true performance; it returns that model's true performance plus the largest favourable error among the 50 draws. The
maximum of a set of noisy estimates is biased upward, and the bias grows with the number
of candidates. With 50 configurations, the winner is partly whichever one happened to
suit the idiosyncrasies of that specific validation split.

The mechanism is that the validation set has been used 50 times to make a choice, so
information about it has flowed into the selected model. It is no longer held out in any
meaningful sense. It has become part of the training procedure, and reporting its score
as an estimate of performance on new data is reporting training performance under a
different name.

Cross-validation should be incorporated by replacing the single fixed split with k folds
inside the training data. Each configuration is scored as the mean across folds, which
reduces the variance of each estimate and therefore reduces how much the maximum is
inflated. It does not eliminate the problem, because the folds are still being reused
across 50 candidates, but averaging over k estimates instead of relying on one makes
each comparison far more stable.

The procedure that separates selection from evaluation is a three-way split, or
equivalently nested cross-validation.

The three-way split holds out a test set at the outset and does not touch it again.
Within the remainder, cross-validation selects the configuration. The chosen
configuration is refitted on the full remainder, and scored once on the test set. That
single score is unbiased because the test set played no part in the selection.

Nested cross-validation generalises this and is preferable when data is limited. An
inner loop selects the hyperparameters; an outer loop estimates the error of the entire
procedure including the selection. What it estimates is subtly different and more
useful: not the performance of one chosen model, but the performance of the method that
chooses models, which is what will be run again on new data.

The discipline underneath both is that any dataset used to make a decision cannot also
be used to evaluate that decision.

## 3(b) K-Fold and Stratified K-Fold with an 11% minority

K-Fold cross-validation partitions the data into k roughly equal folds. The model is
trained k times, each time on k−1 folds and scored on the fold held back, so every
observation is used for validation exactly once and for training k−1 times. The k scores
are averaged for a performance estimate and their spread indicates its stability. The
advantage over a single split is that the estimate does not depend on one arbitrary
partition, and every observation contributes to the evaluation.

Stratified K-Fold adds one constraint: each fold preserves the class proportions of the
full dataset. With an 11% minority, every fold contains approximately 11% minority cases
rather than whatever the random partition delivers.

Two distinct problems arise without stratification at this imbalance.

The first is variance in the estimate. With 11% minority and, say, 1,000 observations
in 5 folds, each fold of 200 holds about 22 minority cases in expectation, but the
random count follows a binomial distribution with a standard deviation of roughly 4.4.
A fold might hold 15 or 30. Metrics that depend on the minority class, which for an
imbalanced problem are the only metrics that matter, are computed on that small and
varying denominator, so recall estimated on 15 cases moves in steps of 6.7 percentage
points per case. The fold-to-fold scores scatter for reasons that have nothing to do
with the model, and the average inherits that noise.

The second is bias in the fits. Each training set has a different class balance, so each
of the k models is fitted on a slightly different problem. The k scores are therefore
not repeated measurements of one quantity, which is what averaging assumes.

At more extreme imbalance a third failure appears: a fold can contain no minority cases
at all, leaving recall undefined and F1 either undefined or silently zero. At 11% with
reasonable fold sizes that is unlikely, but it is the limiting case of the same problem,
and it is why stratification is the default for classification and not an optimisation.

Stratification costs nothing. There is no accuracy trade and no computational penalty,
which is why `StratifiedKFold` should be the default for any classification problem and
`KFold` reserved for regression.

# Question Four: imbalance and an automated workflow

Note on data: `credit.csv` had not been released when this was written. Part (b) runs
against a structurally matched stand-in, a financial dataset with a rare binary outcome
at 5.24% positive. The script reads its file and target from two constants at the top and
discovers feature types from the dataframe, so the supplied file substitutes directly.

## 4(a) Why 98% accuracy can mean nothing

Let the dataset contain N transactions, of which 2% are fraudulent and 98% legitimate.
A classifier that predicts "Legitimate" for every transaction produces:

```
TP = 0            FP = 0
FN = 0.02N        TN = 0.98N

accuracy    = (TP + TN) / N = (0 + 0.98N) / N = 0.98      = 98%
recall      = TP / (TP + FN) = 0 / (0 + 0.02N) = 0        = 0%
precision   = TP / (TP + FP) = 0 / 0, undefined, reported as 0
F1          = 2PR / (P + R) = 0
```

On 100,000 transactions that is 98,000 correct classifications and 2,000 frauds, every
single one of them missed.

Accuracy misleads here because it is a weighted average in which the weights are the
class frequencies, so at 98% prevalence the majority class contributes 98% of the score.
A model can therefore be right about the easy 98% and wrong about the entire 2% that the
system exists to find. The arithmetic gets worse as imbalance deepens: at 0.2% fraud the
same do-nothing classifier scores 99.8%.

For an early-warning system the asymmetry compounds it. Accuracy counts a missed fraud
and a false alarm as one error each, but a missed fraud is a completed theft while a
false alarm is a few minutes of analyst review. Treating them as equal encodes a
preference no fraud team holds. The metrics that survive are the ones computed on the positive class alone: recall, precision, F1, and average precision, which summarises the
precision-recall curve across thresholds instead of committing to one.

## 4(b) The workflow

The script is in this notebook. `credit.csv` has not been released yet, so it runs
against a stand-in with the same structure, a financial dataset with a rare binary
outcome at 5.24% positive. The two constants at the top of the script are the only
change needed when the real file arrives, because nothing downstream refers to a column
by name: the feature types are discovered from the dataframe.

Measured on the stand-in:

```
rows 20,000   predictors 33   positive rate 5.240%   imbalance ratio 1 : 18

baseline unconstrained tree   depth 27   leaves 1,589
                              train F1 0.9995   test F1 0.8234

grid search                   40 configurations x 5 stratified folds = 200 fits
                              scoring: average precision
best                          max_depth 8, min_samples_leaf 10, class_weight None
                              cross-validated average precision 0.8620

held-out test set
  F1-score          0.8780
  average precision 0.8769
  (accuracy 0.9876, against 0.9476 for always predicting the majority class)
```

Three design points are worth drawing out. The search is scored on average precision and not accuracy, for the reason in 4(a). The folds are stratified, so each holds
the same 5.24% positive rate and the fold scores are comparable. And `GridSearchCV` is
fitted on the training partition only, so the test set contributes nothing to the
selection and its score remains an honest estimate.

The grid also produced a result worth reading carefully. The worst configurations are
the shallow trees, `max_depth=3` reaching only 0.6818, while the unconstrained tree
reaches 0.8499 against the winner's 0.8620. On this data the binding risk is
underfitting, and constraining depth to 8 buys only 0.0121 over no limit at all. That is
the opposite of the situation in Question One, which is exactly why the diagnosis must
come from the gap between training and validation performance and can never be read off
the configuration.

# Question Five: preprocessing a heterogeneous credit dataset

Note on data: `home.csv` had not been released when this was written. Part (c) runs
against a credit dataset carrying the same structure the question describes: numeric
variables on very different ranges, nominal categoricals, and an ordinal education
variable. Two constants at the top of the script switch it to the supplied file.

## 5(a) The preprocessing strategy

Four variable types are present and each needs a different treatment.

Missing numerical values, such as `AMT_INCOME_TOTAL` and `DAYS_EMPLOYED`, take the median of the training rows and not the mean, because income distributions are
right-skewed and a mean is pulled by the upper tail. A binary indicator is retained for
each imputed column, since in credit data a blank is rarely random: an applicant who
does not state income differs systematically from one who does. `DAYS_EMPLOYED` needs a
prior check, because in the Home Credit file it carries a sentinel value of 365243 for
the unemployed, which is not missing data but a coded category and must be separated
before any statistic is computed.

Missing categorical values become their own level, `not_reported`, and are never
imputed with the mode. Mode imputation invents a fact about the applicant; an explicit
level lets the model decide what the absence means.

High-cardinality nominal variables such as `ORGANIZATION_TYPE`, with dozens of levels,
should not be one-hot encoded naively, since that produces dozens of columns each
carrying almost no data. Two better options exist. Grouping rare levels into `other`,
by a frequency threshold set on the training data, keeps the common categories separate
and pools the tail. Target encoding replaces each level with the mean outcome for that
level, which is compact but must be computed out-of-fold or it leaks the target
directly; a smoothed version shrinks rare levels towards the global mean.

Ordinal variables such as `NAME_EDUCATION_TYPE` are encoded with integers in the correct
sequence, stated explicitly in code. Leaving the order to alphabetical accident is the
common error: alphabetically, "Higher education" precedes "Secondary", which inverts the
real ranking.

Numerical scaling is standardisation, fitted on training rows only. Min-max is avoided
because its range is set by the single most extreme value, and credit data has extreme
values by nature.

The consequences of getting this wrong differ sharply by algorithm, and the question
asks specifically about that contrast. Tree-based classifiers split on thresholds, so
they are invariant to any monotone rescaling: standardising or not changes nothing about
which splits are available. They are, however, sensitive to encoding. One-hot encoding a
40-level variable gives a tree 40 weak binary questions instead of a single informative multi-way one, which dilutes its splitting and biases feature-importance measures
towards high-cardinality fields.

Distance-based and penalised methods are the mirror image. They are extremely sensitive
to scaling, because a variable measured in the hundreds of thousands dominates any
Euclidean distance and attracts a disproportionate share of any penalty, purely because
of its units. Question Eight measures this directly: with one feature spanning 0 to
216,444 and another 0 to 0.88, the larger contributes 99.999529% of the total variance,
and K-Means becomes a one-dimensional split on that feature alone. They are equally
sensitive to the ordinal error, since an arbitrary integer code imposes both an order
and equal spacing that the data does not support.

## 5(b) The mechanism of the leak, and what a Pipeline does about it

The mechanism is that a statistic computed over the whole dataset is a function of every
row, including the rows that will later serve as validation. Write the imputation value
as m = median(x₁, …, x_N). If row j is held out for validation, m still depends on x_j.
When x_j is missing and filled with m, it is filled with a number partly derived from
itself, and the same applies to the scaler's mean and standard deviation, which are
sums over all N observations. The validation fold is no longer unseen, so the score it
produces is not an estimate of performance on new data.

Measured on the stand-in, the fold statistics do differ:

```
median of AnnualIncome over the whole dataset : 48,428.50
  fold 1: 48,567.00   difference +138.50
  fold 2: 48,352.50   difference  -76.00
  fold 3: 48,482.50   difference  +54.00
  fold 4: 48,442.00   difference  +13.50
  fold 5: 48,338.50   difference  -90.00
spread across folds 228.50
```

The honest finding is that the consequence is negligible here. Fitting the transformers
on everything and then cross-validating gives ROC AUC 0.9097; fitting them inside the
pipeline gives 0.9097. Repeating at sample sizes from 200 to 20,000, and swapping
StandardScaler for MinMaxScaler, the largest gap observed is 0.008 and several are
negative, which is noise. I expected MinMaxScaler to show a clear penalty, since its
minimum and maximum are each set by a single row, and it did not.

That result does not license the practice. It locates the danger instead of dismissing it. The correct approach costs nothing, so the argument for discipline stands on its
own. And the measurement usefully separates two things often taught as one: preprocessing
leakage of this kind is a hygiene issue with a small measured penalty, whereas a
predictor computed from the target is a different category of failure, undetectable by
cross-validation because it is present in every fold, and fatal rather than marginal.

A `Pipeline` prevents the hygiene failure structurally. When `cross_val_score` or
`GridSearchCV` evaluates a pipeline, it calls `fit` on the entire object separately for
each fold, so every transformer estimates its parameters from that fold's training rows
alone and then only `transform` is applied to the held-out rows. The guarantee is
enforced by the fit/transform separation and not by the analyst remembering the order. `ColumnTransformer` extends the same guarantee across branches, applying the
numeric, ordinal and nominal treatments in parallel to their own column subsets and
concatenating the result, so one object carries the whole preprocessing specification
and can be persisted and reloaded at scoring time unchanged.

## 5(c) The script

Implemented in this notebook. It identifies the three feature types from the dataframe, not from a hand-written list, builds one branch per type, combines them with
`ColumnTransformer`, places a classifier in the parent `Pipeline`, and evaluates on a
stratified held-out split. Measured on the stand-in: design matrix 46 columns after
encoding, held-out ROC AUC 0.9197 and average precision 0.7836 against a base rate of
0.100.

# Question Six: thresholds and the metrics that survive imbalance

A discrepancy first: the question states a background fraud rate of approximately 3%,
but the matrix supplied gives 150 fraudulent transactions in 15,000, which is 1.0%. All
figures below follow the matrix, since that is the evidence given. The discrepancy is
noted rather than resolved.

## 6(a) The four metrics

With TP = 120, FN = 30, FP = 450, TN = 14,400:

```
precision   = TP / (TP + FP) = 120 / (120 + 450) = 120 / 570    = 0.2105  = 21.05%
recall      = TP / (TP + FN) = 120 / (120 + 30)  = 120 / 150    = 0.8000  = 80.00%
specificity = TN / (TN + FP) = 14400 / (14400 + 450) = 14400 / 14850 = 0.9697 = 96.97%
F1          = 2PR / (P + R) = 2(0.2105)(0.8000) / (0.2105 + 0.8000)
            = 0.336842 / 1.010526 = 0.3333
```

Precision of 21.05% says that of every hundred transactions the system flags, about 21
are genuinely fraudulent and 79 are legitimate customers being inconvenienced. It is the
cost side of the system, paid in analyst time and customer friction.

Recall of 80.00% says the system catches four of every five frauds. It is the benefit
side, and the twenty per cent it misses are completed thefts.

Specificity of 96.97% says that of legitimate transactions, 97% pass through correctly.
It reads reassuringly high, and that is the trap: with 14,850 legitimate transactions,
the remaining 3.03% still amounts to 450 false alarms, nearly four times the number of
frauds caught. High specificity and poor precision coexist comfortably when the negative
class dominates.

F1 of 0.3333 is the harmonic mean, and it sits close to the lower of the two inputs
and not midway between them. That is the point of using it: a model cannot hide a
precision of 0.21 behind a recall of 0.80.

For contrast, accuracy is (120 + 14,400) / 15,000 = 0.9680, while a model predicting
"legitimate" for everything would score 0.9900 and catch nothing. The do-nothing model
beats the working one on accuracy.

## 6(b) Raising the threshold from 0.50 to 0.80

A higher threshold demands more confidence before flagging, so fewer transactions are
flagged. Mechanically:

| Quantity | Direction | Why |
|---|---|---|
| False positives | Fall | Borderline legitimate transactions no longer clear the bar |
| Precision | Rises | The flagged set shrinks towards the highest-scoring cases, which are enriched in fraud |
| False negatives | Rise | Genuine frauds with middling scores now pass unflagged |
| Recall | Falls | Fewer of the frauds present are caught |

Simulated on score distributions calibrated so that a threshold of 0.50 reproduces the
supplied matrix:

```
 threshold   TP   FN    FP     TN   precision   recall      F1
      0.30  144    6  2430  12420      0.0559   0.9600  0.1057
      0.50  123   27   475  14375      0.2057   0.8200  0.3289
      0.60   98   52   164  14686      0.3740   0.6533  0.4757
      0.70   70   80    28  14822      0.7143   0.4667  0.5645
      0.80   39  111     5  14845      0.8864   0.2600  0.4021
      0.90   13  137     0  14850      1.0000   0.0867  0.1595
```

At 0.80, precision improves from 0.21 to 0.89 while recall collapses from 0.82 to 0.26.
The system would flag almost nothing incorrectly and miss three frauds in four.

The threshold should be driven by the relative costs, not by the metric. If a missed
fraud costs C_FN and a false alarm costs C_FP, the operating point should minimise
C_FN x FN + C_FP x FP. For card fraud the ratio is typically large: an average
fraudulent transaction might cost a few hundred dollars while a review costs a few
dollars of analyst time, so a ratio near 100 to 1 pushes the threshold well below 0.50,
not above it. Two constraints bound that reasoning in practice. Analyst capacity caps
the number of alerts that can be worked, so the threshold cannot fall below the point
that saturates the team. And customer friction from false declines carries a cost that
does not appear in either figure, since a wrongly blocked card damages a relationship
well beyond the value of the transaction.

## 6(c) ROC against precision-recall

The ROC curve plots recall against the false positive rate as the threshold varies.
Both axes are normalised within a class: recall by the positives, false positive rate
by the negatives. It answers how well the model separates the two classes irrespective
of how many of each exist.

The precision-recall curve plots precision against recall. Precision has TP + FP in its
denominator, mixing both classes, so the curve depends on prevalence.

That difference is why the PR curve is more informative under imbalance. The false
positive rate divides by the large negative class, so a large number of false alarms
produces a small movement along the ROC axis. In this matrix, 450 false positives
against 14,850 legitimate transactions is a false positive rate of 0.0303, which places
the operating point comfortably high on the ROC curve, while precision is 0.2105 and
tells the honest story: most alerts are wrong. ROC AUC also has a fixed baseline of 0.5
regardless of prevalence, whereas the PR baseline equals the positive rate, here 0.01,
so a PR curve is judged against the difficulty of the actual problem.

Both have a use. ROC is the right comparison when the class balance may change between
development and deployment, since it is invariant to prevalence. PR is the right
comparison when the operational question is what fraction of alerts are worth acting on.
For fraud detection that is the question being asked.

## 6(d) Threshold sweep in code

Implemented in this notebook alongside the metric calculations. The pattern is:

```python
prob = model.predict_proba(X_test)[:, 1]        # probability of the positive class
for tau in np.arange(0.05, 0.96, 0.05):
    pred = (prob >= tau).astype(int)            # apply a custom threshold
    print(tau, f1_score(y_test, pred), recall_score(y_test, pred))
```

Three details matter. Column 1 of `predict_proba` is the positive class, and taking
column 0 silently inverts the model. The sweep must run on a validation split, not the
test set, or the chosen threshold inherits the same selection bias described in Question
Three. And the choice among the resulting rows is made on the cost function from 6(b),
not by maximising F1, since F1 encodes an equal weighting of precision and recall that
almost no fraud operation actually holds.

# Question Seven: OLS, Ridge and Lasso

Note on data: `prices.csv` had not been released when this was written. The stand-in is a
regression problem carrying more severe multicollinearity than the house-price example in
the question: 103 predictor pairs correlate at 0.90 or above and one pair reaches 1.000.

`prices.csv` has not been released, so the stand-in is a regression problem with more
severe collinearity than the house-price example: 103 predictor pairs correlate at 0.90
or above and one pair reaches 1.000.

## 7(a) Experimental procedure

Partition first: 25% held out and scored once, the remaining 75% carrying all model
selection through ten-fold cross-validation. Nothing about the test set may inform any
choice.

Preprocess inside the pipeline. Standardisation is mandatory here, not cosmetic, because Ridge and Lasso penalise coefficient magnitude: on unscaled
data a predictor measured in thousands attracts a smaller coefficient and therefore a
smaller penalty than one measured in units, so the regularisation would reflect
measurement conventions instead of importance. OLS is scale-invariant in its predictions,
but scaling it too keeps the three comparable.

Tune alpha by cross-validation on the training folds only, over a wide logarithmic grid
so the selected value sits inside the range rather than at a boundary. OLS has nothing
to tune.

Evaluate once on the held-out set, reporting MAE, RMSE and R² together. RMSE and MAE
are both in the units of the target and differ in how they treat large errors; R² is
unitless and says how much variance is explained relative to predicting the mean.

## 7(b) What collinearity does, and how the two penalties answer it

OLS minimises squared error, and the solution requires inverting the matrix of predictor
cross-products. When one predictor is nearly a linear combination of others, that matrix
approaches singularity and its inverse becomes enormous. The estimates stay unbiased;
their variance explodes. The data determines the joint contribution of a correlated
group but cannot apportion it among the members, so noise settles the split.

The measurement makes this concrete. Two predictors in the stand-in correlate at 1.000,
and OLS assigns them:

```
PolicPerPop      +181,585.315
LemasSwFTPerPop  -181,554.078
```

Two coefficients of roughly 181,000 in opposite directions, summing to about 31. The
model has discovered that it can add an arbitrarily large multiple of one and subtract
the same multiple of the other without changing a single prediction. Read literally it
claims two near-identical measures of police staffing have vast and opposing effects on
crime, which is not a finding but the arithmetic of a near-singular matrix.

Ridge adds a penalty on the sum of squared coefficients. Adding a constant to the
diagonal of the cross-product matrix makes it well-conditioned again, which is precisely
why the variance falls. The estimator becomes biased, and that is the intended trade.
Because the squared penalty grows steeply, Ridge prefers several small coefficients to
one large one, so it splits a shared effect between correlated predictors and does not choose: the same pair becomes −0.705 and −0.714.

Lasso penalises the sum of absolute coefficients. The mechanism for exact zeros is the
geometry of that constraint. The L1 constraint region is a diamond with vertices on the
axes, and the optimum occurs where the elliptical contours of the squared-error surface
first touch that region. Because the diamond has corners while the ellipse is smooth,
contact happens at a corner for a wide range of positions, and a corner is exactly a
point where one or more coefficients equal zero. The L2 constraint region is a sphere
with no corners, so contact almost never occurs on an axis and Ridge shrinks without
eliminating. Equivalently, the absolute-value penalty has a constant gradient right down
to zero, so it keeps pushing a small coefficient all the way there, while the squared
penalty's gradient vanishes as the coefficient approaches zero. Lasso set both members of
the r = 1.000 pair to exactly zero.

```
coefficients set exactly to zero:  OLS 0   Ridge 0   Lasso 82 of 124
sum of absolute coefficients:      OLS 373,211.3   Ridge 1,381.7   Lasso 1,218.3
largest single coefficient:        OLS 181,585.3   Ridge 59.4   Lasso 194.9
```

## 7(c) The script and its results

| Model | alpha | MAE | RMSE | R² | Non-zero coefficients |
|---|---|---|---|---|---|
| OLS | | 245.7568 | 358.6590 | 0.5400 | 124 |
| Ridge | 650.97 | 242.8342 | 362.2963 | 0.5306 | 124 |
| Lasso | 6.26 | 240.0126 | 358.3467 | 0.5408 | 42 |

The result that matters is not which R² is highest. It is that OLS's largest coefficient
is 181,585 and Ridge's is 59.4, a factor of three thousand, while their R² values differ
by 0.009. The predictions are almost identical; the explanations are not remotely. A
practitioner reading only the error metrics would never learn that one of these models
is telling a false story about what drives the outcome.

## 7(d) Interpreting the supplied results

The question's figures are OLS 0.70, Ridge 0.76, Lasso 0.74.

The pattern says the OLS fit was inflated by variance that regularisation removed:
Ridge gains 0.06 over OLS, which is the signature of multicollinearity or of a
predictor count large relative to the sample. Ridge beating Lasso slightly suggests most
predictors carry some signal, since Lasso's advantage appears when many are genuinely
irrelevant, and its 0.02 shortfall is what it paid for deleting some that were not.

Four considerations should be examined before choosing, and none is visible in the R²
column.

Whether the differences exceed noise. Three numbers from one test split have sampling
error, and 0.76 against 0.74 may not be a real ordering. Repeated cross-validation with
intervals should settle it before anything is declared best.

How many predictors each retains. If Lasso reaches 0.74 with 15 predictors while Ridge
needs all 80 for 0.76, the two points of R² are bought with 65 additional fields to
collect, validate, store and monitor. That is an operating cost, not a free gain.

Whether the coefficients are stable and readable. The stand-in demonstrates that OLS
coefficients can be meaningless while its predictions are fine. For a property valuation
that a customer may query, a model whose coefficients flip sign between refits is not
defensible whatever its R².

What the residuals show. R² is an average. A model that fits mid-range houses well and
fails on the expensive tail may be worse commercially than its R² suggests, since the
errors that matter most are concentrated where the money is.

My recommendation on these figures is Ridge, provided repeated cross-validation confirms
the gap, with Lasso preferred if its feature reduction is substantial, because the
simpler data pipeline usually outweighs two points of R².

# Question Eight: K-Means on unstandardised features

## 8(a) Four causes of the 95% single cluster

Differences in feature scale, and this is the dominant cause. K-Means minimises squared
Euclidean distance, and a feature spanning 0 to 20,000 contributes squared differences
up to 400,000,000 while a feature spanning 0 to 1 contributes at most 1. Reproduced on a
comparable dataset, the large feature accounts for 99.999529% of the total variance, so
the algorithm is effectively clustering a single variable and the other three are decoration.
Measured: on raw features the largest cluster holds 82.7%; after standardisation it holds
46.2%.

Uninformative features. Variables carrying no group structure contribute distance without
contributing signal, so genuine separation is diluted by noise dimensions. Dropping a
deliberately uninformative column raised the silhouette from 0.2252 to 0.3202 with no
other change.

Outliers. K-Means places centroids at means, and a mean has no resistance to extreme
values. A handful of very high-spend customers can pull a centroid far enough that it
serves only them, leaving one enormous cluster and one or two tiny ones. This is exactly
the reported symptom, and it is why the cluster-size distribution should be inspected
before the score is.

An inappropriate k. If the data contains one dense region and a diffuse tail, asking for
three clusters may yield the dense mass plus two fragments of the tail. K-Means also
partitions convex regions of similar extent, so if the true structure is elongated or
of very different densities, no value of k produces balanced clusters.

## 8(b) A workflow

Inspect first: distributions, ranges, missing values and the correlation structure,
because two near-duplicate features silently double the weight of whatever they measure.

Handle missing values by imputation with an indicator, or by dropping columns that are
mostly empty. K-Means cannot accept missing values at all, so this is not optional.

Select features on whether they express the behaviour being segmented. Purchasing and
usage variables belong; identifiers, near-duplicates and fields with almost no variance
do not.

Treat outliers explicitly: winsorise at a stated percentile, apply a log transform to
right-skewed monetary variables, or use a robust scaler. Removing them outright is
defensible only if they are errors, since the highest-spending customers are usually the
segment the business cares most about.

Standardise. This is the step whose omission caused the failure, and it must be fitted on
training data if any held-out evaluation is planned.

Choose k using inertia and silhouette together, as in 8(c), and constrain the answer with
what the business can act on. Eleven segments that no one can staff is not a solution.

Fit with multiple random initialisations, since K-Means converges to local optima and a
single start can produce a poor partition.

Profile the result: cluster sizes, feature means per cluster against the overall mean,
and a name for each segment that a non-technical colleague would recognise. A cluster
that cannot be described is not usable.

## 8(c) Inertia at k = 3 against silhouette at k = 4

The two measure different things. Inertia is the within-cluster sum of squared distances
to centroids, and it falls monotonically as k rises, reaching zero when every point is
its own cluster. It therefore cannot be maximised or minimised for a choice; the elbow
method looks for the k after which additional clusters stop buying much reduction.
Reading an elbow is a judgement, and on a smooth curve there may be none.

Silhouette measures, for each point, how much closer it sits to its own cluster than to
the nearest other, averaged over all points and bounded between −1 and 1. Unlike inertia
it has an interpretable optimum, and it rewards separation and not compactness alone.

Given an elbow at k = 3 and a silhouette maximum of 0.56 at k = 4, neither result
overrides the other and the disagreement is itself informative: it says the structure is
not sharply defined, and both answers are defensible. Practically, 0.56 indicates
reasonable separation, so k = 4 has the stronger quantitative support, and the elbow at
3 suggests the fourth cluster adds less compactness than the first three. The tie-break
is interpretability. Fit both, profile both, and adopt whichever produces segments the
business can describe and act on. If the fourth cluster is a coherent group with distinct
behaviour, take four; if it is a thin slice of an existing segment, take three.

A caution the measurement makes vivid. In the reproduction, the raw unstandardised
clustering scored a silhouette of 0.7132, far higher than any standardised alternative,
while being the useless one that put 82.7% of customers in a single cluster. It scored
well because on a single dominant dimension the clusters really are well separated. A
high silhouette does not mean a useful segmentation, and neither score should be read
without the cluster sizes beside it.

## 8(d) Why a distinct cluster need not be a segment

K-Means always returns k clusters. It partitions the space whether or not real groups
exist, and the algorithm has no notion of whether a boundary corresponds to anything a
business could act on. A cluster is a region of feature space, and a segment is a set of
customers who respond differently to something the company can do. The first does not
imply the second.

Three specific gaps. A cluster may be statistically distinct and behaviourally
meaningless, separated on a variable nobody can influence. It may be unstable, appearing
with one random seed or one year of data and not the next, which matters because a
segmentation is meant to persist. And it may be uneconomic: forty customers may form a
tight cluster that cannot justify a campaign.

Two kinds of additional evidence should be required before the clusters drive strategy.

Stability evidence. Refit on bootstrap resamples, on different seeds and on a later time
period, and measure whether the same customers group together, using something like the
adjusted Rand index. A segmentation that does not reproduce is an artefact of one sample.

Domain and outcome validation. Present the profiles to the people who know the customers
and ask whether the groups correspond to anything they recognise. Then test the segments
against outcomes not used in the clustering, such as churn, response to a past campaign
or lifetime value. If the segments differ on those, they are real and actionable. The
decisive evidence is a controlled test: run a differentiated treatment on a sample of
one segment and a control, and measure whether the response differs as the segmentation
predicts.