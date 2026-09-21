# MIT91207 Machine Learning: Assignment 2

Name: Mugisha Jean Claude
Registration number: 26016815
Programme: MSc Information Technology, Level 9
Module: MIT91207 Machine Learning
Lecturer: Dr Gustave Udahemuka

## How to read this

Every number quoted is produced by a cell in this notebook. Where the question
supplies figures, such as the confusion matrix in Question Six, the arithmetic is shown
in full and computed rather than asserted.

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

Limit the depth. Setting `max_depth` to a small value, searched over a grid rather than
guessed, forces the tree to spend its splits on the partitions that separate the classes
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

**max_depth.** Controls the maximum number of splits along any root-to-leaf path.
Lowering it stops the tree early, so the leaves cover broader regions and each is
estimated from more observations; raising it lets the tree keep subdividing. Because the
number of leaves can double with each level, this parameter governs complexity more
directly than any other, and constraining it is usually the single most effective
intervention against a 0.36 gap. Sensible grid: 3, 5, 7, 10, 15, None.

**min_samples_leaf.** Controls the minimum observations permitted in a terminal node. A
split that would leave fewer than this on either side is rejected, so raising it prunes
the tree from the bottom and prevents leaves built on one or two transactions. The
effect on generalisation is direct: a leaf's predicted probability is estimated from the
observations it contains, so a floor of 50 gives an estimate with roughly a seventh of
the standard error of one based on a single observation. For fraud detection specifically
the floor must be set with the class imbalance in mind, since too high a value can
eliminate the small pure-fraud leaves that carry the signal. Sensible grid: 1, 5, 20, 50, 100.

**min_samples_split.** Controls the minimum observations a node must hold before it is
eligible to be split at all. It acts one level above `min_samples_leaf`, halting growth
before a split is attempted rather than rejecting the result afterwards, which prunes
whole branches instead of trimming their ends. Raising it produces a shallower tree with
fewer, better-populated decision regions and reduces variance for the same reason.
Sensible grid: 2, 10, 50, 100.

A fourth worth including is `ccp_alpha`, the cost-complexity pruning parameter, which
differs from the three above in that it prunes after growth rather than constraining it
during growth. That lets the tree discover useful deep structure first and remove only
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
such estimates does not return the best model's true performance; it returns the best
model's true performance plus the largest favourable error among the 50 draws. The
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
across 50 candidates, but averaging over k estimates rather than relying on one makes
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
and it is why stratification is the default for classification rather than an
optimisation.

Stratification costs nothing. There is no accuracy trade and no computational penalty,
which is why `StratifiedKFold` should be the default for any classification problem and
`KFold` reserved for regression.
