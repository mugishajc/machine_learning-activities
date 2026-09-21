# MIT91207 Machine Learning

**Name:** Mugisha Jean Claude
**Registration number:** 26016815
**Programme:** MSc Information Technology, Level 9
**Module:** MIT91207 Machine Learning
**Lecturer:** Dr Gustave Udahemuka

Section A is compulsory. From Section B I have answered **Question Four** and
**Question Five**, the two permitted.

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

# Question One: agricultural low-yield early warning (50 marks)

## 1(a) Evaluating the proposal to deploy on predictive performance alone (12 marks)

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

## 1(b) Formulating the problem (8 marks)

The decision the company is making is a triage decision under a capacity constraint:
which farms does a field officer visit before planting. That framing, rather than the
data, should determine the target.

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
district, which is what an operations manager needs for planning fertiliser volumes
rather than officer visits.

### The recommendation

Fit the regression on the continuous yield and derive the classification by
thresholding it, rather than fitting a separate classifier. The underlying quantity is
continuous and the binary label is an administrative convention applied to it, so
modelling the convention discards information. Running both is reasonable in practice,
and disagreement between them is a useful signal that a farm is atypical and deserves
human review.

One caveat that applies to both framings: the prediction must be made at a point in
the season when intervention is still possible. A model that uses rainfall over the
whole season predicts a yield that has already happened. The feature set is therefore a
function of the decision date, and everything observed after it is unusable no matter
how predictive.
