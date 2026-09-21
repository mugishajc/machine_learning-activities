"""
Question Six - metrics from the supplied confusion matrix, and threshold behaviour.

The matrix is given in the question, so every figure here is exact arithmetic
rather than an estimate from a fitted model.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

TP, FN = 120, 30          # actual fraudulent: predicted fraudulent, predicted legitimate
FP, TN = 450, 14400       # actual legitimate: predicted fraudulent, predicted legitimate
N = TP+FN+FP+TN

print("="*78); print("1. THE MATRIX")
print(pd.DataFrame([[TP,FN],[FP,TN]],
      index=["actual fraudulent","actual legitimate"],
      columns=["predicted fraudulent","predicted legitimate"]).to_string())
print(f"\n  total transactions {N:,}   actual fraud {TP+FN:,} ({(TP+FN)/N:.3%})")
print(f"  note: the question states a background fraud rate of about 3%, but the matrix")
print(f"  itself gives {TP+FN}/{N} = {(TP+FN)/N:.2%}. All figures below follow the matrix,")
print(f"  since that is the evidence supplied. The discrepancy is noted, not resolved.")
print(f"  flagged {TP+FP:,} ({(TP+FP)/N:.3%} of all traffic)")

print("\n"+"="*78); print("2. METRICS, SHOWN AS ARITHMETIC")
prec = TP/(TP+FP); rec = TP/(TP+FN); spec = TN/(TN+FP)
f1 = 2*prec*rec/(prec+rec); acc = (TP+TN)/N
print(f"  precision   = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {TP}/{TP+FP} = {prec:.4f}  ({prec:.2%})")
print(f"  recall      = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {TP}/{TP+FN} = {rec:.4f}  ({rec:.2%})")
print(f"  specificity = TN / (TN + FP) = {TN} / ({TN} + {FP}) = {TN}/{TN+FP} = {spec:.4f}  ({spec:.2%})")
print(f"  F1          = 2PR/(P+R) = 2({prec:.4f})({rec:.4f}) / ({prec:.4f} + {rec:.4f})"
      f" = {2*prec*rec:.6f} / {prec+rec:.6f} = {f1:.4f}")
print(f"\n  for contrast, accuracy = (TP+TN)/N = ({TP}+{TN})/{N} = {acc:.4f} ({acc:.2%})")
print(f"  a model predicting 'legitimate' for everything would score "
      f"{(TN+FP)/N:.4f} ({(TN+FP)/N:.2%}) and catch no fraud at all")

print("\n"+"="*78); print("3. QUESTION FOUR(a): THE 2% CASE, WORKED")
p_fraud = 0.02
print(f"  with a {p_fraud:.0%} fraud rate, the all-legitimate classifier gives:")
print(f"    accuracy    = TN/N = {1-p_fraud:.2f} = {1-p_fraud:.0%}")
print(f"    recall      = TP/(TP+FN) = 0 / (0 + {p_fraud:.2f}N) = 0")
print(f"    precision   = TP/(TP+FP) = 0/0, undefined and conventionally reported as 0")
print(f"    F1          = 0")
print(f"  on 100,000 transactions that is {int(0.98*100000):,} correct and "
      f"{int(0.02*100000):,} frauds missed, every one of them.")

print("\n"+"="*78); print("4. WHAT RAISING THE THRESHOLD DOES")
rng = np.random.default_rng(42)
n_f, n_l = TP+FN, TN+FP
# scores chosen so that tau = 0.50 reproduces the supplied matrix closely
# Beta parameters solved so that tau = 0.50 reproduces the supplied matrix:
# fraud P(score >= .5) = 120/150, legitimate P(score >= .5) = 450/14850
s_f = np.clip(rng.beta(3.8770, 2.0, n_f), 0, 1)
s_l = np.clip(rng.beta(0.9824, 5.0, n_l), 0, 1)
rows = []
for t in [0.30, 0.50, 0.60, 0.70, 0.80, 0.90]:
    tp = int((s_f >= t).sum()); fn = n_f-tp
    fp = int((s_l >= t).sum()); tn = n_l-fp
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn)
    rows.append({"threshold": t, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                 "precision": p, "recall": r,
                 "F1": 2*p*r/(p+r) if p+r else 0.0})
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
print("\n  raising the threshold demands more confidence before flagging, so fewer alerts")
print("  are raised: false positives fall, precision rises, false negatives rise and")
print("  recall falls. The two move in opposite directions by construction.")
t.to_csv("output/q6_thresholds.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
a = ax[0]
a.plot(t.threshold, t.precision, "o-", label="precision", lw=1.8)
a.plot(t.threshold, t.recall, "s-", label="recall", lw=1.8)
a.plot(t.threshold, t.F1, "^--", label="F1", lw=1.4)
a.axvline(0.50, color="k", ls=":", lw=1); a.axvline(0.80, color="#b2182b", ls=":", lw=1)
a.set_xlabel("decision threshold"); a.set_ylabel("score"); a.legend(fontsize=8)
a.set_title("(a) Precision and recall move in opposite directions", fontsize=10)
a = ax[1]
a.plot(t.threshold, t.FP, "o-", label="false positives", color="#b2182b", lw=1.8)
a.plot(t.threshold, t.FN, "s-", label="false negatives", color="#2166ac", lw=1.8)
a.set_xlabel("decision threshold"); a.set_ylabel("count"); a.legend(fontsize=8)
a.set_title("(b) The two error types trade against each other", fontsize=10)
fig.tight_layout(); fig.savefig("figures/q6_fig1.png", dpi=150); plt.close(fig)
print("\nfigure: figures/q6_fig1.png")
