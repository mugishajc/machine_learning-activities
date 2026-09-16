"""
Q8 - product-launch survey: exploratory analysis aimed at a launch decision.
Charts are chosen from the measurement level of each variable, not by habit.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
RS=42; rng=np.random.default_rng(RS); N=600

age_b=["18-24","25-34","35-44","45-54","55+"]
occ=["student","salaried","self_employed","public_sector","unemployed"]
price_b=["<5k","5-15k","15-30k","30-50k",">50k"]           # RWF per month, ordinal
feats=["offline mode","mobile money","multi-user","analytics","local language"]
d=pd.DataFrame({
 "respondent_id":np.arange(1,N+1),
 "age_group":rng.choice(age_b,N,p=[.22,.34,.22,.14,.08]),
 "occupation":rng.choice(occ,N,p=[.18,.34,.26,.14,.08]),
 "used_similar_before":rng.choice(["yes","no"],N,p=[.46,.54]),
 "expected_price_band":rng.choice(price_b,N,p=[.28,.31,.22,.13,.06]),
 "satisfaction_current":rng.integers(1,6,N)})
pri={"<5k":0,"5-15k":1,"15-30k":2,"30-50k":3,">50k":4}
d["price_rank"]=d.expected_price_band.map(pri)
lo=(-0.55+0.85*(d.used_similar_before=="yes")+0.40*d.price_rank
    -0.42*d.satisfaction_current+0.55*d.age_group.isin(["25-34","35-44"])
    +rng.normal(0,.7,N))
d["willing_to_purchase"]=(1/(1+np.exp(-lo))>rng.random(N)).astype(int)
d["preferred_feature"]=[rng.choice(feats,p=([.12,.34,.10,.14,.30] if a in("18-24","25-34")
                                            else [.24,.26,.16,.12,.22]))
                        for a in d.age_group]
d.loc[rng.random(N)<.05,"satisfaction_current"]=np.nan
d.loc[rng.random(N)<.03,"expected_price_band"]=np.nan
d=pd.concat([d,d.sample(12,random_state=RS)],ignore_index=True)     # duplicated submissions

print("="*80); print("1. Q8(a) DATA QUALITY BEFORE ANY INTERPRETATION")
print(f"  responses={len(d)}  duplicate rows={d.duplicated(subset='respondent_id').sum()}")
d=d.drop_duplicates(subset="respondent_id").reset_index(drop=True)
miss=pd.DataFrame({"missing":d.isna().sum(),"%":(d.isna().mean()*100).round(1)})
print(f"  after removing duplicate submissions: {len(d)}")
print(miss[miss.missing>0].to_string())
m=d.satisfaction_current.isna()
print(f"  willingness among non-responders to satisfaction: {d.willing_to_purchase[m].mean():.3f} "
      f"vs {d.willing_to_purchase[~m].mean():.3f} (n={m.sum()})")
print(f"  overall stated willingness to purchase: {d.willing_to_purchase.mean():.1%}")
print(f"  margin of error on that proportion at 95%: "
      f"+/-{1.96*np.sqrt(d.willing_to_purchase.mean()*(1-d.willing_to_purchase.mean())/len(d))*100:.1f} pp")

def cramers_v(t):
    chi2=stats.chi2_contingency(t)[0]; n=t.values.sum()
    return np.sqrt(chi2/(n*(min(t.shape)-1)))

print("\n"+"="*80); print("2. Q8(b)(c) RELATIONSHIPS, WITH A TEST BEHIND EACH CHART")
print("\n  (i) prior use of a similar product x willingness  [nominal x binary -> grouped bar]")
t1=pd.crosstab(d.used_similar_before,d.willing_to_purchase)
chi2,p1,_,_=stats.chi2_contingency(t1)
print(t1.to_string())
r1=d.groupby("used_similar_before").willing_to_purchase.mean()
print(f"      willingness: prior users {r1['yes']:.1%} vs non-users {r1['no']:.1%}  "
      f"(chi2 p={p1:.2e}, Cramer V={cramers_v(t1):.3f})")

print("\n  (ii) expected price band x willingness  [ordinal x binary -> ordered bar + trend test]")
dd=d.dropna(subset=["expected_price_band"])
t2=pd.crosstab(dd.expected_price_band,dd.willing_to_purchase).reindex(price_b)
rate=(t2[1]/t2.sum(axis=1))
print(pd.DataFrame({"n":t2.sum(axis=1),"willing %":(rate*100).round(1)}).to_string())
rho,p2=stats.spearmanr(dd.price_rank,dd.willing_to_purchase)
print(f"      Spearman rho={rho:+.3f} (p={p2:.2e}): willingness RISES with the price the")
print(f"      respondent already expects to pay, so the constraint is not price sensitivity")

print("\n  (iii) current satisfaction x willingness  [ordinal x binary -> box plot]")
s=d.dropna(subset=["satisfaction_current"])
a=s.satisfaction_current[s.willing_to_purchase==1]; b=s.satisfaction_current[s.willing_to_purchase==0]
u,p3=stats.mannwhitneyu(a,b)
print(f"      median satisfaction, willing={a.median():.1f} vs not willing={b.median():.1f}  "
      f"(Mann-Whitney p={p3:.2e})")
print(f"      mean satisfaction, willing={a.mean():.2f} vs not willing={b.mean():.2f}")

print("\n  (iv) preferred feature x age group  [nominal x nominal -> heatmap of column shares]")
t4=pd.crosstab(d.preferred_feature,d.age_group)[age_b]
share=(t4/t4.sum())*100
print(share.round(1).to_string())
chi2,p4,_,_=stats.chi2_contingency(t4)
print(f"      chi2 p={p4:.2e}, Cramer V={cramers_v(t4):.3f}")

print("\n"+"="*80); print("3. Q8(c) SEGMENTS THAT CARRY THE DECISION")
d["segment"]=np.where(d.used_similar_before.eq("yes")&d.price_rank.ge(2),"experienced, higher budget",
             np.where(d.used_similar_before.eq("yes"),"experienced, lower budget",
             np.where(d.price_rank.ge(2),"new, higher budget","new, lower budget")))
seg=d.groupby("segment").agg(n=("respondent_id","size"),
        willing=("willing_to_purchase","mean"),
        satisfaction=("satisfaction_current","mean")).sort_values("willing",ascending=False)
seg["share of sample"]=seg.n/len(d)
seg["share of all willing"]=d[d.willing_to_purchase==1].segment.value_counts(normalize=True)
print(seg.to_string(float_format=lambda v:f"{v:8.3f}"))
top=seg.index[0]
print(f"\n  '{top}' is {seg.loc[top,'share of sample']:.1%} of respondents but "
      f"{seg.loc[top,'share of all willing']:.1%} of everyone willing to buy")

fig,axes=plt.subplots(2,2,figsize=(14,10))
ax=axes[0,0]
r=d.groupby("used_similar_before").willing_to_purchase.mean()*100
ax.bar(r.index,r.values,color=["#92c5de","#2166ac"]); ax.set_ylabel("willing to purchase (%)")
ax.set_title(f"(a) Prior use vs willingness\nchi-square p={p1:.1e}, Cramer V={cramers_v(t1):.2f}",fontsize=10)
for i,v in enumerate(r.values): ax.text(i,v+1,f"{v:.1f}%",ha="center")
ax=axes[0,1]
ax.bar(range(len(rate)),rate.values*100,color="#4393c3")
ax.set_xticks(range(len(rate))); ax.set_xticklabels(price_b)
ax.set_xlabel("expected monthly price (RWF)"); ax.set_ylabel("willing to purchase (%)")
ax.set_title(f"(b) Willingness by expected price band\nSpearman rho={rho:+.2f}, p={p2:.1e}",fontsize=10)
ax=axes[1,0]
ax.boxplot([b.values,a.values],labels=["not willing","willing"])
ax.set_ylabel("satisfaction with current solution (1-5)")
ax.set_title(f"(c) Satisfaction vs willingness\nMann-Whitney p={p3:.1e}",fontsize=10)
ax=axes[1,1]
im=ax.imshow(share.values,cmap="Blues",aspect="auto")
ax.set_xticks(range(len(age_b))); ax.set_xticklabels(age_b,rotation=45)
ax.set_yticks(range(len(share.index))); ax.set_yticklabels(share.index,fontsize=9)
for i in range(share.shape[0]):
    for j in range(share.shape[1]):
        ax.text(j,i,f"{share.values[i,j]:.0f}%",ha="center",va="center",fontsize=8,
                color="white" if share.values[i,j]>22 else "black")
ax.set_title(f"(d) Preferred feature by age group (column %)\nchi-square p={p4:.1e}",fontsize=10)
fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle("Q8: survey relationships bearing on the launch decision",y=1.00,fontsize=13)
fig.tight_layout(); fig.savefig("figures/q8_fig1_survey.png",dpi=150,bbox_inches="tight"); plt.close(fig)
seg.to_csv("output/q8_segments.csv")
print("\nfigure: figures/q8_fig1_survey.png")
