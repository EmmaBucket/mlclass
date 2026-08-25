"""
The split IS the question. Same data, same features, same model -- only the split
changes, and each split answers a different product question.

  cold start : this reader has never used the app  -> train on other readers only
  calibrated : the app watched them read parts 1-2 -> train on others + their own history
  test set   : part 4 (a LATER session) for the held-out readers, in both cases

Normalisation is fit on parts 1-2 only, because at deployment time part 4 hasn't
happened yet. Fitting it on all four parts would be a small time-travel leak.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(HERE, "geco_cache2.pkl"))
for c in ["WORD_TOTAL_READING_TIME", "WORD_RUN_COUNT"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["WORD"] = df["WORD"].astype(str)
df = df.sort_values(["PP_NR", "PART", "TRIAL", "WORD_ID_WITHIN_TRIAL"]).reset_index(drop=True)
read = df["WORD_SKIP"] == 0

readers = sorted(df["PP_NR"].unique())
test_readers = readers[::4][:5]
# NOTE: readers come in two cohorts -- odd-numbered read parts 3-4, even read parts
# 1-2, and the two halves of the novel share no text. So "first session" has to be
# defined per reader, not as a fixed part number.
first_part = df.groupby("PP_NR")["PART"].transform("min")
last_part = df.groupby("PP_NR")["PART"].transform("max")
calib = df["PART"] == first_part          # what the app is allowed to have seen
final = df["PART"] == last_part           # the later session we are judged on
# crowd features only transfer between readers who read the SAME text
cohort = df["PP_NR"].map(df.groupby("PP_NR")["PART"].min())

# --- normalise each reader using ONLY their calibration period ---
df["rt_log"] = np.log1p(df["WORD_TOTAL_READING_TIME"])
c = df.loc[read & calib].groupby("PP_NR")["rt_log"]
df["rt_z"] = (df["rt_log"] - df["PP_NR"].map(c.mean())) / df["PP_NR"].map(c.std())
cut = df.loc[read & calib].groupby("PP_NR")["rt_z"].quantile(0.90)
df["dwell"] = ((df["rt_z"] > df["PP_NR"].map(cut)) & read).astype(int)

# --- crowd baseline from training readers' calibration period only ---
test_cohort = cohort[df["PP_NR"].isin(test_readers)].iloc[0]
crowd = (df.loc[read & ~df["PP_NR"].isin(test_readers) & (cohort == test_cohort)]
           .groupby("WORD_ID").agg(crowd_rt_z=("rt_z", "mean"),
                                   crowd_reread=("WORD_RUN_COUNT", lambda s: (s >= 2).mean())))
df = df.join(crowd, on="WORD_ID")
for k in ["crowd_rt_z", "crowd_reread"]:
    df[k] = df[k].fillna(df[k].mean())

clean = df["WORD"].str.lower().str.strip('.,;:!?\"()—’\'')
df["logfreq"] = np.log1p(clean.map(clean.value_counts())).values
df["word_len"] = df["WORD"].str.len()
df["is_punct"] = df["WORD"].str.fullmatch(r"\W+").fillna(False).astype(int)
df["pos"] = df["WORD_ID_WITHIN_TRIAL"]
by = df.groupby(["PP_NR", "TRIAL"])
df["prev_len"] = by["word_len"].shift(1).fillna(0)
df["next_len"] = by["word_len"].shift(-1).fillna(0)
F = ["word_len", "logfreq", "is_punct", "pos", "prev_len", "next_len", "crowd_rt_z", "crowd_reread"]

others = read & ~df["PP_NR"].isin(test_readers) & (cohort == test_cohort)
own = read & df["PP_NR"].isin(test_readers) & calib
test = read & df["PP_NR"].isin(test_readers) & final

def fit_score(train_mask, name):
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    m.fit(df.loc[train_mask, F].astype(float), df.loc[train_mask, "dwell"])
    p = m.predict_proba(df.loc[test, F].astype(float))[:, 1]
    y = df.loc[test, "dwell"].values
    k = max(1, int(0.10 * len(p)))
    print(f"   {name:38s} AUC={roc_auc_score(y, p):.3f}  AP={average_precision_score(y, p):.3f}"
          f"  precision@10%={y[np.argsort(-p)[:k]].mean():.3f}")
    # does it work for every reader, or only on average?
    per = [roc_auc_score(df.loc[test, "dwell"][df.loc[test, "PP_NR"] == r],
                         p[(df.loc[test, "PP_NR"] == r).values]) for r in test_readers]
    print(f"   {'':38s} per-reader AUC: " + " ".join(f"{a:.2f}" for a in per))

print(f"test = later session of readers {test_readers}: {int(test.sum()):,} words, "
      f"base rate {df.loc[test,'dwell'].mean():.3f}\n")
fit_score(others, "cold start (4 same-text readers)")
fit_score(others | own, "calibrated (+ their own session 1)")
fit_score(own, "personal only (their session 1)")
