"""
Word-level "is this reader struggling right now" came out at AUC 0.50 -- chance.
Two possible reasons, and they lead to opposite conclusions:
   (a) personal struggle genuinely isn't predictable  -> adaptation must be reactive
   (b) a single word is too noisy a place to look     -> zoom out and it appears
This script tests (b) by asking the same question about a whole paragraph.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(HERE, "geco_cache2.pkl"))
for c in ["WORD_TOTAL_READING_TIME", "WORD_RUN_COUNT", "WORD_AVERAGE_FIX_PUPIL_SIZE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["WORD"] = df["WORD"].astype(str)
df = df.sort_values(["PP_NR", "PART", "TRIAL", "WORD_ID_WITHIN_TRIAL"]).reset_index(drop=True)
read = df["WORD_SKIP"] == 0

df["rt_log"] = np.log1p(df["WORD_TOTAL_READING_TIME"])
g = df.loc[read].groupby("PP_NR")["rt_log"]
df["rt_z"] = (df["rt_log"] - df["PP_NR"].map(g.mean())) / df["PP_NR"].map(g.std())

readers = sorted(df["PP_NR"].unique())
test_readers = readers[::4][:5]
is_test = df["PP_NR"].isin(test_readers)

crowd = df.loc[read & ~is_test].groupby("WORD_ID")["rt_z"].mean().rename("crowd_rt_z")
df = df.join(crowd, on="WORD_ID")
df["crowd_rt_z"] = df["crowd_rt_z"].fillna(0)
df["residual"] = df["rt_z"] - df["crowd_rt_z"]
df["reread"] = (df["WORD_RUN_COUNT"] >= 2).astype(float)

# is word-level struggle even self-consistent? if it were a real state it would
# persist from one word to the next.
r = df.loc[read].copy()
r["resid_prev"] = r.groupby(["PP_NR", "TRIAL"])["residual"].shift(1)
print(f"word-to-word correlation of personal struggle : {r['residual'].corr(r['resid_prev']):+.3f}")

clean = df["WORD"].str.lower().str.strip('.,;:!?\"()—’\'')
df["logfreq"] = np.log1p(clean.map(clean.value_counts())).values
df["word_len"] = df["WORD"].str.len()

# ------------------------------------------------ zoom out to paragraph level
t = (df[read].groupby(["PP_NR", "PART", "TRIAL"])
     .agg(residual=("residual", "mean"), rt_z=("rt_z", "mean"),
          crowd_rt_z=("crowd_rt_z", "mean"), reread=("reread", "mean"),
          word_len=("word_len", "mean"), word_len_max=("word_len", "max"),
          logfreq=("logfreq", "mean"), logfreq_min=("logfreq", "min"),
          n_words=("WORD", "size"), pupil=("WORD_AVERAGE_FIX_PUPIL_SIZE", "mean"))
     .reset_index().sort_values(["PP_NR", "PART", "TRIAL"]))
print(f"paragraph-to-paragraph correlation           : "
      f"{t['residual'].corr(t.groupby('PP_NR')['residual'].shift(1)):+.3f}\n")

# label: this paragraph cost this reader more than paragraphs usually cost them
t["hard_for_me"] = (t["residual"] > t.groupby("PP_NR")["residual"].transform(lambda s: s.quantile(0.80))).astype(int)

# reader state: only from paragraphs already finished
gp = t.groupby("PP_NR")
t["prev_resid"] = gp["residual"].shift(1)
t["prev_resid_5"] = gp["residual"].shift(1).rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
t["prev_rt_z_5"] = gp["rt_z"].shift(1).rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
t["prev_reread_5"] = gp["reread"].shift(1).rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
t["prev_pupil_5"] = gp["pupil"].shift(1).rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
t["trials_so_far"] = gp.cumcount()

TEXT = ["word_len", "word_len_max", "logfreq", "logfreq_min", "n_words", "crowd_rt_z"]
STATE = ["prev_resid", "prev_resid_5", "prev_rt_z_5", "prev_reread_5", "prev_pupil_5",
         "trials_so_far", "PART"]

tr, te = t[~t["PP_NR"].isin(test_readers)], t[t["PP_NR"].isin(test_readers)]
print(f"{len(tr)} train paragraphs / {len(te)} test paragraphs (unseen readers)\n")

def evaluate(feats, name):
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    m.fit(tr[feats].astype(float), tr["hard_for_me"])
    p = m.predict_proba(te[feats].astype(float))[:, 1]
    y = te["hard_for_me"].values
    k = max(1, int(0.20 * len(p)))
    print(f"   {name:34s} AUC={roc_auc_score(y, p):.3f}  AP={average_precision_score(y, p):.3f}"
          f"  precision@20%={y[np.argsort(-p)[:k]].mean():.3f}  (base {y.mean():.3f})")

print("paragraph is hard FOR THIS READER (beyond its usual cost):")
evaluate(TEXT, "text only")
evaluate(STATE, "reader state only")
evaluate(TEXT + STATE, "text + reader state")
