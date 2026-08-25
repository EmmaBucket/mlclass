"""
Your target definition, made computable.

You said struggle = going back over text, skipping a lot, dwelling in one spot,
leaning in, frowning -- judged both against other readers and against the reader's
own normal. GECO can measure three of those. This script builds them as labels and
asks which ones are predictable, from what.

THE RULE THIS SCRIPT OBEYS:
    eye data about THIS word  -> label   (never a feature)
    eye data from EARLIER words -> feature (allowed: it already happened)
    text -> feature (allowed: you know it before anyone reads)
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(HERE, "geco_cache2.pkl"))

# ---------------------------------------------------------------- 1. clean up
# GECO writes "." where a word was never fixated. to_numeric turns those into NaN.
for c in ["WORD_TOTAL_READING_TIME", "WORD_GAZE_DURATION", "WORD_RUN_COUNT",
          "WORD_FIXATION_COUNT", "WORD_GO_PAST_TIME", "WORD_AVERAGE_FIX_PUPIL_SIZE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["WORD"] = df["WORD"].astype(str)
df = df.sort_values(["PP_NR", "PART", "TRIAL", "WORD_ID_WITHIN_TRIAL"]).reset_index(drop=True)
read = df["WORD_SKIP"] == 0          # the reader's eyes actually landed here

# ------------------------------------------------- 2. "compared to themselves"
# Reading times are skewed (a few huge ones), so log first, THEN z-score.
# Otherwise one 4000ms stare drags the reader's mean and hides everything else.
df["rt_log"] = np.log1p(df["WORD_TOTAL_READING_TIME"])
g = df.loc[read].groupby("PP_NR")["rt_log"]
mu, sd = df["PP_NR"].map(g.mean()), df["PP_NR"].map(g.std())
df["rt_z"] = (df["rt_log"] - mu) / sd          # >0 = slower than this reader's normal

# ---------------------------------------------------- 3. split BEFORE anything
# Held-out readers must not influence the crowd baseline, so the split comes first.
readers = sorted(df["PP_NR"].unique())
test_readers = readers[::4][:5]                # 5 of 19, spread out
is_test = df["PP_NR"].isin(test_readers)
print(f"{len(readers)} readers | test readers held out: {test_readers}\n")

# --------------------------------------------- 4. "compared to other readers"
# What does this exact word usually cost? Averaged over TRAINING readers only.
train_read = read & ~is_test
crowd = df.loc[train_read].groupby("WORD_ID").agg(
    crowd_rt_z=("rt_z", "mean"),
    crowd_reread=("WORD_RUN_COUNT", lambda s: (s >= 2).mean()),
    crowd_skip=("WORD_SKIP", "mean"),
)
df = df.join(crowd, on="WORD_ID")
for c in ["crowd_rt_z", "crowd_reread", "crowd_skip"]:
    df[c] = df[c].fillna(df[c].mean())

# -------------------------------------------------------------- 5. the labels
# A: hard for readers in general (a property of the text)
df["lab_text_hard"] = (df["crowd_rt_z"] > df.loc[train_read, "crowd_rt_z"].quantile(0.90)).astype(int)
# B: hard for THIS reader beyond what the word normally costs (a property of the moment)
df["residual"] = df["rt_z"] - df["crowd_rt_z"]
# .transform on a filtered frame keeps only the filtered rows' index, so reindex
# back onto the full frame before comparing. (rows that were skipped become NaN -> False)
cut = (df.loc[read].groupby("PP_NR")["residual"]
         .transform(lambda s: s.quantile(0.90)).reindex(df.index))
df["lab_personal"] = ((df["residual"] > cut) & read).astype(int)
# C: your first signal, literally -- the reader went back over this word
df["lab_reread"] = ((df["WORD_RUN_COUNT"] >= 2) & read).astype(int)

# ------------------------------------------------------------- 6. text features
clean = df["WORD"].str.lower().str.strip('.,;:!?"()—’\'')
freq = clean.value_counts()
df["logfreq"] = np.log1p(clean.map(freq)).values
df["word_len"] = df["WORD"].str.len()
df["is_punct"] = df["WORD"].str.fullmatch(r"\W+").fillna(False).astype(int)
df["is_number"] = df["WORD"].str.fullmatch(r"\d+").fillna(False).astype(int)
df["ends_sentence"] = df["WORD"].str.contains(r"[.!?]$").astype(int)
df["has_comma"] = df["WORD"].str.contains(",").astype(int)
df["pos"] = df["WORD_ID_WITHIN_TRIAL"]
df["trial_len"] = df.groupby(["PP_NR", "TRIAL"])["WORD"].transform("size")
by_trial = df.groupby(["PP_NR", "TRIAL"])
df["prev_len"] = by_trial["word_len"].shift(1).fillna(0)
df["next_len"] = by_trial["word_len"].shift(-1).fillna(0)
df["prev_logfreq"] = by_trial["logfreq"].shift(1).fillna(0)

TEXT = ["word_len", "logfreq", "is_punct", "is_number", "ends_sentence", "has_comma",
        "pos", "trial_len", "prev_len", "next_len", "prev_logfreq"]
CROWD = ["crowd_rt_z", "crowd_reread", "crowd_skip"]

# ------------------------------------------- 7. reader-state ("tired today") features
# shift(1) BEFORE rolling. Without it the window includes the current word and the
# model reads the answer out of its own input. This one line is the whole ballgame.
def past_mean(col, window):
    return (df.groupby("PP_NR")[col].shift(1)
              .rolling(window, min_periods=10).mean()
              .reset_index(level=0, drop=True))

df["reread_flag"] = (df["WORD_RUN_COUNT"] >= 2).astype(float)
df["recent_rt_z_50"] = past_mean("rt_z", 50)          # slower than usual lately?
df["recent_rt_z_500"] = past_mean("rt_z", 500)        # slow all session? (tired)
df["recent_reread_50"] = past_mean("reread_flag", 50)
df["recent_skip_50"] = past_mean("WORD_SKIP", 50)
df["recent_pupil_500"] = past_mean("WORD_AVERAGE_FIX_PUPIL_SIZE", 500)
df["words_so_far"] = df.groupby("PP_NR").cumcount()
df["part"] = df["PART"]
STATE = ["recent_rt_z_50", "recent_rt_z_500", "recent_reread_50", "recent_skip_50",
         "recent_pupil_500", "words_so_far", "part"]

# ------------------------------------------------------------- 8. evaluate
rows = df[read].copy()
tr, te = rows[~rows["PP_NR"].isin(test_readers)], rows[rows["PP_NR"].isin(test_readers)]

def evaluate(label, feats, name):
    m = HistGradientBoostingClassifier(max_iter=150, random_state=0)
    m.fit(tr[feats].astype(float), tr[label])
    p = m.predict_proba(te[feats].astype(float))[:, 1]
    y = te[label].values
    k = max(1, int(0.10 * len(p)))                       # you can only restyle so many words
    prec_at_k = y[np.argsort(-p)[:k]].mean()
    print(f"   {name:28s} AUC={roc_auc_score(y, p):.3f}  AP={average_precision_score(y, p):.3f}"
          f"  precision@10%={prec_at_k:.3f}  (base rate {y.mean():.3f})")

for label, blurb in [("lab_text_hard", "A. hard for readers in general (text property)"),
                     ("lab_personal",  "B. hard for THIS reader, beyond the word's usual cost"),
                     ("lab_reread",    "C. reader went back over this word")]:
    print(blurb)
    evaluate(label, TEXT, "text only")
    if label != "lab_text_hard":                          # crowd IS label A, can't be its input
        evaluate(label, TEXT + CROWD, "text + crowd baseline")
    evaluate(label, TEXT + CROWD + STATE if label != "lab_text_hard" else TEXT + STATE,
             "+ reader state (past only)")
    print()
