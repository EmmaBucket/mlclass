"""
An honest baseline: predict, from the TEXT ALONE, which words a reader will
dwell on. No eye-tracking columns are used as inputs -- only things you know
before anyone reads the page, which is exactly the situation your adaptive
layout will be in.

Target: this word's total reading time is in the top 10% for THIS reader.
Split: by participant, so the test readers were never seen in training.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(HERE, "geco_cache.pkl"))  # created by label_check.py
df["WORD"] = df["WORD"].astype(str)
df["WORD_TOTAL_READING_TIME"] = pd.to_numeric(df["WORD_TOTAL_READING_TIME"], errors="coerce").fillna(0)

# ---------- target: long dwell relative to this reader's own habits ----------
per_reader_cutoff = df.groupby("PP_NR")["WORD_TOTAL_READING_TIME"].transform(lambda s: s.quantile(0.90))
df["hard"] = (df["WORD_TOTAL_READING_TIME"] > per_reader_cutoff).astype(int)

# ---------- features: text only ----------
clean = df["WORD"].str.lower().str.strip('.,;:!?"()—’\'')
corpus_freq = clean.value_counts()

df["pos"] = df.groupby(["PP_NR", "TRIAL"]).cumcount()      # how far into the paragraph
df["logfreq"] = np.log1p(clean.map(corpus_freq)).values     # common word vs rare word
df["word_len"] = df["WORD"].str.len()
df["is_punct"] = df["WORD"].str.fullmatch(r"\W+").fillna(False).astype(int)
df["is_number"] = df["WORD"].str.fullmatch(r"\d+").fillna(False).astype(int)
df["ends_sentence"] = df["WORD"].str.contains(r"[.!?]$", regex=True).astype(int)
df["has_comma"] = df["WORD"].str.contains(",").astype(int)
df["n_vowels"] = df["WORD"].str.lower().str.count(r"[aeiouy]")
df["prev_len"] = df.groupby(["PP_NR", "TRIAL"])["word_len"].shift(1).fillna(0)
df["next_len"] = df.groupby(["PP_NR", "TRIAL"])["word_len"].shift(-1).fillna(0)
df["prev_logfreq"] = df.groupby(["PP_NR", "TRIAL"])["logfreq"].shift(1).fillna(0)
df["trial_len"] = df.groupby(["PP_NR", "TRIAL"])["word_len"].transform("size")

feats = ["word_len", "logfreq", "is_punct", "is_number", "ends_sentence", "has_comma",
         "n_vowels", "prev_len", "next_len", "prev_logfreq", "pos", "trial_len"]

X = df[feats].fillna(0).values
y = df["hard"].values
groups = df["PP_NR"].values
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups))
print(f"{len(df):,} words | {y.mean():.1%} labelled hard | "
      f"{len(set(groups[tr]))} train readers, {len(set(groups[te]))} unseen test readers\n")

sc = StandardScaler().fit(X[tr])
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
p_lr = lr.predict_proba(sc.transform(X[te]))[:, 1]
gb = HistGradientBoostingClassifier(max_iter=300).fit(X[tr], y[tr])
p_gb = gb.predict_proba(X[te])[:, 1]

print(f"logistic regression : AUC={roc_auc_score(y[te], p_lr):.3f}  AP={average_precision_score(y[te], p_lr):.3f}")
print(f"gradient boosting   : AUC={roc_auc_score(y[te], p_gb):.3f}  AP={average_precision_score(y[te], p_gb):.3f}")
print(f"guessing            : AUC=0.500  AP={y[te].mean():.3f}\n")

print("what the model learned (positive = predicts a long dwell):")
for f, c in sorted(zip(feats, lr.coef_[0]), key=lambda kv: -abs(kv[1])):
    print(f"   {f:14s} {c:+.3f}")
