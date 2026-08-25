"""
Check whether the 'breakdown' label can be predicted from anything OTHER than
the two columns it was built from.

Run:  python3 diagnostics/label_check.py
First run takes ~4 min (reads the 164 MB xlsx), then caches to a .pkl.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "..", "src", "L2ReadingData.xlsx")
CACHE = os.path.join(HERE, "geco_cache.pkl")

KEEP = ["PP_NR", "TRIAL", "WORD", "WORD_GAZE_DURATION", "WORD_AVERAGE_FIX_PUPIL_SIZE",
        "WORD_SKIP", "WORD_TOTAL_READING_TIME", "WORD_FIXATION_%"]

if os.path.exists(CACHE):
    df = pd.read_pickle(CACHE)
else:
    print("reading xlsx (slow, one time)...")
    df = pd.read_excel(XLSX)[KEEP]
    df.to_pickle(CACHE)

# ---------- same preprocessing as the notebook ----------
for c in ["WORD_GAZE_DURATION", "WORD_AVERAGE_FIX_PUPIL_SIZE", "WORD_SKIP",
          "WORD_TOTAL_READING_TIME", "WORD_FIXATION_%"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False), errors="coerce")

df["WORD"] = df["WORD"].astype(str)
df["word_len"] = df["WORD"].str.len()
df["is_punct"] = df["WORD"].str.fullmatch(r"\W+").fillna(False).astype(int)
df["is_number"] = df["WORD"].str.fullmatch(r"\d+").fillna(False).astype(int)

agg = {"WORD_GAZE_DURATION": ["mean", "std", "max"],
       "WORD_AVERAGE_FIX_PUPIL_SIZE": ["mean", "std"],
       "WORD_TOTAL_READING_TIME": ["sum", "mean", "max"],
       "WORD_SKIP": ["mean"], "WORD_FIXATION_%": ["mean"],
       "word_len": ["mean", "std", "max"], "is_punct": ["mean"], "is_number": ["mean"],
       "WORD": ["count"]}
trial = df.groupby(["PP_NR", "TRIAL"]).agg(agg).reset_index()
trial.columns = [f"{a}_{b}" if b else a for a, b in trial.columns.to_flat_index()]
trial.rename(columns={"WORD_count": "n_words_in_trial"}, inplace=True)


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) if (sd == 0 or np.isnan(sd)) else (s - s.mean()) / sd


for c in ["WORD_TOTAL_READING_TIME_sum", "WORD_GAZE_DURATION_mean",
          "WORD_GAZE_DURATION_max", "WORD_AVERAGE_FIX_PUPIL_SIZE_mean"]:
    trial[c + "_z"] = trial.groupby("PP_NR")[c].transform(zscore)

trial["breakdown"] = ((trial["WORD_TOTAL_READING_TIME_sum_z"] > 2.0) |
                      (trial["WORD_GAZE_DURATION_max_z"] > 2.0)).astype(int)

# ---------- proof that the label is a rule, not a discovery ----------
rule = ((trial["WORD_TOTAL_READING_TIME_sum_z"] > 2.0) |
        (trial["WORD_GAZE_DURATION_max_z"] > 2.0)).astype(int)
print(f"a 1-line if-statement using 2 of the input columns reproduces the label "
      f"{(rule == trial['breakdown']).mean():.1%} of the time\n")

y = trial["breakdown"].values
groups = trial["PP_NR"].values
X = trial.drop(columns=["breakdown", "PP_NR", "TRIAL"]).select_dtypes(include=[np.number])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))


def run(cols, name):
    """Train on one subset of columns, report AUC on held-out readers."""
    Xtr, Xte = X.iloc[tr][cols].fillna(0).values, X.iloc[te][cols].fillna(0).values
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), y[tr])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0).fit(sc.transform(Xtr), y[tr])
    print(f"{name:48s} LR={roc_auc_score(y[te], lr.predict_proba(sc.transform(Xte))[:, 1]):.3f}"
          f"  MLP={roc_auc_score(y[te], mlp.predict_proba(sc.transform(Xte))[:, 1]):.3f}")


cols = list(X.columns)
text_only = [c for c in cols if c.startswith(("word_len", "is_punct", "is_number", "n_words"))]
label_parents = ["WORD_TOTAL_READING_TIME_sum_z", "WORD_GAZE_DURATION_max_z"]

run(cols, "everything (the notebook's feature set)")
run(label_parents, "ONLY the 2 columns the label is made of")
run(text_only, "text features only (no eye data)")
