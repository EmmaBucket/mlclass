"""
Teach the page which words are hard FOR THIS READER.

Until now every reader got the same GECO population model (word length up,
frequency down, fitted on 19 strangers). This file re-fits that model on the
reader's OWN recorded sessions and blends the two.

The blend is the important part, and it is a real statistical idea called
shrinkage: with 200 of your own words the population model should still
dominate, with 20,000 yours should. So

    weight_you = n_your_words / (n_your_words + PRIOR_STRENGTH)
    coef = weight_you * your_coef + (1 - weight_you) * geco_coef

With no data you get exactly the population model; with lots you get almost
entirely yourself; in between you slide across smoothly and never fall off a
cliff. (PRIOR_STRENGTH = 3000 words ~= two good sessions to reach 50/50.)

Only sessions with trustworthy calibration are used -- a 300 px session's
word attribution is noise, and training on noise would make the page worse.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import db as _db          # noqa: E402
from recorder import adapt              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "reading.db")
GOOD_CALIB = 150          # px
MIN_WORDS = 150           # below this, don't bother fitting
PRIOR_STRENGTH = 3000.0   # how many of your words it takes to half-outweigh GECO


def collect(conn, user_id):
    """One row per (session, word) the reader actually looked at: text features
    plus how long they dwelled. Only from well-calibrated sessions."""
    rows = conn.execute("""
        SELECT w.word,
               COUNT(*) * 1000.0 / COALESCE(se.camera_fps, 30) AS dwell_ms,
               se.session_id
        FROM samples sa
        JOIN sessions se ON se.session_id = sa.session_id
        JOIN words w ON w.session_id = sa.session_id AND w.word_index = sa.word_index
        WHERE se.user_id = ? AND se.calib_error IS NOT NULL AND se.calib_error < ?
              AND w.right - w.left > 0                       -- skip legacy junk rows
        GROUP BY sa.session_id, sa.word_index
        HAVING COUNT(*) >= 2""", (user_id, GOOD_CALIB)).fetchall()
    return rows


def fit(user_id=1, db_path=DB, verbose=True):
    conn = _db.connect(db_path)
    rows = collect(conn, user_id)
    if verbose:
        print(f"{len(rows)} word-observations from well-calibrated sessions")
    if len(rows) < MIN_WORDS:
        if verbose:
            print(f"need at least {MIN_WORDS} -- keeping the population model for now")
        return None

    words = [r[0] for r in rows]
    dwell = np.array([r[1] for r in rows], float)
    X = np.array([[len(w), adapt.word_logfreq(w)] for w in words], float)
    # "hard for me" = top 20% of this reader's own dwell times
    y = (dwell > np.quantile(dwell, 0.80)).astype(float)

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    # plain logistic regression by gradient descent -- small, transparent, and
    # it keeps the file dependency-free beyond numpy
    w = np.zeros(2); b = 0.0
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= 0.05 * (Z.T @ g / len(y) + 0.01 * w)       # small L2 keeps it sane
        b -= 0.05 * g.mean()

    # translate back to raw-feature space so it can be blended with GECO's,
    # which live in GECO's own standardisation
    own = {"word_len": float(w[0] / sd[0] * adapt.SCALE["word_len"]),
           "logfreq": float(w[1] / sd[1] * adapt.SCALE["logfreq"])}
    n = len(rows)
    wt = n / (n + PRIOR_STRENGTH)
    blended = {k: wt * own[k] + (1 - wt) * adapt.COEF[k] for k in own}

    model = {"coef": blended, "intercept": adapt.INTERCEPT, "n_words": n,
             "weight_you": round(wt, 3), "own_coef": own,
             "sessions": len({r[2] for r in rows})}
    _db.set_setting(conn, f"model_user_{user_id}", json.dumps(model))
    if verbose:
        print(f"your own fit      : word_len {own['word_len']:+.3f}  logfreq {own['logfreq']:+.3f}")
        print(f"population (GECO) : word_len {adapt.COEF['word_len']:+.3f}  "
              f"logfreq {adapt.COEF['logfreq']:+.3f}")
        print(f"blend ({wt:.0%} you) : word_len {blended['word_len']:+.3f}  "
              f"logfreq {blended['logfreq']:+.3f}")
    return model


def load(conn, user_id):
    raw = _db.get_setting(conn, f"model_user_{user_id}")
    return json.loads(raw) if raw else None


if __name__ == "__main__":
    fit(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
