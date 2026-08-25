"""
The pipeline-truth test on Emma's own recordings.

Psycholinguistics (and our GECO models) say: dwell time rises with word length
and falls with word frequency. If Emma's webcam sessions show the same signs,
the whole chain -- camera -> iris -> calibration -> word attribution -> dwell --
is measuring something real. If they don't, the pipeline is noise and no model
trained on it can mean anything. Run after each new session.
"""
import sqlite3, numpy as np, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
conn = sqlite3.connect(os.path.join(HERE, "..", "recorder", "reading.db"))

for sid, in conn.execute("SELECT session_id FROM sessions ORDER BY session_id"):
    fps = conn.execute("SELECT camera_fps FROM sessions WHERE session_id=?", (sid,)).fetchone()[0] or 30
    # dwell per word = frames attributed to it / fps
    rows = conn.execute("""
        SELECT w.word, COUNT(*) * 1000.0 / ? AS dwell_ms
        FROM samples s JOIN words w ON w.session_id = s.session_id AND w.word_index = s.word_index
        WHERE s.session_id = ? GROUP BY s.word_index HAVING COUNT(*) >= 2""", (fps, sid)).fetchall()
    if len(rows) < 30:
        print(f"session {sid}: only {len(rows)} words with dwell -- skipping")
        continue
    words = [r[0] for r in rows]
    dwell = np.log1p([r[1] for r in rows])          # log: dwell is skewed, as in GECO
    wlen = np.array([len(w) for w in words], float)
    # frequency proxy: how common the word is within this session's page
    from collections import Counter
    freq = Counter(w.lower().strip('.,;:!?"()') for w in words)
    wfreq = np.log1p([freq[w.lower().strip('.,;:!?"()')] for w in words])

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    calib = conn.execute("SELECT calib_error FROM sessions WHERE session_id=?", (sid,)).fetchone()[0]
    r_len, r_freq = corr(wlen, dwell), corr(wfreq, dwell)
    exp = "expected: len +, freq -"
    verdict = "SIGNS MATCH" if (r_len > 0 and r_freq < 0) else "signs off"
    print(f"session {sid} (calib {calib:.0f}px, {len(rows)} words): "
          f"dwell~len r={r_len:+.2f}  dwell~freq r={r_freq:+.2f}   {verdict}  ({exp})")
