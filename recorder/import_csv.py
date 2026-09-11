"""
Move the old gaze CSVs into the database, so everything lives in one place.

The early recorder wrote one CSV per run (gaze_data_notebooks/). Those runs
predate calibration, and their vertical gaze is unusable -- the divide-by-zero
bug -- so they are imported as sessions with calib_error = NULL and a note
saying so. They are history, not training data: keep them, label them, and let
every analysis script skip them by the same calibration filter it already uses.

    python3 recorder/import_csv.py            # import anything not already in
    python3 recorder/import_csv.py --dry-run
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from recorder import _env  # noqa: E402,F401  -- re-launches in the mlclass env if needed
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import db as _db          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "reading.db")
CSV_DIR = os.path.join(os.path.dirname(HERE), "gaze_data_notebooks")


def parse_name(fname):
    """gaze_data_Emma B_20260506_123753.csv -> ('Emma B', datetime)"""
    stem = os.path.basename(fname)[len("gaze_data_"):].rsplit(".", 1)[0]
    parts = stem.split("_")
    when = None
    if len(parts) >= 3:
        try:
            when = datetime.strptime("_".join(parts[-2:]), "%Y%m%d_%H%M%S")
            name = "_".join(parts[:-2])
        except ValueError:
            name = stem
    else:
        name = stem
    return name.strip(), when


def import_file(conn, path, dry_run=False):
    name, when = parse_name(path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, "empty"

    started = (when or datetime.now()).replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
    already = conn.execute("""SELECT session_id FROM sessions
                              WHERE started_at = ? AND notes LIKE 'imported from %'""",
                           (started,)).fetchone()
    if already:
        return already[0], "already imported"
    if dry_run:
        return None, f"would import {len(rows)} rows as {name!r}"

    uid = _db.get_or_create_user(conn, name or "unknown")
    sid = _db.start_session(conn, uid, text_source=rows[0].get("current_text") and "web page",
                            device_label="legacy-csv")
    conn.execute("UPDATE sessions SET started_at = ?, notes = ? WHERE session_id = ?",
                 (started, f"imported from {os.path.basename(path)}; pre-calibration, "
                           "vertical gaze unusable (divide-by-zero bug)", sid))

    t0 = float(rows[0]["timestamp"])
    w = _db.SampleWriter(conn, sid)
    for r in rows:
        try:
            t_ms = (float(r["timestamp"]) - t0) * 1000.0
        except (KeyError, ValueError):
            continue
        def num(key):
            try:
                return float(r[key])
            except (KeyError, TypeError, ValueError):
                return None
        w.add(t_ms=t_ms, face_detected=int(float(r.get("face_detected") or 0)),
              norm_x=num("avg_norm_x"), norm_y=num("avg_norm_y"),
              gaze_x=num("screen_x"), gaze_y=num("screen_y"))
    w.flush()
    fps = len(rows) / max(((float(rows[-1]["timestamp"]) - t0) or 1), 1e-9)
    _db.end_session(conn, sid, camera_fps=fps)
    return sid, f"{len(rows)} rows at ~{fps:.0f} fps"


def main(dry_run=False):
    conn = _db.connect(DB)
    files = sorted(f for f in os.listdir(CSV_DIR) if f.endswith(".csv"))
    if not files:
        print("no CSVs found in", CSV_DIR); return
    for f in files:
        sid, msg = import_file(conn, os.path.join(CSV_DIR, f), dry_run)
        print(f"  {f[:52]:54s} -> {'session ' + str(sid) + ': ' if sid else ''}{msg}")
    print("\nthe CSV files are left untouched on disk -- raw data is never deleted, "
          "only copied in")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
