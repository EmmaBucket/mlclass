"""
Storage for reading sessions.

Design rules this file follows:
  1. RAW IS SACRED. Samples go in exactly as measured. Anything computed
     (z-scores, struggle labels, "was this hard") is derived later, in analysis,
     where you can fix a bug and recompute. You cannot re-record a session.
  2. One row per camera frame, batched. Committing per frame would stall the
     capture loop and you would silently drop frames.
  3. Every session stores the conditions it was recorded under -- screen size,
     calibration, self-reported tiredness. Your whole "vs themselves" idea needs
     to know which day it was.
"""
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL UNIQUE,          -- what you click on
    created_at   TEXT NOT NULL,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(user_id),
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    text_source  TEXT,                          -- url or title of what they read
    screen_w     INTEGER,
    screen_h     INTEGER,
    screen_w_mm  REAL,                          -- physical size: 15" laptop vs 27" monitor
    screen_h_mm  REAL,                          --   (same pixels can be very different text sizes)
    device_label TEXT,                          -- 'macbook-air-13' / 'office-monitor' / later: 'phone'
    glasses      INTEGER,                       -- 1/0: lenses reflect and refract; the model should know
    screenshot_path TEXT,                       -- one full-page capture of what was read
    camera_fps   REAL,
    calibration  TEXT,                          -- JSON: the fitted gaze mapping
    calib_error  REAL,                          -- mean px error at calibration points
    self_report  TEXT,                          -- "tired" / "rested" / free text
    notes        TEXT
);

-- one row per camera frame
CREATE TABLE IF NOT EXISTS samples (
    session_id    INTEGER NOT NULL REFERENCES sessions(session_id),
    t_ms          REAL NOT NULL,                -- ms since session start (monotonic)
    face_detected INTEGER NOT NULL,
    norm_x        REAL,                         -- raw iris position in the eye, 0..1
    norm_y        REAL,
    gaze_x        REAL,                         -- calibrated screen pixels
    gaze_y        REAL,
    eye_span      REAL,                         -- lean-in proxy (px between eye corners)
    frown         REAL,                         -- brow-to-eye distance / eye_span
    ear           REAL,                         -- eye aspect ratio (low = blink)
    scroll_y      REAL,
    word_index    INTEGER                       -- which word was under the gaze
);
CREATE INDEX IF NOT EXISTS idx_samples ON samples(session_id, t_ms);

-- where each word sat on screen during THIS session (layout changes per window size)
CREATE TABLE IF NOT EXISTS words (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    word_index INTEGER NOT NULL,
    word       TEXT NOT NULL,
    left       REAL, top REAL, right REAL, bottom REAL,
    PRIMARY KEY (session_id, word_index)
);

-- things that happen during a session (reading-mode switches, future: layout
-- experiments). value at time t, queryable next to the samples timeline.
CREATE TABLE IF NOT EXISTS events (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    t_ms       REAL NOT NULL,
    kind       TEXT NOT NULL,
    value      TEXT
);

-- GROUND TRUTH. Without this you are predicting eye movement, not comprehension.
CREATE TABLE IF NOT EXISTS checks (
    check_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(session_id),
    asked_at    TEXT NOT NULL,
    word_start  INTEGER,                        -- which passage it refers to
    word_end    INTEGER,
    kind        TEXT,                           -- 'self_report' | 'question'
    prompt      TEXT,
    answer      TEXT,
    correct     INTEGER                         -- 1/0, NULL for self-reports
);

-- written once at session end, so progress queries stay fast
CREATE TABLE IF NOT EXISTS session_summary (
    session_id     INTEGER PRIMARY KEY REFERENCES sessions(session_id),
    n_samples      INTEGER,
    duration_s     REAL,
    face_rate      REAL,                        -- fraction of frames with a face
    mean_eye_span  REAL,
    mean_frown     REAL,
    words_visited  INTEGER,
    median_dwell_ms REAL
);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
"""


def connect(path="recorder/reading.db"):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")     # capture loop never blocks on a reader
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
        conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    # additive migration: bring v1 databases up to the current column set
    have = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
    for col, typ in [("screen_w_mm", "REAL"), ("screen_h_mm", "REAL"),
                     ("device_label", "TEXT"), ("screenshot_path", "TEXT"),
                     ("glasses", "INTEGER")]:
        if col not in have:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typ}")
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    conn.commit()
    return conn


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------- users
def list_users(conn):
    """[(user_id, display_name, n_sessions), ...] -- what the picker shows."""
    return conn.execute("""
        SELECT u.user_id, u.display_name, COUNT(s.session_id)
        FROM users u LEFT JOIN sessions s ON s.user_id = u.user_id
        GROUP BY u.user_id ORDER BY u.display_name
    """).fetchall()


def get_or_create_user(conn, display_name):
    """Click 'Emma' -> always the same user_id, even after a rename typo elsewhere."""
    row = conn.execute("SELECT user_id FROM users WHERE display_name = ?",
                       (display_name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO users (display_name, created_at) VALUES (?, ?)",
                       (display_name, now()))
    conn.commit()
    return cur.lastrowid


# -------------------------------------------------------------- sessions
def start_session(conn, user_id, text_source=None, screen_w=None, screen_h=None,
                  self_report=None, screen_w_mm=None, screen_h_mm=None, device_label=None,
                  glasses=None):
    cur = conn.execute("""
        INSERT INTO sessions (user_id, started_at, text_source, screen_w, screen_h,
                              self_report, screen_w_mm, screen_h_mm, device_label, glasses)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, now(), text_source, screen_w, screen_h, self_report,
         screen_w_mm, screen_h_mm, device_label, glasses))
    conn.commit()
    return cur.lastrowid


def save_calibration(conn, session_id, mapping, error_px):
    conn.execute("UPDATE sessions SET calibration = ?, calib_error = ? WHERE session_id = ?",
                 (json.dumps(mapping), error_px, session_id))
    conn.commit()


def add_event(conn, session_id, t_ms, kind, value):
    conn.execute("INSERT INTO events VALUES (?,?,?,?)", (session_id, t_ms, kind, value))
    conn.commit()


def known_devices(conn, user_id):
    """Device labels this user has recorded on before, most recent first."""
    return [r[0] for r in conn.execute("""
        SELECT device_label FROM sessions
        WHERE user_id = ? AND device_label IS NOT NULL AND device_label != ''
        GROUP BY device_label ORDER BY MAX(started_at) DESC""", (user_id,))]


def set_screenshot(conn, session_id, path):
    conn.execute("UPDATE sessions SET screenshot_path = ? WHERE session_id = ?",
                 (path, session_id))
    conn.commit()


def save_words(conn, session_id, word_map):
    conn.executemany(
        "INSERT OR REPLACE INTO words VALUES (?,?,?,?,?,?,?)",
        [(session_id, i, w["text"], w["left"], w["top"], w["right"], w["bottom"])
         for i, w in enumerate(word_map)])
    conn.commit()


def add_check(conn, session_id, kind, prompt, answer, correct=None,
              word_start=None, word_end=None):
    conn.execute("""INSERT INTO checks
        (session_id, asked_at, word_start, word_end, kind, prompt, answer, correct)
        VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, now(), word_start, word_end, kind, prompt, answer, correct))
    conn.commit()


class SampleWriter:
    """Buffers frames and writes them in batches.

    Why: one INSERT+commit per frame is a disk sync per frame, ~5-10 ms, at 30 fps.
    That is how a capture loop starts dropping frames without ever telling you.
    """

    COLS = ("session_id, t_ms, face_detected, norm_x, norm_y, gaze_x, gaze_y, "
            "eye_span, frown, ear, scroll_y, word_index")

    def __init__(self, conn, session_id, batch=200):
        self.conn, self.session_id, self.batch, self.buf = conn, session_id, batch, []
        self.n = 0

    def add(self, t_ms, face_detected, norm_x=None, norm_y=None, gaze_x=None,
            gaze_y=None, eye_span=None, frown=None, ear=None, scroll_y=None,
            word_index=None):
        self.buf.append((self.session_id, t_ms, int(face_detected), norm_x, norm_y,
                         gaze_x, gaze_y, eye_span, frown, ear, scroll_y, word_index))
        self.n += 1
        if len(self.buf) >= self.batch:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.conn.executemany(
            f"INSERT INTO samples ({self.COLS}) VALUES ({','.join('?' * 12)})", self.buf)
        self.conn.commit()
        self.buf.clear()


def end_session(conn, session_id, camera_fps=None):
    """Close the session and precompute the numbers you will look at often."""
    SampleWriter(conn, session_id).flush()
    s = conn.execute("""
        SELECT COUNT(*), MAX(t_ms)/1000.0, AVG(face_detected), AVG(eye_span), AVG(frown),
               COUNT(DISTINCT word_index)
        FROM samples WHERE session_id = ?""", (session_id,)).fetchone()
    dwell = conn.execute("""
        SELECT COUNT(*) * 1000.0 / NULLIF(:fps, 0)
        FROM samples WHERE session_id = :sid AND word_index IS NOT NULL
        GROUP BY word_index ORDER BY 1""",
        {"sid": session_id, "fps": camera_fps or 30}).fetchall()
    median = dwell[len(dwell) // 2][0] if dwell else None
    conn.execute("""INSERT OR REPLACE INTO session_summary VALUES (?,?,?,?,?,?,?,?)""",
                 (session_id, s[0], s[1], s[2], s[3], s[4], s[5], median))
    conn.execute("UPDATE sessions SET ended_at = ?, camera_fps = ? WHERE session_id = ?",
                 (now(), camera_fps, session_id))
    conn.commit()


def user_progress(conn, user_id):
    """One row per session -- this is the 'track your own progress' view."""
    return conn.execute("""
        SELECT s.session_id, s.started_at, s.self_report, s.text_source,
               sm.duration_s, sm.face_rate, sm.words_visited, sm.median_dwell_ms,
               sm.mean_eye_span, sm.mean_frown
        FROM sessions s JOIN session_summary sm USING (session_id)
        WHERE s.user_id = ? ORDER BY s.started_at""", (user_id,)).fetchall()
