"""
Write a session's notes out as a file you can keep, so the reading survives the
app being closed. Markdown, named after what was read.

    python3 recorder/export_notes.py            # newest session
    python3 recorder/export_notes.py 12         # a specific session
    python3 recorder/export_notes.py --user 1   # everything one reader marked
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from recorder import _env  # noqa: E402,F401  -- re-launches in the mlclass env if needed
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import db as _db          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "reading.db")
OUT_DIR = os.path.join(HERE, "notes")

TAG_ORDER = ["question", "definition", "important", "todo", None]
TAG_TITLE = {"question": "Questions", "definition": "Definitions",
             "important": "Important", "todo": "To do", None: "Unsorted"}


def _slug(text, fallback="reading"):
    name = re.sub(r"^https?://", "", (text or "")).strip()
    name = re.sub(r"[^\w\s.-]+", " ", name).strip()
    name = re.sub(r"\s+", "-", name)[:60]
    return name or fallback


def export_session(conn, session_id, out_dir=OUT_DIR):
    row = conn.execute("""SELECT s.text_source, s.started_at, s.self_report, u.display_name
                          FROM sessions s JOIN users u USING(user_id)
                          WHERE s.session_id = ?""", (session_id,)).fetchone()
    if not row:
        return None
    source, started, feeling, reader = row
    notes = conn.execute("""SELECT quote, note, tag, word_index FROM notes
                            WHERE session_id = ? ORDER BY word_index""", (session_id,)).fetchall()
    if not notes:
        return None

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_slug(source)}_{started[:10]}_s{session_id}.md")
    by_tag = {}
    for quote, note, tag, _ in notes:
        by_tag.setdefault(tag, []).append((quote, note))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Notes: {source or 'reading'}\n\n")
        fh.write(f"*{reader} · session {session_id} · {started[:16].replace('T', ' ')}"
                 f"{' · felt ' + feeling if feeling else ''} · "
                 f"{len(notes)} marked passage(s)*\n\n")
        for tag in TAG_ORDER:
            if tag not in by_tag:
                continue
            fh.write(f"## {TAG_TITLE[tag]}\n\n")
            for quote, note in by_tag[tag]:
                fh.write(f"> {quote}\n\n")
                if (note or "").strip():
                    fh.write(f"{note.strip()}\n\n")
    return path


def export_user(conn, user_id, out_dir=OUT_DIR):
    made = []
    for (sid,) in conn.execute("""SELECT session_id FROM sessions WHERE user_id = ?
                                  ORDER BY session_id""", (user_id,)):
        p = export_session(conn, sid, out_dir)
        if p:
            made.append(p)
    return made


if __name__ == "__main__":
    conn = _db.connect(DB)
    if "--user" in sys.argv:
        uid = int(sys.argv[sys.argv.index("--user") + 1])
        for p in export_user(conn, uid):
            print("wrote", p)
    else:
        sid = int(sys.argv[1]) if len(sys.argv) > 1 else conn.execute(
            "SELECT MAX(session_id) FROM sessions").fetchone()[0]
        p = export_session(conn, sid)
        print("wrote " + p if p else f"session {sid} has no notes")
