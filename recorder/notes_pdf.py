"""
Turn a session's highlights into a study sheet you can print, keep, or revise from.

Two things are kept visually distinct on every page, because they are different
kinds of knowledge:
    what the TEXT said   -> the highlighted quote, in the text's own voice
    what YOU said        -> your note, in your voice, clearly yours

Rendered as HTML, then printed to PDF by headless Chrome (already installed for
the recorder), so there is no new dependency and the result looks the same as
what you saw on screen.

    python3 recorder/notes_pdf.py            # newest session with notes
    python3 recorder/notes_pdf.py 24
    python3 recorder/notes_pdf.py --user 1   # one study sheet per session
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from recorder import _env  # noqa: E402,F401  -- re-launches in the mlclass env if needed
import base64
import html as H
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import db as _db          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "reading.db")
OUT_DIR = os.path.join(HERE, "notes")

TAG_ORDER = ["important", "definition", "question", "todo", None]
TAG_LABEL = {"important": "Key points", "definition": "Definitions",
             "question": "Questions to resolve", "todo": "To do", None: "Other highlights"}
TAG_COLOR = {"important": "#d9a520", "definition": "#7b76d6",
             "question": "#e06f8b", "todo": "#5aa75a", None: "#9a9a9a"}

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm; }
* { box-sizing: border-box; }
body { font: 11.5pt/1.6 -apple-system, "Helvetica Neue", sans-serif; color:#1a1a1a; margin:0; }
h1 { font-size: 20pt; margin:0 0 2mm; }
.meta { color:#666; font-size:9.5pt; margin-bottom:8mm; }
.meta b { color:#333; }
h2 { font-size:13pt; margin:9mm 0 3mm; padding-bottom:1.5mm; border-bottom:1.5px solid #eee; }
h2 .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.item { margin:0 0 6mm; page-break-inside: avoid; }
.quote { border-left:3px solid #d8d2c8; background:#faf8f4; padding:3mm 4mm;
         font-size:11pt; color:#333; }
.quote:before { content:"from the text"; display:block; font-size:7.5pt; letter-spacing:.08em;
                text-transform:uppercase; color:#a09884; margin-bottom:1.5mm; }
.mine { margin:2mm 0 0 6mm; padding:2.5mm 4mm; background:#eef4fb; border-left:3px solid #6ea8dc;
        border-radius:0 4px 4px 0; }
.mine:before { content:"my note"; display:block; font-size:7.5pt; letter-spacing:.08em;
               text-transform:uppercase; color:#4a7fb5; margin-bottom:1.5mm; }
.none { color:#999; font-style:italic; margin-left:6mm; font-size:10pt; }
.recall { background:#f4f0fa; border:1px solid #e0d3ef; border-radius:5px; padding:3mm 4mm;
          margin:0 0 4mm; }
.recall:before { content:"written from memory"; display:block; font-size:7.5pt;
                 letter-spacing:.08em; text-transform:uppercase; color:#7a5c96; margin-bottom:1.5mm; }
.foot { margin-top:10mm; border-top:1px solid #eee; padding-top:3mm; color:#999; font-size:8.5pt; }
"""


def _slug(text, fallback="reading"):
    name = re.sub(r"^https?://", "", (text or "")).strip()
    name = re.sub(r"[^\w\s.-]+", " ", name).strip()
    return re.sub(r"\s+", "-", name)[:60] or fallback


def build_html(conn, session_id):
    row = conn.execute("""SELECT s.text_source, s.started_at, s.self_report, u.display_name,
                                 s.session_id FROM sessions s JOIN users u USING(user_id)
                          WHERE s.session_id = ?""", (session_id,)).fetchone()
    if not row:
        return None, None
    source, started, feeling, reader, sid = row
    notes = conn.execute("""SELECT quote, note, tag FROM notes WHERE session_id = ?
                            ORDER BY word_index""", (session_id,)).fetchall()
    recalls = conn.execute("""SELECT answer FROM checks WHERE session_id = ? AND kind = 'recall'
                              ORDER BY check_id""", (session_id,)).fetchall()
    if not notes and not recalls:
        return None, None

    by_tag = {}
    for quote, note, tag in notes:
        by_tag.setdefault(tag, []).append((quote, note))

    parts = [f"<h1>{H.escape(_pretty_title(source))}</h1>",
             f"<div class='meta'>{H.escape(reader)} &middot; {started[:16].replace('T', ' ')}"
             + (f" &middot; felt <b>{H.escape(feeling)}</b>" if feeling else "")
             + f" &middot; <b>{len(notes)}</b> highlight(s)"
             + (f" &middot; <b>{len(recalls)}</b> recall answer(s)" if recalls else "")
             + "</div>"]

    for tag in TAG_ORDER:
        if tag not in by_tag:
            continue
        parts.append(f"<h2><span class='dot' style='background:{TAG_COLOR[tag]}'></span>"
                     f"{TAG_LABEL[tag]}</h2>")
        for quote, note in by_tag[tag]:
            parts.append("<div class='item'>")
            parts.append(f"<div class='quote'>{H.escape(quote or '')}</div>")
            if (note or "").strip():
                parts.append(f"<div class='mine'>{H.escape(note.strip())}</div>")
            else:
                parts.append("<div class='none'>(highlighted, no note written)</div>")
            parts.append("</div>")

    if recalls:
        parts.append("<h2><span class='dot' style='background:#7a5c96'></span>"
                     "What you recalled at section breaks</h2>")
        for (answer,) in recalls:
            parts.append(f"<div class='recall'>{H.escape(answer or '')}</div>")

    parts.append(f"<div class='foot'>Source: {H.escape(source or '')} &middot; "
                 f"session {sid} &middot; generated by your reading recorder</div>")
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>" \
           f"<body>{''.join(parts)}</body></html>", (source, started)


def _pretty_title(source):
    if not source:
        return "Reading notes"
    name = re.sub(r"^https?://", "", source).rstrip("/").split("/")[-1]
    name = re.sub(r"\.(html?|md|txt|pdf)$", "", name).replace("-", " ").replace("_", " ")
    return name.strip().capitalize() or "Reading notes"


def export_pdf(conn, session_id, out_dir=OUT_DIR):
    html, meta = build_html(conn, session_id)
    if not html:
        return None
    os.makedirs(out_dir, exist_ok=True)
    source, started = meta
    stem = f"{_slug(source)}_{started[:10]}_s{session_id}"
    html_path = os.path.join(out_dir, stem + ".html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    from selenium import webdriver
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    d = webdriver.Chrome(options=opts)
    try:
        d.get("file://" + os.path.abspath(html_path))
        pdf = d.execute_cdp_cmd("Page.printToPDF", {
            "printBackground": True, "preferCSSPageSize": True})
        pdf_path = os.path.join(out_dir, stem + ".pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(base64.b64decode(pdf["data"]))
        return pdf_path
    finally:
        d.quit()


if __name__ == "__main__":
    conn = _db.connect(DB)
    if "--user" in sys.argv:
        uid = int(sys.argv[sys.argv.index("--user") + 1])
        for (sid,) in conn.execute("SELECT session_id FROM sessions WHERE user_id=? ORDER BY session_id", (uid,)):
            p = export_pdf(conn, sid)
            if p:
                print("wrote", p)
    else:
        sid = int(sys.argv[1]) if len(sys.argv) > 1 else conn.execute(
            "SELECT session_id FROM notes ORDER BY note_id DESC LIMIT 1").fetchone()[0]
        p = export_pdf(conn, sid)
        print("wrote " + p if p else f"session {sid} has no notes")
