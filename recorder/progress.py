"""
Your reading account: what the sessions add up to.

Answers the questions Emma asked for -- do tired days read faster or slower?
is there more skipping? which reading mode holds attention longest? -- and is
honest about which numbers are trustworthy (gaze needs good calibration; blink,
lean and frown do not).

    python3 recorder/progress.py            # writes + opens recorder/progress.html
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from recorder import _env  # noqa: E402,F401  -- re-launches in the mlclass env if needed
import os
import statistics
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import db as _db          # noqa: E402  (its connect() migrates the schema)

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "reading.db")
OUT = os.path.join(HERE, "progress.html")

GOOD_CALIB = 150          # px: above this, gaze-derived numbers are not comparable


def sessions(conn, user_id):
    rows = conn.execute("""
        SELECT s.session_id, s.started_at, s.self_report, s.device_label, s.glasses,
               s.calib_error, s.text_source, sm.duration_s, sm.face_rate,
               sm.words_visited, sm.median_dwell_ms, sm.mean_eye_span, sm.mean_frown
        FROM sessions s LEFT JOIN session_summary sm USING(session_id)
        WHERE s.user_id = ? AND sm.n_samples > 0
        ORDER BY s.started_at""", (user_id,)).fetchall()
    out = []
    for r in rows:
        sid = r[0]
        # per-session behaviour that does NOT depend on calibration
        blink = conn.execute("""
            SELECT AVG(CASE WHEN ear < 0.7 * (SELECT AVG(ear) FROM samples
                            WHERE session_id = ? AND ear IS NOT NULL)
                       THEN 1.0 ELSE 0 END)
            FROM samples WHERE session_id = ? AND ear IS NOT NULL""", (sid, sid)).fetchone()[0]
        skipped = conn.execute("""
            SELECT AVG(CASE WHEN word_index IS NULL THEN 1.0 ELSE 0 END)
            FROM samples WHERE session_id = ? AND face_detected = 1""", (sid,)).fetchone()[0]
        notes = conn.execute("SELECT COUNT(*) FROM notes WHERE session_id = ?", (sid,)).fetchone()[0]
        wpm = (r[9] / (r[7] / 60.0)) if (r[7] and r[9]) else None
        out.append(dict(sid=sid, when=r[1][:16].replace("T", " "), feel=(r[2] or "?"),
                        device=r[3], glasses=r[4], calib=r[5], text=(r[6] or "")[:40],
                        dur=r[7], face=r[8], words=r[9], dwell=r[10], span=r[11],
                        frown=r[12], blink=blink, offtext=skipped, notes=notes, wpm=wpm))
    return out


def profile_time(conn, user_id):
    """Seconds spent in each reading mode -- 'which model keeps attention longest'."""
    rows = conn.execute("""
        SELECT e.session_id, e.t_ms, e.value, sm.duration_s
        FROM events e JOIN sessions s USING(session_id)
        LEFT JOIN session_summary sm ON sm.session_id = e.session_id
        WHERE s.user_id = ? AND e.kind = 'profile'
        ORDER BY e.session_id, e.t_ms""", (user_id,)).fetchall()
    totals = {}
    for i, (sid, t_ms, prof, dur) in enumerate(rows):
        nxt = rows[i + 1][1] if i + 1 < len(rows) and rows[i + 1][0] == sid else (dur or 0) * 1000
        if prof:
            totals[prof] = totals.get(prof, 0) + max(0, (nxt - t_ms) / 1000.0)
    return totals


def bar(value, vmax, color):
    w = 0 if not vmax else max(1, round(100 * value / vmax))
    return (f"<div class='bar'><i style='width:{w}%;background:{color}'></i>"
            f"<span>{value:.0f}</span></div>")


def build(user_id=1):
    conn = _db.connect(DB)             # creates/migrates missing tables (e.g. notes)
    name = conn.execute("SELECT display_name FROM users WHERE user_id=?", (user_id,)).fetchone()[0]
    ss = sessions(conn, user_id)
    if not ss:
        print("no sessions with samples yet"); return None

    by_feel = {}
    for s in ss:
        by_feel.setdefault(s["feel"].lower(), []).append(s)

    def avg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else None

    max_wpm = max((s["wpm"] or 0) for s in ss) or 1
    rows_html = "".join(
        f"<tr><td>{s['sid']}</td><td>{s['when']}</td><td>{s['feel']}</td>"
        f"<td>{s['device']}</td><td class='{'ok' if (s['calib'] or 999) < GOOD_CALIB else 'bad'}'>"
        f"{(s['calib'] or 0):.0f}px</td>"
        f"<td>{(s['dur'] or 0)/60:.1f} min</td>"
        f"<td>{bar(s['wpm'] or 0, max_wpm, '#7a5c96')}</td>"
        f"<td>{(s['blink'] or 0):.0%}</td><td>{(s['offtext'] or 0):.0%}</td>"
        f"<td>{s['notes'] or ''}</td><td class='txt'>{s['text']}</td></tr>"
        for s in ss)

    feel_html = ""
    for feel, rows in sorted(by_feel.items()):
        w, b, o = avg(rows, "wpm"), avg(rows, "blink"), avg(rows, "offtext")
        feel_html += (f"<tr><td><b>{feel}</b></td><td>{len(rows)}</td>"
                      f"<td>{w:.0f} wpm</td>" if w else
                      f"<tr><td><b>{feel}</b></td><td>{len(rows)}</td><td>-</td>")
        feel_html += (f"<td>{b:.0%}</td>" if b is not None else "<td>-</td>")
        feel_html += (f"<td>{o:.0%}</td></tr>" if o is not None else "<td>-</td></tr>")

    pt = profile_time(conn, user_id)
    if pt:
        mx = max(pt.values())
        prof_html = "".join(f"<tr><td><b>{k}</b></td><td>{bar(v/60, mx/60, '#2e7d32')} min</td></tr>"
                            for k, v in sorted(pt.items(), key=lambda kv: -kv[1]))
    else:
        prof_html = ("<tr><td colspan=2><i>No reading-mode switches recorded yet. "
                     "Read an adaptive page and try Comfort / Focus / Skim.</i></td></tr>")

    att = conn.execute("""
        SELECT COUNT(*) FROM events e JOIN sessions s USING(session_id)
        WHERE s.user_id = ? AND e.kind = 'attention' AND e.value LIKE 'low%'""",
        (user_id,)).fetchone()[0]
    rec = conn.execute("""
        SELECT c.answer, c.session_id FROM checks c JOIN sessions s USING(session_id)
        WHERE s.user_id = ? AND c.kind = 'recall' ORDER BY c.check_id DESC LIMIT 8""",
        (user_id,)).fetchall()
    focus_html = (f"<p>Focus mode was triggered by your own gaze <b>{att}</b> time(s) "
                  f"across your sessions.</p>")
    if rec:
        focus_html += "<p>What you wrote from memory at section breaks:</p>" + "".join(
            f"<div class='note'><b>{a}</b><em>session {sid}</em></div>" for a, sid in rec)
    else:
        focus_html += ("<p class='caveat'>No recall answers yet. The one-sentence prompt "
                       "at each section break is the strongest known way to hold attention "
                       "in long technical reading &mdash; retrieving beats re-reading &mdash; "
                       "and your answers double as comprehension ground truth, which the "
                       "eye data alone can never give you.</p>")

    perf = _db.profile_performance(conn, user_id)
    if perf:
        rows_p = ""
        for prof, d in sorted(perf.items(), key=lambda kv: -kv[1]["seconds"]):
            wpm = f"{d['wpm']:.0f}" if d["wpm"] else "-"
            ot = f"{d['on_text_pct']:.0%}" if d["on_text_pct"] is not None else "-"
            lows = f"{d['lows_per_10min']:.1f}" if d["lows_per_10min"] is not None else "-"
            rows_p += (f"<tr><td><b>{prof}</b></td><td>{d['seconds']/60:.0f} min</td>"
                       f"<td>{wpm} words/min</td><td>{ot} on text</td>"
                       f"<td>{lows} attention dips / 10 min</td></tr>")
        perf_html = ("<table><tr><th>mode</th><th>time</th><th>coverage</th>"
                     "<th>gaze on text</th><th>drift</th></tr>" + rows_p + "</table>"
                     "<p class='caveat'>Careful with this one: you choose the mode, so the "
                     "comparison is not an experiment. If you switch to focus for the hard "
                     "passages, focus will look slower even if it helps. To settle it, read "
                     "one chapter in one mode and the next chapter in the other on the same "
                     "day, and compare those pairs.</p>")
    else:
        perf_html = ("<p class='caveat'>No mode switches recorded yet.</p>")

    notes = conn.execute("""
        SELECT n.session_id, n.quote, n.note FROM notes n JOIN sessions s USING(session_id)
        WHERE s.user_id = ? ORDER BY n.note_id DESC LIMIT 25""", (user_id,)).fetchall()
    notes_html = "".join(f"<div class='note'><b>&ldquo;{q}&rdquo;</b>"
                         f"<span>{nt or '<i>no note written</i>'}</span>"
                         f"<em>session {sid}</em></div>" for sid, q, nt in notes) or \
        "<i>Nothing marked yet. Press M while reading to mark a passage.</i>"

    from recorder import personalize
    pm = personalize.load(conn, user_id)
    if pm:
        pstate_html = (
            f"<table><tr><td>learned from you</td><td>{bar(pm['weight_you']*100, 100, '#7a5c96')}%</td></tr>"
            f"<tr><td>your words used</td><td>{pm['n_words']:,} from {pm['sessions']} well-calibrated sessions</td></tr>"
            f"<tr><td>your own effect of word length</td><td>{pm['own_coef']['word_len']:+.3f}</td></tr>"
            f"<tr><td>general model's word length</td><td>+0.557</td></tr></table>"
            "<p class='caveat'>Every session you record re-fits this. The blend shifts "
            "toward you as your own reading accumulates &mdash; half yours at about 3,000 "
            "well-tracked words. Only sessions calibrated under 150px are used, because "
            "word-level gaze from a badly calibrated session is noise, and training on "
            "noise would make the page worse rather than better.</p>")
    else:
        pstate_html = ("<p class='caveat'>Not personalised yet &mdash; pages use the "
                       "general reading model. It needs at least 150 words from sessions "
                       "calibrated under 150px.</p>")

    prefs = _db.get_prefs(conn, user_id)
    typo = prefs.get("typo") or {}
    rows_t = ""
    for mode in ("comfort", "focus", "skim"):
        t = typo.get(mode) or {}
        if t:
            parts = [f"{k} {v}" for k, v in t.items()]
            rows_t += f"<tr><td><b>{mode}</b></td><td>{', '.join(parts)}</td></tr>"
    n_layout = conn.execute("""SELECT COUNT(*) FROM events e JOIN sessions s USING(session_id)
                               WHERE s.user_id=? AND e.kind='layout'""", (user_id,)).fetchone()[0]
    adapt_html = (
        "<table><tr><th>what</th><th>how it adapts</th></tr>"
        f"<tr><td>which words are marked</td><td>your own dwell data, refit after every "
        f"well-calibrated session ({pm['weight_you']:.0%} you)</td></tr>" if pm else
        "<table><tr><th>what</th><th>how it adapts</th></tr>"
        "<tr><td>which words are marked</td><td>general model until you have enough tracked reading</td></tr>")
    adapt_html += (
        f"<tr><td>how many words are marked</td><td>follows your mode usage &mdash; more time in "
        f"Focus means a calmer Comfort</td></tr>"
        f"<tr><td>read-along speed</td><td>your measured pace, then whatever you set "
        f"({prefs.get('wpm') or 'measured'} wpm)</td></tr>"
        f"<tr><td>default mode, voice, theme</td><td>remembered: {prefs.get('profile') or 'most used'}, "
        f"{prefs.get('voice') or 'best available'}, {prefs.get('theme') or 'auto'}</td></tr>"
        f"<tr><td>type size, spacing, line length</td><td>{'your own settings per mode (below)' if rows_t else 'the mode defaults &mdash; press T while reading to change them'}</td></tr>"
        "</table>")
    if rows_t:
        adapt_html += "<table>" + rows_t + "</table>"
    adapt_html += (
        "<p class='caveat'><b>What this deliberately does not do:</b> it does not run experiments on "
        "your typography behind your back. We looked hard at doing that, and the honest conclusion "
        "was that a webcam cannot measure the thing a font change moves (word-level gaze error here "
        "is 90&ndash;140 px against 18&ndash;42 px lines), so any automatic 'this layout is better' "
        "verdict would be noise wearing a confidence interval. What it does instead is keep every "
        "setting you choose, log every change, and show you here what you actually keep. "
        f"Layout changes re-measured so far: {n_layout}.</p>")

    good = [s for s in ss if (s["calib"] or 999) < GOOD_CALIB]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{name}'s reading account</title><style>
body {{ font:15px/1.6 -apple-system,sans-serif; max-width:960px; margin:32px auto;
        padding:0 20px; color:#1a1a1a; background:#faf8f4; }}
h1 {{ font-size:26px; margin-bottom:2px; }} h2 {{ font-size:18px; margin-top:32px; }}
.sub {{ color:#666; margin-top:0; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid #e6e0d8; }}
th {{ color:#666; font-weight:600; }}
.bar {{ position:relative; background:#eee; border-radius:4px; height:16px; min-width:80px; }}
.bar i {{ display:block; height:100%; border-radius:4px; }}
.bar span {{ position:absolute; left:6px; top:0; font-size:11px; line-height:16px; color:#222; }}
.ok {{ color:#2e7d32; }} .bad {{ color:#b03030; }}
.txt {{ color:#777; font-size:12px; }}
.note {{ border-left:3px solid #7a5c96; padding:6px 10px; margin:8px 0; background:#fff; }}
.note b {{ display:block; }} .note em {{ color:#999; font-size:12px; }}
.caveat {{ background:#fff6e0; border:1px solid #f0e0b0; padding:10px 14px; border-radius:8px; }}
</style></head><body>
<h1>{name}'s reading account</h1>
<p class="sub">{len(ss)} sessions &middot; {sum((s['dur'] or 0) for s in ss)/60:.0f} minutes read
&middot; {len(good)} with trustworthy calibration (&lt;{GOOD_CALIB}px)</p>

<h2>How personalised your pages are</h2>
{pstate_html}

<h2>What adapts, and what you set yourself</h2>
{adapt_html}

<h2>Every session</h2>
<table><tr><th>#</th><th>when</th><th>felt</th><th>screen</th><th>calib</th><th>length</th>
<th>words/min</th><th>blink rate</th><th>off-text</th><th>notes</th><th>text</th></tr>
{rows_html}</table>

<h2>Tired days vs rested days</h2>
<table><tr><th>felt</th><th>sessions</th><th>reading speed</th><th>blink rate</th>
<th>eyes off the text</th></tr>{feel_html}</table>
<p class="caveat">Read this as a hint, not a result: with a handful of sessions the
differences are still well inside noise. It becomes real once each mood has
several sessions on the same screen. Blink rate and off-text time do not depend on
calibration, so they are comparable across all sessions; words/min needs good
calibration to mean anything.</p>

<h2>Which reading mode holds you longest</h2>
<table>{prof_html}</table>

<h2>How you read in each mode</h2>
{perf_html}

<h2>Attention and recall</h2>
{focus_html}

<h2>What you marked</h2>
{notes_html}
</body></html>"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    return OUT


if __name__ == "__main__":
    path = build()
    if path:
        print("wrote", path)
        webbrowser.open("file://" + path)
