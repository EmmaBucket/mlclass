"""
Build an adaptive reading page from a plain text / markdown file.

The page has three ingredients, mapped to the project's findings:
  PREDICTIVE  hard words get extra letter-spacing and weight, scored by the
              text-difficulty model trained on GECO (word length up, frequency
              down -- the two effects that also replicated in Emma's own
              sessions). Computed here in Python; the page just wears classes.
  AGENCY      a visible profile switcher: Comfort / Focus / Skim. The choice is
              written to window.__profile so the recorder logs every switch --
              user preference becomes data, not anecdote.
  RELIEF      continuous read-along via the browser's built-in speech
              synthesis: switched on ONCE at the top (no per-paragraph clicking
              fatigue), karaoke-highlights the word being spoken, and defaults
              to the reader's own measured pace from their recorded sessions.
              A +/- control tunes it; every change is logged as an event.

Every word sits in its own <span data-w=INDEX>, so the recorder's word map and
gaze attribution keep working across all profiles.
"""
import csv
import html
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the exported GECO model: P(long dwell | text). Provenance: trained on all
# 19 readers, target = top-10% dwell within reader; see export in git history.
MEAN = {"word_len": 5.334, "logfreq": 6.049}
SCALE = {"word_len": 2.684, "logfreq": 2.276}
COEF = {"word_len": 0.557, "logfreq": -0.287}
INTERCEPT = -0.246

_freq = None
def word_logfreq(word):
    global _freq
    if _freq is None:
        with open(os.path.join(HERE, "geco_word_freq.csv")) as fh:
            _freq = {row["WORD"]: math.log1p(int(row["count"]))
                     for row in csv.DictReader(fh)}
    return _freq.get(word.lower().strip('.,;:!?"()—’\''), 0.0)


def hardness(word):
    """0..1: probability this word gets a long dwell, per the GECO model."""
    z_len = (len(word) - MEAN["word_len"]) / SCALE["word_len"]
    z_frq = (word_logfreq(word) - MEAN["logfreq"]) / SCALE["logfreq"]
    logit = INTERCEPT + COEF["word_len"] * z_len + COEF["logfreq"] * z_frq
    return 1 / (1 + math.exp(-logit))


CSS = """
:root { --ink:#1a1a1a; --paper:#faf8f4; --accent:#7a5c96; }
body { margin:0; background:var(--paper); color:var(--ink); }
#bar { position:sticky; top:0; background:var(--paper); border-bottom:1px solid #ddd;
       padding:10px 16px; display:flex; gap:8px; align-items:center; z-index:9; }
#bar button { font:15px/1 -apple-system,sans-serif; padding:8px 14px; border:1px solid #bbb;
              border-radius:16px; background:white; cursor:pointer; }
#bar button.on { background:var(--accent); color:white; border-color:var(--accent); }
#brand { font:700 15px -apple-system,sans-serif; color:var(--accent); margin-right:6px; }
#hint { background:#f3ecfb; border-bottom:1px solid #e0d3ef; padding:10px 16px;
        font:14px/1.5 -apple-system,sans-serif; display:flex; gap:10px; align-items:center; }
#hint .h3 { padding:0 3px; }
#hint button { margin-left:auto; border:none; background:none; cursor:pointer;
               font-size:15px; opacity:.5; }
#text { margin:32px auto; padding:0 24px; }
p { position:relative; }
.speak { position:absolute; left:-34px; top:2px; border:none; background:none;
         cursor:pointer; font-size:15px; opacity:.35; }
.speak:hover { opacity:1; }
#rbar { margin-left:auto; display:flex; gap:6px; align-items:center;
        font:14px -apple-system,sans-serif; }
#rbar button { border-radius:8px; padding:6px 10px; }
#play.on { background:#2e7d32; color:white; border-color:#2e7d32; }
.speaking { background:#ffe9a8; border-radius:3px; }
body.skim .speaking { opacity:1 !important; }

/* COMFORT: the dyslexia-informed default. Big type, generous leading, short
   lines, extra inter-word air. Hard words get letter-spacing + weight. */
body.comfort #text { max-width:34rem; font:22px/1.9 "Atkinson Hyperlegible",Verdana,sans-serif;
                     word-spacing:.16em; }
body.comfort p { margin:0 0 1.4em; }
body.comfort .h2 { letter-spacing:.045em; font-weight:600; }
body.comfort .h3 { letter-spacing:.09em; font-weight:700; background:#efe8f7; border-radius:3px; }

/* FOCUS: the deep-reading block. Uniform, calm, minimal signalling -- dense on
   purpose; only the very hardest words get a nudge. */
body.focus #text { max-width:44rem; font:19px/1.65 Georgia,serif; }
body.focus p { margin:0 0 1.1em; }
body.focus .h3 { letter-spacing:.05em; }

/* SKIM: typographic differentiation for cascading. First words of each
   paragraph bolded, hard words dimmed LESS than easy ones -- the skeleton
   stands out, the reader's eye can rappel down it. */
body.skim #text { max-width:38rem; font:19px/1.75 -apple-system,sans-serif; }
body.skim p { margin:0 0 1.3em; }
body.skim .lead { font-weight:700; }
body.skim span[data-w] { opacity:.62; }
body.skim .lead, body.skim .h2, body.skim .h3 { opacity:1; }
"""

JS = """
const PROFILES = ["comfort","focus","skim"];
window.__profile = localStorage.getItem("profile") || "comfort";
function setProfile(p){
  window.__profile = p; localStorage.setItem("profile", p);
  document.body.className = p;
  for (const b of document.querySelectorAll("#bar button[data-p]"))
    b.classList.toggle("on", b.dataset.p === p);
}
document.addEventListener("DOMContentLoaded", () => {
  for (const b of document.querySelectorAll("#bar button[data-p]"))
    b.onclick = () => setProfile(b.dataset.p);
  for (const b of document.querySelectorAll(".speak"))
    b.onclick = () => startFrom(+b.dataset.par);   // "start reading from here"
  document.getElementById("play").onclick = () =>
    tts.on ? stopTTS() : startFrom(tts.par || 0);
  document.getElementById("slower").onclick = () => bumpWpm(-15);
  document.getElementById("faster").onclick = () => bumpWpm(+15);
  setProfile(window.__profile);
  updateTTSUI();
  const hint = document.getElementById("hint");
  if (localStorage.getItem("hint_seen")) hint.style.display = "none";
  document.getElementById("hintx").onclick = () => {
    hint.style.display = "none"; localStorage.setItem("hint_seen", "1");
  };
});

// ---------------- continuous read-along ----------------
// One decision at the top, then it flows: paragraph n ends -> n+1 begins,
// the page scrolls itself, and the spoken word is highlighted so eyes and
// voice stay locked together. Default pace = the reader's measured wpm.
const tts = { on:false, par:0,
              wpm:+(localStorage.getItem("wpm") || DEFAULT_WPM) };
function publish(){                    // the recorder polls this and logs changes
  window.__tts = JSON.stringify({on:tts.on, wpm:tts.wpm});
  updateTTSUI();
}
function updateTTSUI(){
  const p = document.getElementById("play");
  p.textContent = tts.on ? "\u25a0 stop" : "\u25b6 read along";
  p.classList.toggle("on", tts.on);
  document.getElementById("wpm").textContent = tts.wpm + " wpm";
}
function paragraphs(){ return [...document.querySelectorAll("#text p")]; }
function clearHi(){ for (const s of document.querySelectorAll(".speaking")) s.classList.remove("speaking"); }
function stopTTS(){ tts.on = false; speechSynthesis.cancel(); clearHi(); publish(); }
function bumpWpm(d){
  tts.wpm = Math.min(320, Math.max(60, tts.wpm + d));
  localStorage.setItem("wpm", tts.wpm); publish();
  if (tts.on) { speechSynthesis.cancel(); speakPar(tts.par); }  // take effect now
}
function startFrom(i){ tts.on = true; publish(); speechSynthesis.cancel(); speakPar(i); }
function speakPar(i){
  const pars = paragraphs();
  if (!tts.on || i >= pars.length) { stopTTS(); return; }
  tts.par = i;
  const spans = [...pars[i].querySelectorAll("span[data-w]")];
  // build the utterance from the word spans and remember where each word
  // starts, so boundary events (charIndex) map back to a span to highlight
  let text = "", starts = [];
  for (const sp of spans) { starts.push(text.length); text += sp.innerText + " "; }
  const u = new SpeechSynthesisUtterance(text);
  u.rate = tts.wpm / 180;                        // rate 1.0 ~ 180 wpm
  u.onboundary = (e) => {
    if (e.name && e.name !== "word") return;
    let k = starts.findIndex(st => st > e.charIndex) - 1;
    if (k < -1+1 && starts[starts.length-1] <= e.charIndex) k = starts.length-1;
    if (k >= 0) { clearHi(); spans[k].classList.add("speaking");
                  spans[k].scrollIntoView({block:"center", behavior:"smooth"}); }
  };
  u.onend = () => { if (tts.on) speakPar(i + 1); };
  speechSynthesis.speak(u);
}
"""


def build_page(text_path, out_dir, wpm=135):
    """text/markdown file -> adaptive html page. Returns the output path.
    wpm: the reader's own measured pace (recorder computes it from their
    best-calibrated sessions); becomes the read-along default speed."""
    raw = open(text_path, encoding="utf-8", errors="replace").read()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    widx = 0
    body = []
    for par in paragraphs:
        # light markdown: strip #/##/** noise but keep the text
        par = re.sub(r"^#{1,6}\s*", "", par)
        par = par.replace("**", "").replace("__", "")
        words_html = []
        for j, w in enumerate(par.split()):
            h = hardness(w)
            cls = ["lead"] if j < 2 else []          # skim skeleton: first 2 words
            if h > 0.75: cls.append("h3")            # hardest: spacing + mark
            elif h > 0.6: cls.append("h2")           # hard: spacing + weight
            words_html.append(f'<span data-w="{widx}"{" class=" + chr(34) + " ".join(cls) + chr(34) if cls else ""}>{html.escape(w)}</span>')
            widx += 1
        body.append(f'<p><button class="speak" data-par="{len(body)}" '
                    'title="read aloud from here">&#128264;</button>'
                    + " ".join(words_html) + "</p>")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(text_path) + ".adaptive.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><html><head><meta charset='utf-8'>"
                 f"<title>{html.escape(os.path.basename(text_path))}</title>"
                 f"<style>{CSS}</style><script>const DEFAULT_WPM={int(wpm)};{JS}</script></head>"
                 "<body class='comfort'>"
                 "<div id='bar'><span id='brand'>&#128065; Adaptive Reading</span>"
                 "<b style='font:14px -apple-system'>mode:</b>"
                 "<button data-p='comfort'>Comfort</button>"
                 "<button data-p='focus'>Focus</button>"
                 "<button data-p='skim'>Skim</button>"
                 "<div id='rbar'><button id='slower'>&minus;</button>"
                 "<span id='wpm'></span><button id='faster'>+</button>"
                 "<button id='play'>&#9654; read along</button></div></div>"
                 "<div id='hint'><span>This page adapts to you: "
                 "<span class='h3'>marked words</span> are ones readers usually find "
                 "hard &middot; pick a mode above &middot; &#9654; reads along at your "
                 "own measured pace.</span><button id='hintx' title='got it'>&#10005;"
                 "</button></div>"
                 f"<div id='text'>{''.join(body)}</div></body></html>")
    return out
