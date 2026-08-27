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

_MODEL = None          # per-reader coefficients, set by build_page()

_freq = None
def word_logfreq(word):
    global _freq
    if _freq is None:
        with open(os.path.join(HERE, "geco_word_freq.csv")) as fh:
            _freq = {row["WORD"]: math.log1p(int(row["count"]))
                     for row in csv.DictReader(fh)}
    return _freq.get(word.lower().strip('.,;:!?"()—’\''), 0.0)


def hardness(word, model=None):
    """0..1: probability this word gets a long dwell.

    Uses the reader's personalised coefficients when they have enough of their
    own reading recorded, otherwise the GECO population model."""
    coef = (model or _MODEL or {}).get("coef", COEF)
    intercept = (model or _MODEL or {}).get("intercept", INTERCEPT)
    z_len = (len(word) - MEAN["word_len"]) / SCALE["word_len"]
    z_frq = (word_logfreq(word) - MEAN["logfreq"]) / SCALE["logfreq"]
    logit = intercept + coef["word_len"] * z_len + coef["logfreq"] * z_frq
    return 1 / (1 + math.exp(-logit))


def _state_line(model):
    """One line telling the reader how much of this page is tuned to them."""
    if not model:
        return ("Word difficulty: general reading model (not yet personalised) &mdash; "
                "read a few well-calibrated sessions and this page starts tuning to you.")
    return (f"Word difficulty: <b>{model['weight_you']:.0%} learned from your own reading</b> "
            f"({model['n_words']:,} words across {model['sessions']} sessions), "
            f"{1 - model['weight_you']:.0%} general reading model.")


CSS = """
:root { --ink:#1a1a1a; --paper:#faf8f4; --accent:#7a5c96; }
body { margin:0; background:var(--paper); color:var(--ink); }
#bar { position:sticky; top:0; background:var(--paper); border-bottom:1px solid #ddd;
       padding:10px 16px; display:flex; gap:8px; align-items:center; z-index:9; }
#bar button { font:15px/1 -apple-system,sans-serif; padding:8px 14px; border:1px solid #bbb;
              border-radius:16px; background:white; cursor:pointer; }
#bar button.on { background:var(--accent); color:white; border-color:var(--accent); }
#brand { font:700 15px -apple-system,sans-serif; color:var(--accent); margin-right:6px; }
#pstate { background:#efe8f7; color:#4b3b60; padding:6px 16px;
          font:12.5px -apple-system,sans-serif; border-bottom:1px solid #e0d3ef; }
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
#rbar select { font:13px -apple-system,sans-serif; padding:5px; border-radius:8px;
               border:1px solid #bbb; background:white; max-width:140px; }
#play.on { background:#2e7d32; color:white; border-color:#2e7d32; }
.speaking { background:#ffe9a8; border-radius:3px; }
.marked { background:#d7f0ff; box-shadow:0 1px 0 #67b7e6; border-radius:2px; }
.marked.hasnote { background:#c9e8c9; box-shadow:0 1px 0 #5aa75a; }
/* sits BELOW the toolbar (bar is z-index 9, top ~52px) so the play button,
   voice picker and notes button are never covered */
#notes { position:fixed; right:0; top:52px; bottom:0; width:300px; background:#fff;
         border-left:1px solid #ddd; padding:14px 14px 40px; overflow:auto;
         display:none; font:14px/1.5 -apple-system,sans-serif; z-index:8;
         box-shadow:-4px 0 12px rgba(0,0,0,.06); }
#notes.open { display:block; }
/* and the text moves over instead of hiding underneath */
body.notes-open #text { margin-right:336px; }   /* 300 panel + border + shadow + air */
@media (max-width:820px) { body.notes-open #text { margin-right:0; } }
#notes h4 { margin:0 0 10px; font-size:15px; padding-right:26px; }
#noteclose { position:absolute; right:10px; top:10px; border:none; background:none;
             font-size:20px; cursor:pointer; opacity:.5; line-height:1; }
#noteclose:hover { opacity:1; }
#notes .note { border-bottom:1px solid #eee; padding:8px 0; cursor:pointer; }
#notes .note b { display:block; color:#555; font-weight:600; }
#notes textarea { width:100%; height:54px; font:13px -apple-system,sans-serif;
                  border:1px solid #ccc; border-radius:6px; padding:6px; }
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
window.__profile = localStorage.getItem("profile") || DEFAULT_PROFILE;
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
  fillVoiceMenu();
  document.getElementById("voice").onchange = (e) => {
    localStorage.setItem("voice", e.target.value);
    if (tts.on) { speechSynthesis.cancel(); speakPar(tts.par); }
  };
  document.getElementById("slower").onclick = () => bumpWpm(-15);
  document.getElementById("faster").onclick = () => bumpWpm(+15);
  setProfile(window.__profile);
  updateTTSUI();
  document.getElementById("marks").onclick = () => showNotes();
  document.getElementById("noteclose").onclick = () => showNotes(false);
  for (const sp of document.querySelectorAll("#text span[data-w]"))
    sp.onclick = () => markWord(sp);
  restoreMarks();
  const hint = document.getElementById("hint");
  if (localStorage.getItem("hint_seen")) hint.style.display = "none";
  document.getElementById("hintx").onclick = () => {
    hint.style.display = "none"; localStorage.setItem("hint_seen", "1");
  };
});

// ---------------- marking + notes ----------------
// Reading and writing pull in opposite directions: stopping to type loses your
// place and breaks the flow the layout is trying to protect. So marking is one
// keystroke (M) or one click, the mark PERSISTS, and the note can be written
// later -- the passage is still highlighted and one click away.
const marks = JSON.parse(localStorage.getItem("marks") || "{}");
function saveMarks(){
  localStorage.setItem("marks", JSON.stringify(marks));
  window.__marks = JSON.stringify(Object.keys(marks).map(k => ({
    w: +k, note: marks[k].note || "", text: marks[k].text })));   // recorder logs this
  renderNotes();
}
function markWord(sp){
  if (!sp) return;
  const id = sp.dataset.w;
  if (marks[id]) { delete marks[id]; sp.classList.remove("marked", "hasnote"); }
  else {
    const sibs = [...sp.parentElement.querySelectorAll("span[data-w]")];
    const k = sibs.indexOf(sp);
    marks[id] = { text: sibs.slice(Math.max(0, k - 3), k + 5).map(x => x.innerText).join(" "),
                  note: "" };
    sp.classList.add("marked");
  }
  saveMarks();
}
function renderNotes(){
  const list = document.getElementById("notelist");
  if (!list) return;
  const ids = Object.keys(marks).sort((a, b) => a - b);
  list.innerHTML = ids.length ? "" : "<i>Nothing marked yet.<br>Press M while reading, "
                                     + "or click a word, to mark it.</i>";
  for (const id of ids){
    const d = document.createElement("div");
    d.className = "note";
    d.innerHTML = "<b>&ldquo;" + marks[id].text + "&rdquo;</b>";
    const ta = document.createElement("textarea");
    ta.value = marks[id].note; ta.placeholder = "your note...";
    ta.onchange = () => {
      marks[id].note = ta.value;
      const sp = document.querySelector(`span[data-w="${id}"]`);
      if (sp) sp.classList.toggle("hasnote", !!ta.value.trim());
      saveMarks();
    };
    d.onclick = (e) => {
      if (e.target === ta) return;
      const sp = document.querySelector(`span[data-w="${id}"]`);
      if (sp) sp.scrollIntoView({block:"center", behavior:"smooth"});
    };
    d.appendChild(ta); list.appendChild(d);
  }
}
function restoreMarks(){
  for (const id of Object.keys(marks)){
    const sp = document.querySelector(`span[data-w="${id}"]`);
    if (sp){ sp.classList.add("marked"); if (marks[id].note) sp.classList.add("hasnote"); }
  }
  renderNotes(); saveMarks();
}
// M marks whatever you are reading right now: the spoken word if the voice is
// running, otherwise the word nearest the middle of the screen.
function currentWord(){
  const spoken = document.querySelector(".speaking");
  if (spoken) return spoken;
  const mid = window.innerHeight / 2;
  let best = null, bestD = 1e9;
  for (const sp of document.querySelectorAll("#text span[data-w]")){
    const r = sp.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) continue;
    const d = Math.abs(r.top + r.height / 2 - mid);
    if (d < bestD){ bestD = d; best = sp; }
  }
  return best;
}
function showNotes(open){
  const panel = document.getElementById("notes");
  const want = (open === undefined) ? !panel.classList.contains("open") : open;
  panel.classList.toggle("open", want);
  document.body.classList.toggle("notes-open", want);   // shifts the text over
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") {                 // Esc leaves a note field
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.key === "m" || e.key === "M") markWord(currentWord());
  if (e.key === "n" || e.key === "N") showNotes();
  if (e.key === "Escape") showNotes(false);
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
// macOS ships several voice tiers. The default is the flat robotic one; the
// Siri / Premium / Enhanced voices are markedly more human. Pick the best
// available once, and let the reader change it.
function bestVoice(){
  const vs = speechSynthesis.getVoices().filter(v => v.lang.startsWith("en"));
  if (!vs.length) return null;
  const saved = localStorage.getItem("voice");
  if (saved) { const hit = vs.find(v => v.name === saved); if (hit) return hit; }
  const rank = v => (/siri/i.test(v.name) ? 0 :
                     /premium|enhanced|natural|neural/i.test(v.name) ? 1 :
                     /samantha|ava|allison|serena|zoe|evan|tom/i.test(v.name) ? 2 : 3);
  return vs.sort((a, b) => rank(a) - rank(b))[0];
}
function fillVoiceMenu(){
  const sel = document.getElementById("voice");
  const vs = speechSynthesis.getVoices().filter(v => v.lang.startsWith("en"));
  if (!vs.length || sel.options.length) return;
  const best = bestVoice();
  for (const v of vs){
    const o = document.createElement("option");
    o.value = v.name; o.textContent = v.name.replace(/ \(.*\)/, "");
    if (best && v.name === best.name) o.selected = true;
    sel.appendChild(o);
  }
}
speechSynthesis.onvoiceschanged = fillVoiceMenu;

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
  const v = bestVoice(); if (v) u.voice = v;
  u.rate = tts.wpm / 180;                        // rate 1.0 ~ 180 wpm
  u.pitch = 1.0;
  u.volume = 1.0;
  u.onboundary = (e) => {
    if (e.name && e.name !== "word") return;
    let k = starts.findIndex(st => st > e.charIndex) - 1;
    if (k < -1+1 && starts[starts.length-1] <= e.charIndex) k = starts.length-1;
    if (k >= 0) { clearHi(); spans[k].classList.add("speaking");
                  spans[k].scrollIntoView({block:"center", behavior:"smooth"}); }
  };
  // a short breath between paragraphs: continuous speech with no pauses is a
  // big part of what makes synthetic reading feel relentless
  u.onend = () => { if (tts.on) setTimeout(() => speakPar(i + 1), 450); };
  speechSynthesis.speak(u);
}
"""


def build_page(text_path, out_dir, wpm=135, model=None, profile="comfort"):
    """text/markdown file -> adaptive html page. Returns the output path.
    wpm: the reader's own measured pace (recorder computes it from their
    best-calibrated sessions); becomes the read-along default speed."""
    global _MODEL
    _MODEL = model                       # used by hardness() for every word below
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
            h = hardness(w, model)
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
                 f"<style>{CSS}</style><script>const DEFAULT_WPM={int(wpm)};"
                 f"const DEFAULT_PROFILE={profile!r};{JS}</script></head>"
                 f"<body class='{profile}'>"
                 "<div id='bar'><span id='brand'>&#128065; Adaptive Reading</span>"
                 "<b style='font:14px -apple-system'>mode:</b>"
                 "<button data-p='comfort'>Comfort</button>"
                 "<button data-p='focus'>Focus</button>"
                 "<button data-p='skim'>Skim</button>"
                 "<div id='rbar'><button id='slower'>&minus;</button>"
                 "<span id='wpm'></span><button id='faster'>+</button>"
                 "<button id='marks' title='marked passages'>&#9998; notes</button>"
                 "<select id='voice' title='voice'></select>"
                 "<button id='play'>&#9654; read along</button></div></div>"
                 f"<div id='pstate'>{_state_line(model)}</div>"
                 "<div id='hint'><span>This page adapts to you: press <b>M</b> to "
                 "mark what you are reading, <b>N</b> for your notes &middot; "
                 "<span class='h3'>marked words</span> are ones readers usually find "
                 "hard &middot; pick a mode above &middot; &#9654; reads along at your "
                 "own measured pace.</span><button id='hintx' title='got it'>&#10005;"
                 "</button></div>"
                 f"<div id='text'>{''.join(body)}</div>"
                 "<div id='notes'><button id='noteclose' title='close'>&times;</button>"
                 "<h4>Marked while reading</h4>"
                 "<div id='notelist'></div>"
                 "<p style='color:#888;font-size:12px'>M marks the passage you are on "
                 "&middot; N or Esc closes this panel</p></div></body></html>")
    return out
