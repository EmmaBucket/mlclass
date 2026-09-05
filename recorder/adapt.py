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


def _tokenize(par):
    """Split a paragraph into (word, style) pairs, where style comes from the
    ORIGINAL document: **bold**, *italic*, `code`. Word-level granularity keeps
    every word individually gaze-trackable while still looking like the source."""
    out = []
    for chunk in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)", par):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            style, text = "b", chunk[2:-2]
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            style, text = "i", chunk[1:-1]
        elif chunk.startswith("`") and chunk.endswith("`"):
            style, text = "codeword", chunk[1:-1]
        else:
            style, text = "", chunk
        out += [(w, style) for w in text.split()]
    return out


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
#next { display:block; margin:40px auto 80px; max-width:34rem; padding:16px 20px;
        background:var(--accent); color:#fff; border-radius:12px; text-decoration:none;
        font:600 17px -apple-system,sans-serif; text-align:center; }
#next small { display:block; font-weight:400; opacity:.85; margin-top:3px; font-size:13px; }
#next:hover { filter:brightness(1.08); }
pre.code { background:#f4f1ec; border:1px solid #e2ddd4; border-left:3px solid var(--accent);
           border-radius:6px; padding:10px 14px; overflow-x:auto; margin:14px 0; }
pre.code code { font:14px/1.55 "SF Mono",Menlo,Consolas,monospace; white-space:pre;
                letter-spacing:normal !important; word-spacing:normal !important; }
pre.code .cl { display:block; }
/* code keeps its own look in every reading mode */
body.skim pre.code .cl, body.comfort pre.code .cl, body.focus pre.code .cl { opacity:1; }
.math { font-family:"SF Mono",Menlo,monospace; letter-spacing:normal !important;
        background:#f6f4ef; padding:0 2px; border-radius:3px; }
#nextbar { margin-left:8px; }
#profile { font:13px -apple-system,sans-serif; text-decoration:none; color:var(--accent);
           border:1px solid #d6c9e6; border-radius:16px; padding:7px 12px; }
#profile:hover { background:#efe8f7; }
/* focus spotlight: everything except the passage you are on recedes.
   Used manually (F) and automatically when attention drops. */
body.focusing #text p, body.focusing #text pre { opacity:.28; transition:opacity .5s; }
body.focusing #text p.here, body.focusing #text pre.here { opacity:1; }
#text p.here { box-shadow:-14px 0 0 -11px var(--accent); }
/* where you left off when you looked away to write */
.resume { animation:resumeflash 2.4s ease-out 1; }
@keyframes resumeflash { 0% { background:#ffe9a8; } 100% { background:transparent; } }
#rail { position:fixed; left:0; top:0; height:3px; background:var(--accent);
        width:0; z-index:30; transition:width .3s; }
#left { position:fixed; right:14px; bottom:12px; font:12px -apple-system,sans-serif;
        color:#7a6a8c; background:#fffffff0; padding:5px 10px; border-radius:12px;
        border:1px solid #e6dcf0; z-index:12; }
.recall { margin:26px auto; max-width:34rem; background:#fff; border:1px solid #e0d3ef;
          border-left:4px solid var(--accent); border-radius:10px; padding:14px 16px;
          font:15px/1.5 -apple-system,sans-serif; }
.recall b { display:block; margin-bottom:6px; }
.recall textarea { width:100%; height:52px; border:1px solid #ccc; border-radius:6px;
                   padding:6px; font:14px -apple-system,sans-serif; }
.recall .done { color:#2e7d32; font-size:13px; }
.nudge { position:fixed; left:50%; transform:translateX(-50%); bottom:26px; z-index:40;
         background:#4b3b60; color:#fff; padding:10px 16px; border-radius:20px;
         font:14px -apple-system,sans-serif; box-shadow:0 4px 16px rgba(0,0,0,.2); }
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
#notes .note { border-bottom:1px solid #eee; padding:8px 24px 8px 0; position:relative; }
#notes .note b { cursor:pointer; }
#notes .del { position:absolute; right:0; top:8px; border:none; background:none;
              font-size:17px; color:#b03030; cursor:pointer; opacity:.45; line-height:1; }
#notes .del:hover { opacity:1; }
#focusbtn.on { background:var(--accent); color:#fff; border-color:var(--accent); }
::selection { background:#cfe8ff; }
#notes .note b { display:block; color:#555; font-weight:600; }
#notes .tags { margin:6px 0 4px; display:flex; gap:4px; flex-wrap:wrap; }
#notes .tags button { border:1px solid #ccc; background:#fff; border-radius:12px;
                      font-size:11px; padding:3px 8px; cursor:pointer; }
#notes .tags button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
#notes .filter { margin-bottom:10px; font-size:12px; color:#666; }
#notes .filter select { font-size:12px; padding:3px; }
.marked.tag-question { background:#ffe0e6; box-shadow:0 1px 0 #e06f8b; }
.marked.tag-definition { background:#e2e0ff; box-shadow:0 1px 0 #7b76d6; }
.marked.tag-important { background:#ffeab0; box-shadow:0 1px 0 #d9a520; }
.marked.tag-todo { background:#d8f0d8; box-shadow:0 1px 0 #5aa75a; }
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
body.comfort .b { background:#fff3cd; padding:0 2px; border-radius:3px; }

/* FOCUS: the deep-reading block. Uniform, calm, minimal signalling -- dense on
   purpose. Difficulty marks are switched OFF here; only the author's own
   emphasis survives, so nothing competes with the argument. */
body.focus #text { max-width:44rem; font:19px/1.65 Georgia,serif; }
body.focus p { margin:0 0 1.1em; }
body.focus .h2, body.focus .h3 { letter-spacing:normal; font-weight:inherit;
                                 background:none; }

/* SKIM: the page's SKELETON. Topic sentence of each paragraph stays full
   strength, the rest recedes; headings and the author's own bold stay loud.
   (This used to bold the first two words of every paragraph, which cut
   sentences mid-phrase and read as random highlighting.) */
body.skim #text { max-width:38rem; font:19px/1.75 -apple-system,sans-serif; }
body.skim p { margin:0 0 1.3em; }
body.skim #text p span[data-w] { opacity:.4; }
body.skim #text p span.lead { opacity:1; font-weight:500; }     /* topic sentence */
body.skim #text p span.b { opacity:1; }                          /* author's own bold */
body.skim #text h2 span, body.skim #text h3 span { opacity:1; }
body.skim #text pre.code span { opacity:.75; }

/* the document's OWN emphasis, in every mode */
.b { font-weight:700; }
.i { font-style:italic; }
.codeword { font-family:"SF Mono",Menlo,monospace; font-size:.92em; background:#f2efe9;
            padding:0 3px; border-radius:3px; letter-spacing:normal !important; }
#text h2 { font-size:1.35em; margin:1.6em 0 .5em; line-height:1.3; }
#text h3 { font-size:1.12em; margin:1.3em 0 .4em; line-height:1.35; }
#text h2 span[data-w], #text h3 span[data-w] { letter-spacing:normal; }
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
  document.getElementById("vtest").onclick = () => {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(
      "Reading should feel easy. This is how this voice sounds at your pace.");
    const v = bestVoice(); if (v) u.voice = v;
    u.rate = tts.wpm / 180; speechSynthesis.speak(u);
  };
  document.getElementById("voice").onchange = (e) => {
    localStorage.setItem("voice", e.target.value);
    if (tts.on) { speechSynthesis.cancel(); speakPar(tts.par); }
  };
  document.getElementById("slower").onclick = () => bumpWpm(-15);
  document.getElementById("faster").onclick = () => bumpWpm(+15);
  setProfile(window.__profile);
  updateTTSUI();
  if (sessionStorage.getItem("autoplay")) {      // arrived here mid-read-along
    sessionStorage.removeItem("autoplay");
    setTimeout(() => startFrom(0), 400);
  }
  document.getElementById("focusbtn").onclick = () => toggleFocus();
  document.getElementById("marks").onclick = () => showNotes();
  document.getElementById("noteclose").onclick = () => showNotes(false);
  for (const sp of document.querySelectorAll("#text span[data-w]"))
    sp.onclick = () => markWord(sp);
  restoreMarks();
  setupRecall();
  updateProgress();
  markHere();
  if (localStorage.getItem("focusing")) setFocusing(true);
  const hint = document.getElementById("hint");
  if (localStorage.getItem("hint_seen")) hint.style.display = "none";
  document.getElementById("hintx").onclick = () => {
    hint.style.display = "none"; localStorage.setItem("hint_seen", "1");
  };
});

// ---------------- attention, progress, recall ----------------
// Long technical reading fails in a specific way: you keep looking at the page
// while your attention has left. Three counters against that, in increasing
// order of intrusiveness:
//   1. visible progress (how far, how much left, in YOUR minutes)
//   2. a focus spotlight that quiets everything except the passage you are on
//   3. a short recall prompt at section boundaries -- retrieval, not re-reading,
//      is what actually keeps attention and memory alive
function currentPar(){
  const mid = window.innerHeight * 0.42;
  let best = null, bestD = 1e9;
  for (const p of document.querySelectorAll("#text p, #text pre")){
    const r = p.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) continue;
    const d = Math.abs(r.top + Math.min(r.height, 200) / 2 - mid);
    if (d < bestD){ bestD = d; best = p; }
  }
  return best;
}
function markHere(){
  const p = currentPar();
  for (const el of document.querySelectorAll(".here")) el.classList.remove("here");
  if (p) p.classList.add("here");
}
function updateProgress(){
  const doc = document.documentElement;
  const done = Math.min(1, (window.scrollY + window.innerHeight) / doc.scrollHeight);
  document.getElementById("rail").style.width = (done * 100).toFixed(1) + "%";
  const words = document.querySelectorAll("#text p span[data-w]").length;
  const leftWords = Math.max(0, Math.round(words * (1 - done)));
  const mins = leftWords / Math.max(60, tts.wpm);
  document.getElementById("left").textContent =
    leftWords < 40 ? "nearly done" :
    (mins < 1 ? "under a minute left" : `about ${Math.ceil(mins)} min left at your pace`);
}
function setFocusing(on){
  document.body.classList.toggle("focusing", on);
  markHere();
  localStorage.setItem("focusing", on ? "1" : "");
  const b = document.getElementById("focusbtn");
  if (b){ b.classList.toggle("on", on); b.textContent = on ? "focus: on" : "focus"; }
}
// the ONLY way focus changes, from any of: F, Esc, the toolbar button, the
// nudge's exit link, or the eye tracker
function toggleFocus(force){
  const on = (force === undefined) ? !document.body.classList.contains("focusing") : force;
  setFocusing(on);
  if (!on){
    // a manual exit is a decision: do not let the tracker override it again
    localStorage.setItem("autofocus_off", "1");
    nudge("Focus mode off — it will stay off");
  }
}
// the recorder calls this from the eye tracker: "low" when the reader's gaze has
// been off the text for a while, "ok" when it comes back.
// The recorder calls this when the reader physically looks away (taking notes
// on paper, for instance) and when they come back. Away is not distraction:
// nothing gets dimmed, the voice simply waits for them.
window.setPresence = function(state){
  window.__presence = state;
  if (state === "away"){
    if (tts.on){ speechSynthesis.pause(); window.__ttsPaused = true; }
    nudge("Paused — take your time with your notes");
  } else if (state === "back"){
    if (window.__ttsPaused){ speechSynthesis.resume(); window.__ttsPaused = false; }
    const here = document.querySelector(".here") || document.querySelector(".speaking");
    if (here){
      here.classList.add("resume");
      setTimeout(() => here.classList.remove("resume"), 2600);
      nudge("Welcome back — you left off here");
    }
  }
};
window.setAttention = function(level){
  window.__attention = level;
  // THE BUG THIS FIXES: this used to switch focus mode on for every dip in
  // attention without checking whether the reader had already turned it off,
  // so pressing F appeared to do nothing -- the page grabbed control back a few
  // seconds later, with the text greyed out and no way out.
  if (localStorage.getItem("autofocus_off") === "1") return;
  if (level === "low" && !document.body.classList.contains("focusing")){
    setFocusing(true);
    nudge("Focus mode on — press F or Esc to leave it");
  }
};
let nudgeTimer = null;
function nudge(msg){
  let el = document.querySelector(".nudge");
  if (!el){ el = document.createElement("div"); el.className = "nudge"; document.body.appendChild(el); }
  el.textContent = msg;
  clearTimeout(nudgeTimer);
  nudgeTimer = setTimeout(() => el.remove(), 3200);
}
function setupRecall(){
  // one prompt per section boundary, about every 700 words of prose
  const pars = [...document.querySelectorAll("#text p")];
  let acc = 0, n = 0;
  for (const p of pars){
    acc += p.querySelectorAll("span[data-w]").length;
    if (acc >= 700){
      acc = 0; n += 1;
      const card = document.createElement("div");
      card.className = "recall";
      card.innerHTML = "<b>Quick check &mdash; in one sentence, what was that section about?</b>";
      const ta = document.createElement("textarea");
      ta.placeholder = "type it from memory, do not scroll back...";
      const done = document.createElement("div");
      ta.onchange = () => {
        if (!ta.value.trim()) return;
        recalls.push({at: n, text: ta.value.trim()});
        window.__recall = JSON.stringify(recalls);
        done.className = "done"; done.textContent = "saved with your notes \u2713";
      };
      card.appendChild(ta); card.appendChild(done);
      p.after(card);
    }
  }
}
const recalls = [];
document.addEventListener("scroll", () => { updateProgress(); markHere(); }, {passive: true});
document.addEventListener("mousemove", markHere, {passive: true});
setInterval(markHere, 1200);   // and while the voice reads and nothing else moves

// ---------------- highlighting + notes ----------------
// One highlight = one note. Select a sentence with the mouse (or press M to take
// the sentence you are on) and it becomes a single entry you can tag, annotate
// and delete. Marks are stored as {id: {ws:[word ids], text, note, tag}} so a
// highlight spanning twenty words is still ONE note, not twenty.
const TAGS = ["question", "definition", "important", "todo"];
let marks = JSON.parse(localStorage.getItem("marks") || "{}");
// migrate the old one-word-per-note format
for (const id of Object.keys(marks)) if (!marks[id].ws) marks[id].ws = [+id];

function spanOf(id){ return document.querySelector(`#text span[data-w="${id}"]`); }
function saveMarks(){
  localStorage.setItem("marks", JSON.stringify(marks));
  window.__marks = JSON.stringify(Object.keys(marks).map(k => ({
    w: +k, ws: marks[k].ws, note: marks[k].note || "",
    text: marks[k].text, tag: marks[k].tag || null })));
  renderNotes();
}
function paintMark(id){
  const m = marks[id];
  if (!m) return;
  for (const w of m.ws){
    const sp = spanOf(w);
    if (!sp) continue;
    sp.classList.add("marked");
    for (const t of TAGS) sp.classList.remove("tag-" + t);
    if (m.tag) sp.classList.add("tag-" + m.tag);
    sp.classList.toggle("hasnote", !!(m.note || "").trim());
    sp.dataset.mark = id;
  }
}
function removeMark(id){
  for (const w of (marks[id] || {}).ws || []){
    const sp = spanOf(w);
    if (!sp) continue;
    sp.classList.remove("marked", "hasnote", ...TAGS.map(t => "tag-" + t));
    delete sp.dataset.mark;
  }
  delete marks[id];
  saveMarks();
}
function addMark(spans){
  spans = spans.filter(Boolean);
  if (!spans.length) return;
  const ws = spans.map(sp => +sp.dataset.w).sort((x, y) => x - y);
  // if this overlaps an existing highlight, replace it rather than stacking
  for (const id of Object.keys(marks))
    if (marks[id].ws.some(w => ws.includes(w))) removeMark(id);
  const id = ws[0];
  marks[id] = {ws, text: spans.map(sp => sp.innerText).join(" ").trim(), note: "", tag: null};
  paintMark(id);
  saveMarks();
  return id;
}
// the sentence around a word, inside its own paragraph
function sentenceSpans(sp){
  if (!sp) return [];
  const sibs = [...sp.parentElement.querySelectorAll("span[data-w]")];
  let i = sibs.indexOf(sp);
  if (i < 0) return [sp];
  let start = i, end = i;
  while (start > 0 && !/[.!?]["')]?$/.test(sibs[start - 1].innerText)) start--;
  while (end < sibs.length - 1 && !/[.!?]["')]?$/.test(sibs[end].innerText)) end++;
  return sibs.slice(start, end + 1);
}
// what the mouse selected, snapped to whole words
function selectedSpans(){
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return [];
  const range = sel.getRangeAt(0);
  const host = document.getElementById("text");
  if (!host.contains(range.commonAncestorContainer)) return [];
  return [...host.querySelectorAll("span[data-w]")]
    .filter(sp => range.intersectsNode(sp));
}
document.addEventListener("mouseup", () => {
  const spans = selectedSpans();
  if (spans.length){
    const id = addMark(spans);
    window.getSelection().removeAllRanges();
    showNotes(true);
    const el = document.querySelector(`#notelist [data-note="${id}"] textarea`);
    if (el) el.focus();
  }
});

function renderNotes(){
  const list = document.getElementById("notelist");
  if (!list) return;
  const filter = (document.getElementById("tagfilter") || {}).value || "all";
  const ids = Object.keys(marks).map(Number).sort((x, y) => x - y)
                    .filter(id => filter === "all" || marks[id].tag === filter);
  list.innerHTML = "";
  if (!ids.length){
    list.innerHTML = "<i>Nothing highlighted yet.<br>Select a sentence with the mouse, "
                   + "or press M to take the sentence you are reading.</i>";
    return;
  }
  for (const id of ids){
    const m = marks[id];
    const d = document.createElement("div");
    d.className = "note";
    d.dataset.note = id;

    const del = document.createElement("button");
    del.className = "del"; del.title = "delete this highlight"; del.innerHTML = "&times;";
    del.onclick = (ev) => { ev.stopPropagation(); removeMark(id); };
    d.appendChild(del);

    const quote = document.createElement("b");
    quote.textContent = "“" + m.text + "”";
    quote.onclick = () => { const sp = spanOf(m.ws[0]);
                            if (sp) sp.scrollIntoView({block: "center", behavior: "smooth"}); };
    d.appendChild(quote);

    const tagbar = document.createElement("div");
    tagbar.className = "tags";
    for (const t of TAGS){
      const b = document.createElement("button");
      b.textContent = t;
      if (m.tag === t) b.classList.add("on");
      b.onclick = (ev) => { ev.stopPropagation();
                            m.tag = (m.tag === t) ? null : t; paintMark(id); saveMarks(); };
      tagbar.appendChild(b);
    }
    d.appendChild(tagbar);

    const ta = document.createElement("textarea");
    ta.value = m.note || ""; ta.placeholder = "your note...";
    ta.onchange = () => { m.note = ta.value; paintMark(id); saveMarks(); };
    d.appendChild(ta);
    list.appendChild(d);
  }
}
function restoreMarks(){
  for (const id of Object.keys(marks)) paintMark(id);
  const f = document.getElementById("tagfilter");
  if (f) f.onchange = renderNotes;
  renderNotes(); saveMarks();
}
// M takes the sentence you are on: the spoken one if the voice is reading,
// otherwise the one nearest the middle of the screen.
function currentWord(){
  const spoken = document.querySelector(".speaking");
  if (spoken) return spoken;
  const mid = window.innerHeight / 2;
  let best = null, bestD = 1e9;
  for (const sp of document.querySelectorAll("#text p span[data-w]")){
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
  document.body.classList.toggle("notes-open", want);
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA"){
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.key === "m" || e.key === "M"){
    const id = addMark(sentenceSpans(currentWord()));
    if (id !== undefined) nudge("Sentence highlighted — press N to write a note");
  }
  if (e.key === "n" || e.key === "N") showNotes();
  if (e.key === "f" || e.key === "F") toggleFocus();
  if (e.key === "Escape"){
    if (document.body.classList.contains("focusing")) toggleFocus(false);
    else showNotes(false);
  }
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

function goNext(){
  const a = document.getElementById("next");
  if (!a) return false;
  // carry the read-along state over the boundary: if the voice was reading,
  // the next chapter starts reading itself
  if (tts.on) sessionStorage.setItem("autoplay", "1");
  location.href = a.getAttribute("href");
  return true;
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight" && (e.metaKey || e.altKey)) goNext();
});

function speakPar(i){
  const pars = paragraphs();
  if (!tts.on || i >= pars.length) {
    // finished the chapter while reading aloud -> roll straight into the next
    if (tts.on && document.getElementById("next")) { goNext(); return; }
    stopTTS(); return;
  }
  tts.par = i;
  const spans = [...pars[i].querySelectorAll("span[data-w]")];
  // Speak one SENTENCE at a time. A whole paragraph in one utterance makes the
  // synthesiser run out of breath and flatten its intonation; sentence-sized
  // chunks let it shape each one, and the gaps land where a human would pause.
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
                  if (document.body.classList.contains("focusing")){
                    for (const el of document.querySelectorAll(".here")) el.classList.remove("here");
                    pars[i].classList.add("here");     // spotlight follows the voice
                  }
                  spans[k].scrollIntoView({block:"center", behavior:"smooth"}); }
  };
  // a short breath between paragraphs: continuous speech with no pauses is a
  // big part of what makes synthetic reading feel relentless
  // pause length follows the punctuation the paragraph ends on
  const tail = text.trim().slice(-1);
  const gap = tail === "." || tail === "!" || tail === "?" ? 480 :
              tail === ":" || tail === ";" ? 340 : 260;
  u.onend = () => { if (tts.on) setTimeout(() => speakPar(i + 1), gap); };
  speechSynthesis.speak(u);
}
"""


def build_page(text_path, out_dir, wpm=135, model=None, profile="comfort",
               next_href=None, next_title=None):
    """text/markdown file -> adaptive html page. Returns the output path.
    wpm: the reader's own measured pace (recorder computes it from their
    best-calibrated sessions); becomes the read-along default speed."""
    global _MODEL
    _MODEL = model                       # used by hardness() for every word below
    raw = open(text_path, encoding="utf-8", errors="replace").read()
    # split on blank lines, but keep ```fenced``` blocks whole
    parts, buf, in_code = [], [], False
    for line in raw.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                buf.append(line); parts.append("\n".join(buf)); buf, in_code = [], False
            else:
                if buf: parts.append("\n".join(buf))
                buf, in_code = [line], True
            continue
        if in_code:
            buf.append(line)
        elif line.strip():
            buf.append(line)
        else:
            if buf: parts.append("\n".join(buf))
            buf = []
    if buf: parts.append("\n".join(buf))
    paragraphs = [p for p in parts if p.strip()]

    # ADAPTIVE THRESHOLD: mark the hardest ~12% of THIS document's words rather
    # than everything over a fixed score. As the personal model learns, which
    # words fall in that 12% changes -- but the amount of marking stays steady,
    # so a dense chapter is not a sea of highlights and an easy one is not bare.
    prose_words = []
    for par in paragraphs:
        if par.lstrip().startswith("```"):
            continue
        prose_words += [w for w in re.sub(r"[*`#]", "", par).split()]
    scores = sorted(hardness(w, model) for w in prose_words if len(w) > 2) or [0.6, 0.75]
    cut_hard = scores[int(0.88 * (len(scores) - 1))]
    cut_very = scores[int(0.96 * (len(scores) - 1))]

    widx = 0
    body = []
    for par in paragraphs:
        # CODE AND OUTPUT: never restyled, never re-spaced, never read aloud.
        # Code is not prose -- letter-spacing and word-wrapping destroy its
        # meaning, and "hard word" styling on a variable name is nonsense.
        # Each LINE gets one span so gaze still maps to it.
        if par.lstrip().startswith("```"):
            lines = [l for l in par.split("\n") if not l.strip().startswith("```")]
            rendered = []
            for line in lines:
                rendered.append(f'<span data-w="{widx}" class="cl">{html.escape(line)}</span>')
                widx += 1
            body.append("<pre class='code'><code>" + "\n".join(rendered) + "</code></pre>")
            continue
        # (this used to strip # and ** as "noise" -- that was the bug that made
        # the adaptive page lose every heading and every bolded term)
        # headings keep their level; the author's own hierarchy IS the emphasis
        tag, heading = "p", 0
        m = re.match(r"^(#{1,3})\s+", par)
        if m:
            heading = len(m.group(1))
            tag = {1: "h2", 2: "h2", 3: "h3"}[heading]
            par = par[m.end():]

        words_html = []
        first_sentence = True          # skim mode leans on topic sentences
        par = re.sub(r"(?<!\*)\*(?!\*)\s*$", "", par)      # dangling emphasis markers
        for j, (w, style) in enumerate(_tokenize(par)):
            # math stays exactly as written: \(x^2\), $\alpha$, 3.14e-8
            if re.match(r"^(\\\(|\\\[|\$|\\begin)", w) or re.search(r"[=^_{}\\]", w):
                words_html.append(f'<span data-w="{widx}" class="math">{html.escape(w)}</span>')
                widx += 1
                continue
            w = w.strip("*`")                       # unbalanced marker, never show it
            if not w:
                continue
            cls = []
            if style: cls.append(style)             # b / i / codeword from the source
            if first_sentence and not heading: cls.append("lead")
            if not style and not heading:
                h = hardness(w, model)
                if h >= cut_very: cls.append("h3")
                elif h >= cut_hard: cls.append("h2")
            words_html.append(
                f'<span data-w="{widx}"' + (f' class="{" ".join(cls)}"' if cls else "")
                + f'>{html.escape(w)}</span>')
            widx += 1
            if re.search(r"[.!?][\"\')]?$", w):
                first_sentence = False
        speak = ("" if heading else
                 f'<button class="speak" data-par="{len(body)}" '
                 'title="read aloud from here">&#128264;</button>')
        body.append(f"<{tag}>" + speak + " ".join(words_html) + f"</{tag}>")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(text_path) + ".adaptive.html")
    nav = ""
    if next_href:
        label = html.escape(next_title or "Next chapter")
        nav = (f"<a id='next' href='{html.escape(next_href)}'>Next chapter &rarr;"
               f"<small>{label}</small></a>")
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
                 "<a id='profile' href='../progress.html' title='your reading profile'>"
                 "&#128100; my reading</a>"
                 "<button id='focusbtn' title='dim everything except the passage "
                 "you are reading (F)'>focus</button>"
                 "<button id='marks' title='highlights and notes (N)'>&#9998; notes</button>"
                 "<select id='voice' title='voice'></select>"
                 "<button id='vtest' title='hear this voice'>&#9835;</button>"
                 "<button id='play'>&#9654; read along</button></div></div>"
                 f"<div id='pstate'>{_state_line(model)}</div>"
                 "<div id='hint'><span>This page adapts to you: "
                 "<b>select any sentence</b> to highlight and note it "
                 "(or <b>M</b> for the sentence you are on) &middot; <b>N</b> opens your "
                 "notes &middot; <b>F</b> or <b>Esc</b> toggles focus &middot; "
                 "<span class='h3'>marked words</span> are ones readers usually find "
                 "hard &middot; pick a mode above &middot; &#9654; reads along at your "
                 "own measured pace.</span><button id='hintx' title='got it'>&#10005;"
                 "</button></div>"
                 "<div id='rail'></div><div id='left'></div>"
                 f"<div id='text'>{''.join(body)}{nav}</div>"
                 "<div id='notes'><button id='noteclose' title='close'>&times;</button>"
                 "<h4>Marked while reading</h4>"
                 "<div class='filter'>show: <select id='tagfilter'>"
                 "<option value='all'>all</option><option>question</option>"
                 "<option>definition</option><option>important</option>"
                 "<option>todo</option></select></div>"
                 "<div id='notelist'></div>"
                 "<p style='color:#888;font-size:12px'>M marks the passage you are on "
                 "&middot; N or Esc closes this panel</p></div></body></html>")
    return out
