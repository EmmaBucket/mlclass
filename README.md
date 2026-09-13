# Adaptive Reading

**An eye-tracking reading tool that adapts text to the person reading it — built by a dyslexic reader, for readers who find long technical text hard.**

Point a webcam at yourself, open a textbook chapter, and the page rebuilds itself: cleaner typography, the hard words marked, a voice that reads along at *your* measured pace, highlights that become a study sheet, and a layout that quietly learns which words slow *you* down. Everything runs locally; nothing leaves your machine.

This started as a machine-learning course project (Math 408, CSU Channel Islands) and became a working tool I use for my own reading.

Repository: https://github.com/EmmaBucket/mlclass

---

## What it does

| While you read | What happens |
|---|---|
| **Adaptive page** | Any web chapter or local text file is rebuilt with dyslexia-informed typography: measured line lengths (62 characters, not 38 by accident — see *lessons*), generous spacing, the author's own headings and bold preserved, code blocks and equations rendered properly (`\beta_0` → β₀, spoken as "beta zero"). |
| **Three reading modes** | *Comfort* (spacing + difficulty marks), *Focus* (calm, dense, no marks), *Skim* (topic sentences stay sharp, the rest recedes). Your choice is remembered. |
| **Text controls** | Size, line spacing, line length, word and letter spacing — per mode, saved, restored, with a one-click reset. Light and dark themes with computed WCAG contrast (11:1 body text in both). |
| **Read-along** | The browser's own speech engine reads at the pace *your* eyes were measured to read, with karaoke highlighting, resume-where-you-stopped, and per-paragraph start buttons. Pauses when you look down at your notes; resumes when you look back. |
| **Highlight → note → study sheet** | Drag to highlight a sentence, tag it (question / definition / important / to-do), write a note whenever. At the end you get Markdown and a printable PDF that separates *what the text said* from *what you thought*. |
| **Attention-aware focus** | If your gaze drifts off the text for a sustained stretch (relative to *your own* baseline that session), the page dims everything but the passage you're on. One key turns it off, and it stays off. |
| **Chapters flow** | The next chapters are pre-built; the voice rolls straight into the next one. |
| **Your reading profile** | Every session feeds a personal page: sessions, tired-vs-rested comparisons, which mode holds your attention, what you highlighted, and *exactly* what the system has learned about you and what it deliberately hasn't. |

Camera optional: a **just read** mode gives you the page, voice, and notes with no tracking at all.

## How it works

```
web page / text file
        │  extract prose, headings, emphasis, code, figures, TeX
        ▼
   adapt.py ──► self-contained HTML page (modes, controls, voice, notes, focus)
        │                        ▲
        │ opens in Chrome        │ per-reader difficulty model + preferences
        ▼                        │
   record.py ──► webcam ► MediaPipe iris ► 9-point calibration ► gaze → word
        │  every frame: gaze, blink, lean-in, frown, presence, scroll, mode, notes
        ▼
   reading.db (SQLite, local) ──► personalize.py (refit after every session)
                               ──► progress.py  (your reading profile)
                               ──► notes_pdf.py (study sheets)
```

**The difficulty model.** Trained first on the [GECO corpus](https://expsy.ugent.be/downloads/geco/) (Ghent Eye-Tracking Corpus: 19 readers, 534,000 words, lab eye-tracker). A word's predicted difficulty comes from its length and corpus frequency — the two effects that replicate across every reader, including me. After enough well-calibrated sessions, the model is refit on *your* dwell data and blended with the population model using shrinkage (`weight = n / (n + 3000)`), so it slides from "general reader" to "you" as evidence accumulates, and never jumps.

**The attention rule.** Compares the last ~10 s of gaze-on-text to the last ~100 s of the *same session*. An absolute threshold was tried first and flagged 83–100 % of every session as distracted — on-text rate is only ~35 % even while reading normally, because words are small and webcam gaze is coarse. Thresholds on physiological signals must be relative to the person and the moment.

## Lessons this project learned the hard way

These are the findings I'm proudest of, because each one was a confident-looking number that turned out to be wrong.

1. **The first model was cheating.** My original "reading breakdown" label was defined as *reading time > 2 SD*, and reading time was also an input feature. AUC 0.97 — and a one-line `if` statement reproduced the label 100 % of the time. A model that scored 0.999 on just the two columns the label was built from proved it. The honest replacement — predict dwell from **text alone**, evaluated on readers the model has never seen — scores **AUC 0.75–0.80**. A real 0.75 is worth more than a fake 0.97.

2. **Which words are hard is predictable; whether *you* are struggling right now is not.** Text features predict dwell at AUC 0.80. "Harder for this reader than the word usually is" sits at AUC 0.50 — chance — from text, from the crowd, and from recent behaviour. It *persists* across paragraphs (r = 0.25), so it can be measured and reacted to, but not forecast. That split is the architecture: **predict the text, measure the reader.**

3. **Personalising *which* words are hard barely helps; personalising *when to intervene* does.** Training on a reader's own first session vs. four strangers changed AUC from 0.679 to 0.685. Hard words are hard for everyone. What is personal is the threshold.

4. **A webcam resolves about a 21 × 11 grid.** Crossing a 13-inch screen moves the iris ~5 % of the eye's width; divided by jitter, that's ~70 × 80 px cells. Word-level gaze is physically out of reach on a laptop, line-level on a monitor. So the design deliberately does **not** run automatic typography experiments: the instrument cannot measure what a font change moves, and the natural "reading speed" reward is coupled to layout in a way that would push toward smaller text and longer lines. Instead, every setting you choose is kept, logged, and shown back to you.

5. **Comfort mode had been showing 38 characters per line for weeks.** `max-width: 34rem` measures against the page's base font, not the reading font. Line length is now set in `ch` and the defaults were measured on rendered lines.

6. **Bugs that don't crash are the dangerous ones.** A scrolling container the recorder couldn't see silently attributed every gaze to the wrong word for a whole session; a reused calibration from a different screen carried a "good" error number; a mouse-up handler turned every stray click into a permanent highlight. All three produced clean-looking data. The fix each time was a check that compares the data to something it must agree with — and now those checks run automatically.

## Evidence behind the design choices

- **Read-aloud support** has the strongest evidence of anything in the app (Wood et al., 2018, *g* ≈ 0.3–0.4). The eye-tracking is the weakest link. The design leans on the former.
- **Increased letter and word spacing** (Zorzi et al., 2012, PNAS) is the one typographic intervention with replicated gains. It's on by default in Comfort.
- **Dyslexia fonts** (OpenDyslexic, Dyslexie) show no benefit in three independent studies (Rello & Baeza-Yates 2013; Wery & Diliberto 2017; Kuster et al. 2018) — so they are not the default. The stack tries Atkinson Hyperlegible, falls back to Verdana.
- **"Bionic" bolding of word beginnings** has no measured speed benefit and slightly worse comprehension; an early Skim mode did exactly this and was replaced with topic-sentence emphasis.
- **Coloured overlays** for "scotopic sensitivity" are not supported by the ophthalmology literature; a warm off-white background is offered as a comfort preference, not a treatment.
- **Retrieval practice** beats re-reading for both attention and retention; the page inserts a one-sentence recall prompt at section breaks, and your answers double as the comprehension ground truth that eye data cannot provide.

## Setup (macOS)

```bash
# 1. environment (Python 3.10 in conda; the scripts re-launch themselves here automatically)
conda create -n mlclass python=3.10
conda activate mlclass
pip install -r requirements.txt

# 2. a natural voice (free): System Settings → Accessibility → Read & Speak →
#    System voice → Manage Voices… → download "Ava (Premium)" or "Zoe (Premium)"

# 3. Google Chrome installed (the page runs in it; the driver is fetched automatically)
```

The camera, MediaPipe face mesh, and macOS display/camera enumeration (via PyObjC) are macOS-only today. Everything else — page building, voice, notes, profile — is plain Python + a browser.

## Run

Double-click **`Read.command`**, or:

```bash
python recorder/record.py
```

One setup window: camera, screen, how you feel, calibration reuse, and what to read (a web page in the adaptive layout, the original page, or a local `.txt`/`.md`/`.html`). Tick **just read** to skip the camera entirely.

Keys on the page: **M** highlight the sentence you're on · **1–4** tag it · **U** undo · **N** notes · **T** text settings · **F** / **Esc** focus mode · **⌘/Alt →** next chapter · **▶** read along.

Afterwards: `python recorder/progress.py` opens your reading profile; study sheets are offered at the end of every session.

## Data and privacy

All data lives in **`recorder/reading.db`** on your machine — one SQLite file with your sessions, per-frame gaze samples, highlights, notes, and preferences. Nothing is uploaded anywhere. No video is ever stored: the camera frames are reduced to a few numbers per frame (gaze estimate, blink, lean-in, frown) and discarded.

This repository contains **code only**. The database, session screenshots, exported notes, and downloaded chapter text are excluded (see `.gitignore`).

## Limitations, honestly

- **n = 1.** The personal model, the attention rule, and every "my data shows" statement in the profile page are one reader's data. The population model is 19 bilingual adults reading one Agatha Christie novel — a proxy for reading difficulty, not a dyslexia dataset.
- **Comprehension is not yet measured.** Dwell time is a proxy. The recall prompts exist to fix this; until they accumulate, "faster" cannot be distinguished from "skimmed."
- **Webcam precision** limits gaze to line-level at best (see lesson 4). Blink rate, presence, lean-in, and frown are calibration-free and comparable across sessions; anything word-level needs calibration under ~60 px.
- **The reading-mode comparison in the profile is not an experiment.** You choose when to switch modes, so a mode used for hard passages will look slower even if it helps.

## Project structure

```
recorder/
  record.py        capture loop, setup UI, calibration, page bridge
  adapt.py         builds the adaptive page (typography, modes, voice, notes, TeX → Unicode)
  features.py      iris → gaze, blink (EAR), lean-in, frown from MediaPipe landmarks
  calibrate.py     9-point polynomial gaze calibration
  db.py            SQLite schema (users, sessions, samples, words, events, notes, checks, settings)
  personalize.py   per-reader difficulty model with shrinkage toward the GECO prior
  progress.py      your reading profile (HTML)
  notes_pdf.py     printable study sheets (Chrome print-to-PDF)
  export_notes.py  Markdown export
  import_csv.py    migrates the early CSV recordings into the database
diagnostics/       the analyses behind the lessons above (label-leak check, honest baseline,
                   personalisation test, granularity test, first validation on my own sessions)
Notebooks/         the original coursework (MLP and LSTM on GECO)
Read.command       double-click launcher
requirements.txt
```

## Acknowledgements

- **GECO** — Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). *Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading.* Behavior Research Methods, 49, 602–615.
- **MediaPipe** face mesh for iris landmarks; **Selenium** for the page bridge.
- Reading material used during development: Daniel Nettle, *From Questions to Knowledge: Data Analysis for Psychology and Behavioural Science using R* (bookdown.org).
- Built with AI-assisted pair programming (Claude Code) for implementation, debugging, and the adversarial review of the analyses. The design decisions, the target definitions, and the reading experience are mine — and so are the lessons.

*If you read differently, you should be able to read anyway.*
