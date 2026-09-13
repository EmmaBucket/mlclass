# LinkedIn post — draft

*Pick one. Both are first-person and true to the repo; edit freely.*

---

## Version A — the story (recommended)

I'm dyslexic, and long technical reading has always been the hardest part of studying. So for my machine-learning course I built something for myself: an eye-tracking reading tool that adapts the text to the person reading it.

Point a webcam at yourself, open a textbook chapter, and the page rebuilds: better typography, the hard words marked, a voice that reads along at the pace my eyes were measured to read, highlights that turn into a study sheet, and a layout that learns which words slow *me* down. Everything runs locally.

The part I'm proudest of isn't a feature. It's a mistake.

My first model scored AUC 0.97 at detecting "reading breakdowns." It was cheating: the label was built from the same reading-time columns I was feeding it. A one-line if-statement reproduced it perfectly. The honest version — predict difficulty from the text alone, tested on readers the model has never seen — scores 0.75 to 0.80. A real 0.75 taught me more than a fake 0.97.

A few other things the data insisted on:
• Which words are hard is predictable. Whether *I* am struggling right now is not — it can be measured and reacted to, but not forecast. So the design predicts the text and measures the reader.
• A webcam resolves roughly a 21×11 grid of gaze positions. Word-level tracking on a laptop is physics, not a bug. I stopped trying to automate typography experiments the instrument can't measure, and built controls the reader owns instead.
• The bugs that never crash are the ones that matter. Three of them produced perfectly clean-looking data.

It's open on GitHub — code only, my reading data stays on my machine — and I'm still using it to read every week.

github.com/EmmaBucket/mlclass

#Dyslexia #Accessibility #MachineLearning #EyeTracking #Python #AssistiveTechnology #LearningInPublic

---

## Version B — shorter

Built a reading tool for my own dyslexia and learned machine learning properly along the way.

It watches your eyes through a webcam, rebuilds any textbook chapter with dyslexia-informed typography, reads along at your own measured pace, turns highlights into study sheets, and learns which words slow you down. All local, all yours.

Biggest lesson: my first model's AUC 0.97 was a label leak. The honest model scores 0.75 — and that number is worth more.

Code on GitHub: github.com/EmmaBucket/mlclass

#Dyslexia #Accessibility #MachineLearning #EyeTracking

---

## Notes before posting

- Add one screenshot of the adaptive page (Comfort mode with a highlight and the notes panel open) — posts with an image get far more reach. Take it on a **non-personal** chapter; the profile page shows your own data.
- The post says "code only": make sure the privacy cleanup below has been done before the link goes live.
- If your instructor's name or course code matter to you, add a line — but the post reads fine without them.
