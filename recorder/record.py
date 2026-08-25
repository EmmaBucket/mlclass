"""
The recorder, rebuilt around the database.

    python recorder/record.py            -> pick a user, calibrate, read, record

Flow: user picker (click a name or add one) -> 9-dot calibration -> the browser
opens the text -> recording runs until you press q in the camera window -> a
"how hard was that?" prompt -> session summary saved.

Kept from your original: MediaPipe face mesh, the word-map idea (every word
wrapped in a span so gaze can be attributed to a word), Selenium for scroll.
Changed: eye geometry (recorder/features.py -- the old vertical math divided by
zero), norm->screen mapping is now calibrated per sitting (recorder/calibrate.py),
and storage is SQLite keyed by user_id instead of loose CSVs keyed by whatever
name was typed that day ("Emma" and "Emma B" are different people to a CSV).
"""
import time
import tkinter as tk
from tkinter import simpledialog

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import db, features, calibrate

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reading.db")
DEFAULT_URL = "https://gsas.harvard.edu/news/reading-between-leaves-importance-being-good-reader"


# ---------------------------------------------------------------- user picker
def pick_user(conn):
    """A window with one button per known user, plus 'new user...'.
    Clicking 'Emma' returns Emma's user_id -- the pairing you asked for."""
    result = {}
    root = tk.Tk()
    root.title("Who is reading?")
    root.geometry("300x400")
    tk.Label(root, text="Who is reading?", font=("Helvetica", 16)).pack(pady=10)

    def choose(uid):
        result["uid"] = uid
        root.destroy()

    for uid, name, n_sessions in db.list_users(conn):
        tk.Button(root, text=f"{name}  ({n_sessions} sessions)", width=24,
                  command=lambda u=uid: choose(u)).pack(pady=4)

    def new_user():
        name = simpledialog.askstring("New user", "Name:", parent=root)
        if name and name.strip():
            choose(db.get_or_create_user(conn, name.strip()))

    tk.Button(root, text="+ new user...", width=24, command=new_user).pack(pady=12)
    root.mainloop()
    return result.get("uid")


def display_info():
    """Physical screen size in mm (macOS). Pixels alone can't compare devices:
    2560px on a 27-inch monitor and on a 13-inch laptop are ~2x different letter
    heights on the retina, and reading behaviour follows the physical size."""
    try:
        import Quartz
        d = Quartz.CGMainDisplayID()
        size = Quartz.CGDisplayScreenSize(d)
        return float(size.width), float(size.height)
    except Exception:
        return None, None


def pick_device(conn, uid):
    """This user's known screens as buttons, plus 'new screen...'. Same idea as
    the user picker: a stored label beats retyping (and re-typo-ing) it."""
    known = db.known_devices(conn, uid)
    if not known:
        return ask("Device", "Short name for this machine/screen (e.g. 'macbook-13'):")
    result = {}
    root = tk.Tk(); root.title("Which screen?"); root.geometry("300x300")
    tk.Label(root, text="Which screen are you on?", font=("Helvetica", 14)).pack(pady=10)

    def choose(label):
        result["d"] = label; root.destroy()

    for label in known:
        tk.Button(root, text=label, width=24, command=lambda l=label: choose(l)).pack(pady=4)

    def new():
        name = simpledialog.askstring("New screen", "Short name:", parent=root)
        if name and name.strip():
            choose(name.strip())

    tk.Button(root, text="+ new screen...", width=24, command=new).pack(pady=12)
    root.mainloop()
    return result.get("d")


def ask(title, prompt):
    root = tk.Tk(); root.withdraw()
    answer = simpledialog.askstring(title, prompt, parent=root)
    root.destroy()
    return answer


# ------------------------------------------------------------------ camera picker
def pick_camera(max_probe=5):
    """Show a live thumbnail from every camera macOS offers; press its number.

    Why not just VideoCapture(0): on a Mac, index 0 is whichever camera the OS
    currently favours -- often the iPhone (Continuity Camera) or a monitor's
    webcam, not the laptop. Indices also reshuffle as devices come and go, so a
    hardcoded number breaks next week. Seeing the pictures is the only robust
    answer to 'which camera is which'.
    """
    found = []                                     # (index, VideoCapture, frame)
    for i in range(max_probe):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            found.append((i, cap, frame))
        else:
            cap.release()
            if found:      # indices are contiguous; first gap = end of the list
                break
    if not found:
        return None
    if len(found) == 1:                            # nothing to choose between
        return found[0][1]

    win = "press the number of the camera showing YOU (Esc = first)"
    choice = found[0][0]
    while True:
        tiles = []
        for i, cap, _ in found:                    # live view, not a stale frame
            ok, frame = cap.read()
            if not ok:
                continue
            t = cv2.resize(frame, (320, 240))
            cv2.putText(t, str(i), (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
            tiles.append(t)
        cv2.imshow(win, np.hstack(tiles))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if ord("0") <= k <= ord("9") and any(i == k - ord("0") for i, _, _ in found):
            choice = k - ord("0")
            break
    cv2.destroyWindow(win)
    keep = None
    for i, cap, _ in found:
        if i == choice:
            keep = cap
        else:
            cap.release()
    return keep


# ------------------------------------------------------------ tracking preview
def tracking_preview(cap, face_mesh):
    """Show the user their own face with the tracker's dots drawn on their irises,
    BEFORE calibration. If the dots aren't on your pupils, no amount of
    calibration can fix it -- fix seating/lighting here instead.

    SPACE continues (only enabled once tracking is healthy), Esc aborts."""
    win = "tracking check -- SPACE when the dots sit on your pupils"
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        msgs, healthy = [], False
        # lighting: mean brightness of the frame centre, where the face should be
        h, w = frame.shape[:2]
        brightness = float(frame[h//4:3*h//4, w//4:3*w//4].mean())
        if brightness < 60:
            msgs.append("too dark -- add light in front of you (not behind)")
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            f = features.extract(lms, frame.shape)
            # draw where the tracker thinks the irises are
            for idx_set in (features.R_IRIS, features.L_IRIS):
                c = np.mean([[lms[i].x * w, lms[i].y * h] for i in idx_set], axis=0)
                cv2.circle(frame, (int(c[0]), int(c[1])), 4, (0, 255, 0), -1)
            for i in (features.R_CORNER_OUT, features.R_CORNER_IN, features.R_LID_UP,
                      features.R_LID_LOW, features.L_CORNER_OUT, features.L_CORNER_IN,
                      features.L_LID_UP, features.L_LID_LOW):
                cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 2, (255, 200, 0), -1)
            if f["norm_x"] is None:
                msgs.append("eye too small in frame -- move closer to the camera")
            elif f["ear"] is not None and f["ear"] < 0.18:
                msgs.append("eyes read as nearly closed -- raise screen or chin")
            elif f["eye_span"] < 90:
                msgs.append("sit a bit closer")
            else:
                healthy = True
                msgs.append("tracking OK -- press SPACE")
        else:
            msgs.append("no face found -- face the camera")
        for j, m in enumerate(msgs):
            cv2.putText(frame, m, (12, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if healthy else (0, 0, 255), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(15) & 0xFF
        if k == 27:
            cv2.destroyWindow(win); return False
        if k == 32 and healthy:
            cv2.destroyWindow(win); return True


# ---------------------------------------------------------------- calibration
def run_calibration(cap, face_mesh, screen_w, screen_h):
    """Fullscreen dots; hold your gaze on each, press SPACE, we sample 15 frames.
    Returns (mapping, mean_error_px) or (None, None) if aborted."""
    pts = calibrate.grid_points(screen_w, screen_h)
    norm_obs, screen_obs = [], []
    win = "calibration"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for (px, py) in pts:
        # wait for SPACE while showing the dot
        while True:
            canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
            cv2.circle(canvas, (px, py), 14, (0, 0, 255), -1)
            cv2.putText(canvas, "look at the dot, press SPACE (Esc quits)",
                        (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                cv2.destroyWindow(win); return None, None
            if k == 32:
                break
        # sample frames, then keep the MEDIAN of blink-free ones.
        # A blink is a TRANSIENT dip in eye openness. It cannot be an absolute
        # cutoff: session 2 proved that looking at the bottom of the screen
        # lowers the lids legitimately (mean EAR 0.24 down there vs 0.28 up
        # top), so a fixed 0.22 threshold rejected the bottom dots entirely and
        # the fit extrapolated -- 398 px error. Instead: compare each frame to
        # the MEDIAN EAR AT THIS DOT. Sustained low (gazing down) keeps a low
        # median and passes; a blink dips far below whatever the median is.
        raw = []
        for _ in range(25):
            ok, frame = cap.read()
            if not ok:
                continue
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                f = features.extract(res.multi_face_landmarks[0].landmark, frame.shape)
                if f["norm_x"] is not None and f["ear"] is not None:
                    raw.append((f["norm_x"], f["norm_y"], f["ear"]))
        if len(raw) < 5:
            print(f"  dot ({px},{py}): face lost, skipping")
            continue
        med_ear = float(np.median([r[2] for r in raw]))
        samples = [(nx, ny) for nx, ny, ear in raw
                   if ear > 0.7 * med_ear and -0.5 <= ny <= 1.5]
        if len(samples) < 5:
            print(f"  dot ({px},{py}): too few stable frames, skipping")
            continue
        norm_obs.append(np.median(samples, axis=0))
        screen_obs.append((px, py))

    cv2.destroyWindow(win)
    if len(norm_obs) < 6:
        return None, None
    return calibrate.fit(norm_obs, screen_obs)


# ------------------------------------------------------------------- reading material
def pick_text():
    """URL, or a local file (.txt/.md/.html). Local text is wrapped in a clean,
    readable page -- which later becomes the page the adaptive layout rewrites.
    Returns (url_for_selenium, human_name)."""
    from tkinter import filedialog
    root = tk.Tk(); root.title("What to read?"); root.geometry("320x180")
    choice = {}

    def use_url():
        choice["v"] = ("url", None); root.destroy()

    def use_file():
        path = filedialog.askopenfilename(parent=root, title="Choose a text file",
            filetypes=[("text", "*.txt *.md *.html *.htm"), ("all", "*.*")])
        if path:
            choice["v"] = ("file", path); root.destroy()

    tk.Label(root, text="Read from:", font=("Helvetica", 14)).pack(pady=10)
    tk.Button(root, text="a web page (URL)", width=22, command=use_url).pack(pady=4)
    tk.Button(root, text="a file on this computer", width=22, command=use_file).pack(pady=4)
    root.mainloop()
    kind, path = choice.get("v", ("url", None))
    if kind == "url":
        url = ask("Text", "URL to read (blank = default):") or DEFAULT_URL
        return url, url
    if path.lower().endswith((".html", ".htm")):
        return "file://" + os.path.abspath(path), os.path.basename(path)
    # plain text/markdown -> the adaptive page (profiles + hard-word styling)
    from recorder import adapt
    out = adapt.build_page(path, os.path.join(os.path.dirname(DB_PATH), "texts"))
    return "file://" + os.path.abspath(out), os.path.basename(path)


# ------------------------------------------------------------------- browser
def open_text(url):
    """Selenium + word map, same approach as your original."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3)
    word_map = driver.execute_script("""
        var wordMap = [];
        var own = document.querySelectorAll('span[data-w]');
        if (own.length > 0) {              // adaptive page: it brings its own
            own.forEach(span => {          // indexed spans -- rewrapping would
                var r = span.getBoundingClientRect();   // destroy its styling
                wordMap.push({text: span.innerText.trim(), left: r.left,
                              top: r.top + window.scrollY,
                              right: r.right, bottom: r.bottom + window.scrollY});
            });
            return wordMap;
        }
        document.querySelectorAll('p, h1, h2, li').forEach(node => {
            node.innerHTML = node.innerText.split(/\\s+/)
                .map(w => `<span>${w}</span>`).join(' ');
            node.querySelectorAll('span').forEach(span => {
                var r = span.getBoundingClientRect();
                if (span.innerText.trim().length > 0)
                    wordMap.push({text: span.innerText.trim(), left: r.left,
                                  top: r.top + window.scrollY,   // document coords
                                  right: r.right, bottom: r.bottom + window.scrollY});
            });
        });
        return wordMap;
    """)
    rect = driver.get_window_rect()
    header = rect["height"] - driver.execute_script("return window.innerHeight;")
    return driver, word_map, rect["x"], rect["y"] + header


class WordIndex:
    """Which word is under the gaze? Four numpy comparisons over every box at once.

    The obvious python loop costs ~3.5 ms per frame on a 3000-word page; this
    costs ~0.02 ms. Same answer (verified against the loop on 2000 random
    gazes) -- the win is that numpy does the comparisons in C, in bulk."""

    def __init__(self, word_map):
        self.L = np.array([w["left"] for w in word_map], float)
        self.R = np.array([w["right"] for w in word_map], float)
        self.T = np.array([w["top"] for w in word_map], float)
        self.B = np.array([w["bottom"] for w in word_map], float)

    def at(self, gx, gy, scroll_y, off_x, off_y):
        x, y = gx - off_x, gy - off_y + scroll_y      # screen -> document coords
        hit = np.flatnonzero((self.L <= x) & (x <= self.R) & (self.T <= y) & (y <= self.B))
        return int(hit[0]) if hit.size else None


# ---------------------------------------------------------------------- main
def main():
    conn = db.connect(DB_PATH)
    uid = pick_user(conn)
    if uid is None:
        print("no user chosen"); return

    self_report = ask("Before you read", "One word: how do you feel? (rested / tired / ...)")
    screen_w, screen_h = pyautogui.size()

    cap = pick_camera()
    if cap is None:
        print("no camera found -- check System Settings > Privacy & Security > Camera")
        return
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    # can the tracker even see your pupils? settle that before calibrating.
    if not tracking_preview(cap, face_mesh):
        print("aborted at tracking check"); cap.release(); return

    while True:
        print("calibrating...")
        mapping, err = run_calibration(cap, face_mesh, screen_w, screen_h)
        if mapping is None:
            print("calibration aborted"); cap.release(); return
        verdict = ("word-level ok" if err < 40 else
                   "line-level only" if err < 120 else "poor")
        print(f"calibration error: {err:.0f} px ({verdict})")
        if err < 120:
            break
        again = ask("Calibration", f"Error {err:.0f} px is poor. Type r to redo "
                    "(fix lighting/seating first), anything else to continue anyway:")
        if (again or "").strip().lower() != "r":
            break

    url, text_name = pick_text()
    browser, word_map, off_x, off_y = open_text(url)

    mm_w, mm_h = display_info()
    device = pick_device(conn, uid)
    glasses = ask("Glasses", "Wearing glasses right now? (y/n)")
    glasses = 1 if (glasses or "").strip().lower().startswith("y") else 0
    sid = db.start_session(conn, uid, text_source=text_name, screen_w=screen_w,
                           screen_h=screen_h, self_report=self_report,
                           screen_w_mm=mm_w, screen_h_mm=mm_h, device_label=device,
                           glasses=glasses)
    db.save_calibration(conn, sid, mapping, err)
    db.save_words(conn, sid, word_map)

    # one screenshot of the page as this reader saw it (~60 KB, ~0.7 s, once).
    # Layout analysis uses the word boxes; this is for human eyes -- rendering
    # heatmaps over the real page when hunting for where readers bottleneck.
    shots = os.path.join(os.path.dirname(DB_PATH), "screenshots")
    os.makedirs(shots, exist_ok=True)
    shot_path = os.path.join(shots, f"session_{sid}.png")
    try:
        browser.save_screenshot(shot_path)
        db.set_screenshot(conn, sid, shot_path)
    except Exception as e:
        print(f"screenshot failed ({e}); continuing without")
    writer = db.SampleWriter(conn, sid)

    word_index = WordIndex(word_map)
    t0 = time.monotonic()          # monotonic: wall clocks jump (NTP), session clocks must not
    n_frames = 0
    scroll_y = 0.0                 # asked every 5th frame: a 2 ms browser round-trip
                                   # per frame buys nothing -- people scroll on a
                                   # scale of seconds, gaze moves on a scale of ms
    profile = None                 # current reading mode, logged on every change
    print("recording -- press q in the camera window to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = (time.monotonic() - t0) * 1000
            n_frames += 1
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_face_landmarks:
                f = features.extract(res.multi_face_landmarks[0].landmark, frame.shape)
                gaze_x = gaze_y = wi = None
                if f["norm_x"] is not None:
                    gaze_x, gaze_y = calibrate.apply(mapping, f["norm_x"], f["norm_y"])[0]
                    if n_frames % 5 == 1:
                        try:
                            sy = browser.execute_script("return window.scrollY;")
                            if sy is not None:          # None while the page is mid-load
                                scroll_y = float(sy)
                        except Exception:
                            pass                        # browser busy/navigating: keep last value
                    wi = word_index.at(gaze_x, gaze_y, scroll_y, off_x, off_y)
                writer.add(t_ms, 1, f["norm_x"], f["norm_y"], gaze_x, gaze_y,
                           f["eye_span"], f["frown"], f["ear"],
                           scroll_y if f["norm_x"] is not None else None, wi)
            else:
                writer.add(t_ms, 0)

            cv2.imshow("recording (q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # runs on q, on Ctrl-C, and on any bug: the session is always closed out
        writer.flush()
    fps = n_frames / max(time.monotonic() - t0, 1e-9)

    # ground truth, while the reading is fresh
    difficulty = ask("After reading", "How hard was that, 1 (easy) to 5 (very hard)?")
    if difficulty:
        db.add_check(conn, sid, "self_report", "difficulty 1-5", difficulty,
                     word_start=0, word_end=len(word_map) - 1)

    db.end_session(conn, sid, camera_fps=fps)
    browser.quit(); cap.release(); cv2.destroyAllWindows()

    print(f"\nsession {sid} saved ({n_frames} frames at {fps:.0f} fps)")
    print("history for this user:")
    for row in db.user_progress(conn, uid):
        print(f"  session {row[0]}  {row[1]}  feel={row[2]}  {row[4]:.0f}s  "
              f"face {row[5]:.0%}  {row[6]} words  median dwell {row[7] or 0:.0f} ms")


if __name__ == "__main__":
    main()
