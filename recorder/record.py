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
import json
import re
import subprocess
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


def setup_form(conn, uid):
    """One window, every pre-session question -- her call: many sequential
    dialogs fatigue the user before the reading even starts. Returns a dict:
    device, glasses, feeling, source(kind, value), reuse_calibration."""
    from tkinter import ttk, filedialog
    known = db.known_devices(conn, uid)
    root = tk.Tk(); root.title("Session setup"); root.geometry("440x620")
    root.resizable(True, True)
    pad = {"padx": 14, "pady": 5, "anchor": "w"}

    out = {}

    def start():
        cam = camera.get().strip()
        out.update(camera=(None if cam.startswith("(") else cam),
                   device=(device.get() or "unnamed").strip(),
                   glasses=int(glasses.get()),
                   feeling=(feeling.get() or None),
                   reuse=bool(reuse.get()),
                   kind=kind.get(),
                   url=(url_box.get().strip() or DEFAULT_URL),
                   file=picked["file"])
        root.destroy()

    # packed FIRST with side="bottom": the button owns the bottom strip no
    # matter how tall the rest grows -- packed last, it was the first thing
    # clipped when macOS fonts made the column taller than the window.
    tk.Button(root, text="Start session", font=("Helvetica", 13, "bold"),
              command=start).pack(side="bottom", pady=14)

    tk.Label(root, text="Session setup", font=("Helvetica", 16, "bold")).pack(**pad)

    tk.Label(root, text="Camera (which one is pointed at you):",
             font=("Helvetica", 12, "bold")).pack(**pad)
    cam_names = [n for n, _ in camera_inventory()]
    camera = ttk.Combobox(root, values=cam_names + ["(choose visually)"], width=28)
    remembered_name = db.get_setting(conn, "camera_name")
    camera.set(remembered_name if remembered_name in cam_names
               else (cam_names[0] if cam_names else "(choose visually)"))
    camera.pack(**pad)

    tk.Label(root, text="Screen you are reading on:",
             font=("Helvetica", 12, "bold")).pack(**pad)
    device = ttk.Combobox(root, values=known, width=28)
    if known: device.set(known[0])
    device.pack(**pad)

    glasses = tk.BooleanVar()
    tk.Checkbutton(root, text="wearing glasses", variable=glasses).pack(**pad)

    tk.Label(root, text="How do you feel?").pack(**pad)
    feeling = ttk.Combobox(root, values=["Energized", "Rested", "Neutral",
                                         "Tired", "Fatigued"], width=28)
    feeling.pack(**pad)

    reuse = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="reuse last good calibration for this device",
                   variable=reuse).pack(**pad)

    tk.Label(root, text="Read:").pack(**pad)
    kind = tk.StringVar(value="url_adaptive")
    tk.Radiobutton(root, text="web page -- ADAPTIVE layout", variable=kind,
                   value="url_adaptive").pack(**pad)
    tk.Radiobutton(root, text="web page -- original look", variable=kind,
                   value="url").pack(**pad)
    url_box = tk.Entry(root, width=44); url_box.pack(**pad)
    picked = {"file": None}

    def browse():
        f = filedialog.askopenfilename(parent=root, title="Choose a text file",
            filetypes=[("text", "*.txt *.md *.html *.htm"), ("all", "*.*")])
        if f:
            picked["file"] = f
            kind.set("file")
            file_btn.config(text="file: " + os.path.basename(f))

    file_btn = tk.Button(root, text="...or choose a file on this computer", command=browse)
    file_btn.pack(**pad)

    root.mainloop()
    return out or None


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
def camera_inventory():
    """Cameras in AVFOUNDATION ORDER -- the same order OpenCV indexes -- without
    opening any of them (no camera light, no iPhone chime). [(name, uid), ...]

    Why not system_profiler (what this used to use): it OMITS Continuity
    cameras entirely. With an iPhone connected, system_profiler said
    [FaceTime, C920] while OpenCV saw [FaceTime, C920, iPhone] -- or a
    different order again -- so "position 1" meant two different cameras and
    picking the monitor webcam opened the phone.
    """
    try:                                    # PyObjC ships with Anaconda
        import objc
        ns = {}
        objc.loadBundle("AVFoundation", ns,
                        bundle_path="/System/Library/Frameworks/AVFoundation.framework")
        devs = ns["AVCaptureDevice"].devicesWithMediaType_("vide")
        got = [(str(d.localizedName()), str(d.uniqueID())) for d in devs]
        if got:
            return got
    except Exception:
        pass
    try:                                    # fallback: misses Continuity cameras
        out = subprocess.run(["system_profiler", "-json", "SPCameraDataType"],
                             capture_output=True, text=True, timeout=15).stdout
        cams = json.loads(out).get("SPCameraDataType", [])
        return [(c.get("_name", "?"), c.get("spcamera_unique-id", c.get("_name", "?")))
                for c in cams]
    except Exception:
        return []


def camera_fingerprint():
    """A stable signature of the current camera line-up: sorted Unique IDs.
    Same fingerprint => a saved index still means the same physical device."""
    return "|".join(sorted(uid for _, uid in camera_inventory()))


def _alive(cap, deadline_s=2.0):
    """A camera is alive if ANY read succeeds within the deadline. macOS needs
    warm-up frames, and a single failed first read used to send the user
    straight back to the picker."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        ok, _ = cap.read()
        if ok:
            return True
        time.sleep(0.1)
    return False


def _sees_a_face(cap, face_mesh, tries=8):
    """Does this camera have a person in front of it? The only question that
    actually matters -- and unlike names or indices, the camera itself answers."""
    for _ in range(tries):
        ok, frame = cap.read()
        if not ok:
            continue
        if face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks:
            return True
    return False


def auto_camera(face_mesh, preferred_name=None, max_probe=6):
    """Open the camera that sees the reader -- no questions asked.

    Naming cameras turned out to be unreliable on macOS: system_profiler omits
    Continuity cameras, and even AVFoundation's order did not match OpenCV's
    indices on this machine (verified from a user screenshot: labels landed on
    the wrong tiles). So we stop trusting names and ask each camera directly.

    Order tried: the remembered/preferred one, then everything else, with
    phone-ish devices LAST so a nearby iPhone never wins by default.
    """
    inv = camera_inventory()
    names = [n for n, _ in inv]
    n_cams = max(len(inv), 2)

    def phoneish(i):
        n = names[i].lower() if i < len(names) else ""
        return any(w in n for w in ("iphone", "ipad", "continuity", "desk view"))

    order = list(range(min(n_cams, max_probe)))
    if preferred_name and preferred_name in names:
        p = names.index(preferred_name)
        order = [p] + [i for i in order if i != p]
    # snapshot positions BEFORE sorting: .index() on a list being sorted reads
    # the half-sorted list and raises/misorders
    pos = {v: k for k, v in enumerate(order)}
    order.sort(key=lambda i: (phoneish(i), pos[i]))           # phones last, stable

    opened = None
    for i in order:
        cap = open_camera(i, warm=1.0)
        if cap is None:
            continue
        if _sees_a_face(cap, face_mesh):
            label = names[i] if i < len(names) else f"index {i}"
            print(f"camera: index {i} ({label}) -- it can see you")
            auto_camera.chosen = i
            return cap
        if opened is None:                    # keep the first working one as a backup
            opened = (i, cap)
        else:
            cap.release()
    if opened:
        i, cap = opened
        print(f"camera: index {i} (no face detected yet -- press C in the preview "
              f"to try another)")
        auto_camera.chosen = i
        return cap
    return None


def open_camera(index, warm=1.2):
    """Open one camera index and confirm it actually delivers frames."""
    cap = cv2.VideoCapture(int(index))
    if cap.isOpened() and _alive(cap, deadline_s=warm):
        return cap
    cap.release()
    return None


def pick_camera(max_probe=5, remembered=None, fingerprint=None, preferred_name=None):
    """Show a live thumbnail from every camera macOS offers; press its number.

    Why not just VideoCapture(0): on a Mac, index 0 is whichever camera the OS
    currently favours -- often the iPhone (Continuity Camera) or a monitor's
    webcam, not the laptop. Indices also reshuffle as devices come and go, so a
    hardcoded number breaks next week. Seeing the pictures is the only robust
    answer to 'which camera is which'.
    """
    # A remembered camera opens directly -- probing every device wakes them all
    # (a Continuity iPhone chimes every time).
    #
    # But an OpenCV index is not an identity: macOS renumbers devices when the
    # iPhone appears or disappears, so index 1 can be the laptop today and gone
    # tomorrow. So we remember the index TOGETHER with a fingerprint of the
    # camera list at the time it was chosen, and only trust the index when the
    # list still looks the same. The inventory check opens nothing.
    pick_camera.chosen = None
    pick_camera.explicit = False               # only a real user click saves a preference

    # The reader named a camera in the setup form: use its position in the
    # system_profiler list. Names are stable; indices are not.
    if preferred_name:
        names = [n for n, _ in camera_inventory()]
        if preferred_name in names:
            idx = names.index(preferred_name)
            cap = open_camera(idx)
            if cap is not None:
                pick_camera.chosen = idx
                pick_camera.explicit = True
                print(f"camera: index {idx} = {preferred_name} "
                      f"(press C in the preview if that is the wrong one)")
                return cap
            print(f"'{preferred_name}' (index {idx}) would not start -- falling back")
    now_fp = camera_fingerprint()
    # A remembered index is only meaningful together with the camera line-up it
    # was chosen from. Without a stored fingerprint we do NOT trust it -- that
    # is how a stale '1' opened an iPhone's rear camera.
    if remembered is not None and fingerprint and fingerprint == now_fp:
        try:
            cap = cv2.VideoCapture(int(remembered))
        except (TypeError, ValueError):
            cap = None
        if cap is not None:
            if cap.isOpened() and _alive(cap):
                pick_camera.chosen = int(remembered)
                return cap
            cap.release()
    elif remembered is not None:
        print("camera line-up changed since last time -- re-checking")

    found = []                                     # (index, VideoCapture, frame)
    n_known = len(camera_inventory()) or max_probe
    for i in range(max(max_probe, n_known)):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and _alive(cap, deadline_s=0.8)
        frame = cap.read()[1] if ok else None
        if ok and frame is not None:
            found.append((i, cap, frame))
        else:
            cap.release()
            # stop once we have seen every camera the OS reported (a transient
            # failure on one index used to hide every camera after it)
            if len(found) >= n_known:
                break
    if not found:
        return None
    if len(found) == 1:                            # nothing to choose between
        pick_camera.chosen = found[0][0]
        return found[0][1]

    win = "press the number of the camera showing YOU (Esc = first)"
    choice = found[0][0]
    names = [n for n, _ in camera_inventory()]
    while True:
        tiles = []
        for k, (i, cap, last) in enumerate(found):  # live view, not a stale frame
            ok, frame = cap.read()
            if not ok:
                frame = last                        # one dropped read != blank strip
            t = cv2.resize(frame, (320, 240))
            # label by the camera's OWN index (found may skip indices), and
            # strip non-ASCII -- OpenCV's font drew a curly apostrophe as "???"
            raw = names[i] if i < len(names) else ""
            label = f"{i}: {raw.encode('ascii', 'ignore').decode()}" if raw else str(i)
            cv2.putText(t, label[:22], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            tiles.append(t)
        if not tiles:                               # np.hstack([]) would raise
            break
        cv2.imshow(win, np.hstack(tiles))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if ord("0") <= k <= ord("9") and any(i == k - ord("0") for i, _, _ in found):
            choice = k - ord("0")
            pick_camera.explicit = True             # a real choice: worth saving
            break
    cv2.destroyWindow(win)
    keep = None
    for i, cap, _ in found:
        if i == choice:
            keep = cap
        else:
            cap.release()
    pick_camera.chosen = choice
    return keep


# ------------------------------------------------------------ tracking preview
def tracking_preview(cap, face_mesh):
    """Show the user their own face with the tracker's dots drawn on their irises,
    BEFORE calibration. If the dots aren't on your pupils, no amount of
    calibration can fix it -- fix seating/lighting here instead.

    SPACE continues (only enabled once tracking is healthy), C switches to the
    next camera (wrong one opened?), Esc aborts.
    Returns True to continue, "switch" to try another camera, False to abort."""
    win = "tracking check -- SPACE = go, C = other camera, Esc = quit"
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
        cv2.putText(frame, "SPACE = start   C = different camera   Esc = quit",
                    (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(15) & 0xFF
        if k == 27:
            cv2.destroyWindow(win); return False
        if k in (ord("c"), ord("C")):
            cv2.destroyWindow(win); return "switch"
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
def measured_wpm(conn, uid, default=135):
    """The reader's own pace: distinct words visited per minute, averaged over
    their sessions with trustworthy calibration (<150 px). Becomes the
    read-along default -- the voice starts at the speed their eyes go."""
    row = conn.execute("""
        SELECT SUM(n)*60.0/SUM(dur) FROM (
          SELECT COUNT(DISTINCT sa.word_index) n, MAX(sa.t_ms)/1000.0 dur
          FROM samples sa JOIN sessions se USING(session_id)
          WHERE se.user_id=? AND se.calib_error < 150 AND sa.word_index IS NOT NULL
          GROUP BY sa.session_id HAVING dur > 30)""", (uid,)).fetchone()
    return round(row[0]) if row and row[0] else default


def fetch_article_text(url):
    """Pull the readable text out of a web page: headings, paragraphs, list
    items -- skipping script/style/nav chrome. Deliberately simple (stdlib
    HTMLParser); good for article-like pages, not web apps."""
    import urllib.request
    from html.parser import HTMLParser

    class Grab(HTMLParser):
        KEEP = {"p", "h1", "h2", "h3", "li"}
        SKIP = {"script", "style", "nav", "footer", "header", "aside"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.blocks, self.buf, self.keeping, self.skipping = [], [], 0, 0

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP: self.skipping += 1
            elif tag in self.KEEP and not self.skipping: self.keeping += 1

        def handle_endtag(self, tag):
            if tag in self.SKIP: self.skipping = max(0, self.skipping - 1)
            elif tag in self.KEEP and self.keeping:
                self.keeping -= 1
                text = " ".join("".join(self.buf).split())
                if len(text.split()) >= 3:        # drop menu crumbs
                    self.blocks.append(text)
                self.buf = []

        def handle_data(self, data):
            if self.keeping and not self.skipping: self.buf.append(data)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html_bytes = urllib.request.urlopen(req, timeout=20).read()
    g = Grab(); g.feed(html_bytes.decode("utf-8", errors="replace"))
    text = "\n\n".join(g.blocks)
    if len(text.split()) >= 50:
        return text
    # thin shell -> the page paints its content with JavaScript (bookdown does
    # this: 51 KB of HTML, zero <p> tags). urllib sees the skeleton; a real
    # browser sees the text. Render it headless and harvest what a reader sees.
    print("static fetch too thin -- rendering the page in a headless browser...")
    from selenium import webdriver
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    d = webdriver.Chrome(options=opts)
    try:
        d.get(url)
        time.sleep(3)                       # let the scripts paint
        blocks = d.execute_script("""
            return [...document.querySelectorAll('p, h1, h2, h3, li')]
              .filter(el => !el.closest('nav, footer, aside, header'))
              .map(el => el.innerText.trim().replace(/\\s+/g, ' '))
              .filter(t => t.split(' ').length >= 3);
        """)
        return "\n\n".join(blocks)
    finally:
        d.quit()


def favourite_profile(conn, user_id, default="comfort"):
    """The reading mode this reader actually spends the most time in."""
    rows = conn.execute("""
        SELECT e.value, COUNT(*) FROM events e JOIN sessions s USING(session_id)
        WHERE s.user_id = ? AND e.kind = 'profile' AND e.value IS NOT NULL
        GROUP BY e.value ORDER BY COUNT(*) DESC LIMIT 1""", (user_id,)).fetchone()
    return rows[0] if rows else default


def resolve_text(cfg):
    """Turn the setup form's answers into (url_for_selenium, human_name)."""
    if cfg["kind"] == "file" and cfg["file"]:
        path = cfg["file"]
        if path.lower().endswith((".html", ".htm")):
            return "file://" + os.path.abspath(path), os.path.basename(path)
        from recorder import adapt
        out = adapt.build_page(path, os.path.join(os.path.dirname(DB_PATH), "texts"),
                               wpm=pick_text.wpm, model=resolve_text.model,
                               profile=resolve_text.profile)
        return "file://" + os.path.abspath(out), os.path.basename(path)
    url = cfg["url"]
    if cfg["kind"] == "url_adaptive":
        try:
            text = fetch_article_text(url)
            if len(text.split()) < 50:
                raise ValueError("page yielded too little text")
            tmp_dir = os.path.join(os.path.dirname(DB_PATH), "texts")
            os.makedirs(tmp_dir, exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", url.split("//")[-1])[:60]
            src = os.path.join(tmp_dir, name + ".txt")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(text)
            from recorder import adapt
            out = adapt.build_page(src, tmp_dir, wpm=pick_text.wpm,
                                   model=resolve_text.model, profile=resolve_text.profile)
            return "file://" + os.path.abspath(out), url
        except Exception as e:
            # LOUD failure. This used to print only to a terminal the reader
            # wasn't watching, so the session silently recorded the plain
            # website and the adaptive design appeared "not to work".
            print(f"could not adapt {url} ({e}); asking the reader")
            retry = ask("Adaptive layout not possible",
                        f"This page gave too little text to adapt ({e}).\n\n"
                        "It is probably a contents page or a stub chapter.\n\n"
                        "Enter a different URL to adapt instead, or leave blank "
                        "to read the ORIGINAL page with no adaptive layout:")
            if retry and retry.strip():
                return resolve_text({**cfg, "url": retry.strip()})
            print("continuing with the ORIGINAL page (no adaptive layout)")
    return url, url


def pick_text():
    """URL, or a local file (.txt/.md/.html). Local text is wrapped in a clean,
    readable page -- which later becomes the page the adaptive layout rewrites.
    Returns (url_for_selenium, human_name)."""
    from tkinter import filedialog
    root = tk.Tk(); root.title("What to read?"); root.geometry("320x180")
    choice = {}

    def use_url_adaptive():
        choice["v"] = ("url_adaptive", None); root.destroy()

    def use_url():
        choice["v"] = ("url", None); root.destroy()

    def use_file():
        path = filedialog.askopenfilename(parent=root, title="Choose a text file",
            filetypes=[("text", "*.txt *.md *.html *.htm"), ("all", "*.*")])
        if path:
            choice["v"] = ("file", path); root.destroy()

    tk.Label(root, text="Read from:", font=("Helvetica", 14)).pack(pady=10)
    tk.Button(root, text="web page -- ADAPTIVE layout", width=26,
              command=use_url_adaptive).pack(pady=4)
    tk.Button(root, text="web page -- original look", width=26,
              command=use_url).pack(pady=4)
    tk.Button(root, text="a file on this computer", width=26,
              command=use_file).pack(pady=4)
    root.mainloop()
    kind, path = choice.get("v", ("url", None))
    if kind in ("url", "url_adaptive"):
        url = ask("Text", "URL to read (blank = default):") or DEFAULT_URL
        if kind == "url_adaptive":
            try:
                text = fetch_article_text(url)
                if len(text.split()) < 50:
                    raise ValueError("page yielded too little text")
                tmp_dir = os.path.join(os.path.dirname(DB_PATH), "texts")
                os.makedirs(tmp_dir, exist_ok=True)
                name = re.sub(r"[^\w.-]+", "_", url.split("//")[-1])[:60]
                src = os.path.join(tmp_dir, name + ".txt")
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write(text)
                from recorder import adapt
                out = adapt.build_page(src, tmp_dir, wpm=pick_text.wpm)
                return "file://" + os.path.abspath(out), url
            except Exception as e:
                print(f"could not adapt {url} ({e}); opening as-is")
        return url, url
    if path.lower().endswith((".html", ".htm")):
        return "file://" + os.path.abspath(path), os.path.basename(path)
    # plain text/markdown -> the adaptive page (profiles + hard-word styling)
    from recorder import adapt
    out = adapt.build_page(path, os.path.join(os.path.dirname(DB_PATH), "texts"),
                           wpm=pick_text.wpm)
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
                if (r.width <= 0 || r.height <= 0) return;
                wordMap.push({text: span.innerText.trim(), left: r.left,
                              top: r.top + window.scrollY,
                              right: r.right, bottom: r.bottom + window.scrollY});
            });
            return wordMap;
        }
        // Only LEAF blocks, and never site chrome. Rewriting a parent <li>
        // detaches its nested children (a gitbook sidebar is ~200 nested <li>),
        // which produced ~1,600 zero-area duplicate "words" and destroyed every
        // navigation link on the page.
        [...document.querySelectorAll('p, h1, h2, li')]
          .filter(n => !n.querySelector('p, h1, h2, li'))
          .filter(n => !n.closest('nav, aside, header, footer'))
          .forEach(node => {
            node.innerHTML = node.innerText.split(/\\s+/)
                .map(w => `<span>${w}</span>`).join(' ');
            node.querySelectorAll('span').forEach(span => {
                var r = span.getBoundingClientRect();
                // a zero-area box is a hidden or collapsed element: it can never
                // be looked at, and it would pollute the words table
                if (span.innerText.trim().length > 0 && r.width > 0 && r.height > 0)
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
RECORDER_VERSION = "v13: the page learns your reading; notes panel fixed"


def main():
    print(f"recorder {RECORDER_VERSION}")
    conn = db.connect(DB_PATH)
    uid = pick_user(conn)
    if uid is None:
        print("no user chosen"); return

    screen_w, screen_h = pyautogui.size()   # calibration + session row need these

    cfg = setup_form(conn, uid)
    if not cfg:
        print("setup cancelled"); return

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    cap = auto_camera(face_mesh,
                      preferred_name=cfg.get("camera") or db.get_setting(conn, "camera_name"))
    if cap is None:
        print("no camera found -- check System Settings > Privacy & Security > Camera")
        return
    pick_camera.chosen = getattr(auto_camera, "chosen", 0)

    # Can the tracker see your pupils -- and is this even the right camera?
    # C cycles to the next one, so a wrong camera is never a dead end.
    names = [n for n, _ in camera_inventory()]
    while True:
        verdict = tracking_preview(cap, face_mesh)
        if verdict is True:
            break
        cap.release()
        if verdict is False:
            print("aborted at tracking check"); return
        start = (pick_camera.chosen or 0) + 1        # "switch": try the next one
        cap = None
        for step in range(max(len(names), 5)):
            idx = (start + step) % max(len(names), 5)
            cap = open_camera(idx)
            if cap is not None:
                pick_camera.chosen = idx
                pick_camera.explicit = True
                print(f"camera: {names[idx] if idx < len(names) else 'index ' + str(idx)}")
                break
        if cap is None:
            print("no other camera responded"); return

    # Save the choice the reader CONFIRMED by pressing SPACE on a good preview,
    # together with the camera line-up it belongs to.
    if pick_camera.chosen is not None:
        db.set_setting(conn, "camera_index", pick_camera.chosen)
        db.set_setting(conn, "camera_fingerprint", camera_fingerprint())
        if pick_camera.chosen < len(names):
            db.set_setting(conn, "camera_name", names[pick_camera.chosen])

    calib_note = None
    prev = db.last_calibration(conn, uid, cfg["device"]) if cfg["reuse"] else None
    if prev:
        # returning reader on a known screen: skip the dots. The trade is drift
        # (today's seating differs from that day's), so the reuse is on record.
        mapping, err = json.loads(prev[1]), prev[2]
        calib_note = f"calibration reused from session {prev[0]} ({prev[3][:10]})"
        print(f"reusing calibration from session {prev[0]}: {err:.0f} px on {cfg['device']}")
    else:
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

    pick_text.wpm = measured_wpm(conn, uid)
    from recorder import personalize
    resolve_text.model = personalize.load(conn, uid)      # None until enough data
    resolve_text.profile = favourite_profile(conn, uid)
    if resolve_text.model:
        print(f"word difficulty: {resolve_text.model['weight_you']:.0%} learned from "
              f"your own reading ({resolve_text.model['n_words']:,} words)")
    else:
        print("word difficulty: general model (not personalised yet)")
    url, text_name = resolve_text(cfg)
    print("ADAPTIVE layout: " + ("YES -> " + os.path.basename(url)
                                 if url.endswith(".adaptive.html")
                                 else "NO (reading the original page)"))
    browser, word_map, off_x, off_y = open_text(url)

    mm_w, mm_h = display_info()
    sid = db.start_session(conn, uid, text_source=text_name, screen_w=screen_w,
                           screen_h=screen_h, self_report=cfg["feeling"],
                           screen_w_mm=mm_w, screen_h_mm=mm_h,
                           device_label=cfg["device"], glasses=cfg["glasses"])
    if calib_note:
        conn.execute("UPDATE sessions SET notes = ? WHERE session_id = ?",
                     (calib_note, sid)); conn.commit()
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
    tts_state = None               # read-along state, logged on every change
    marks_state = None             # marked passages + notes, mirrored to the db
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
                            sy, prof, tts, marks = browser.execute_script(
                                "return [window.scrollY, window.__profile || null,"
                                " window.__tts || null, window.__marks || null];")
                            if sy is not None:          # None while the page is mid-load
                                scroll_y = float(sy)
                            if prof != profile:         # reader switched mode: that's data
                                profile = prof
                                db.add_event(conn, sid, t_ms, "profile", prof)
                            if tts != tts_state:        # read-along toggled / speed changed
                                tts_state = tts
                                db.add_event(conn, sid, t_ms, "tts", tts)
                            if marks != marks_state:    # reader marked/annotated a passage
                                marks_state = marks
                                db.add_event(conn, sid, t_ms, "mark", marks)
                                try:
                                    db.save_notes(conn, sid, json.loads(marks or "[]"))
                                except Exception:
                                    pass
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
        # runs on q, on Ctrl-C, and on any bug: the session is ALWAYS closed out
        # and every resource released. Previously only the flush was protected,
        # so Ctrl-C left the session unsummarised and Chrome/camera running.
        writer.flush()
        fps = n_frames / max(time.monotonic() - t0, 1e-9)
        db.end_session(conn, sid, camera_fps=fps)
        for release in (browser.quit, cap.release, cv2.destroyAllWindows):
            try:
                release()
            except Exception:
                pass

    # ground truth, while the reading is fresh
    difficulty = ask("After reading", "How hard was that, 1 (easy) to 5 (very hard)?")
    if difficulty:
        db.add_check(conn, sid, "self_report", "difficulty 1-5", difficulty,
                     word_start=0, word_end=max(len(word_map) - 1, 0))

    # re-fit the reader's difficulty model with this session included, so the
    # NEXT page they open is tuned a little more to them
    try:
        from recorder import personalize
        m = personalize.fit(uid, DB_PATH, verbose=False)
        if m:
            print(f"personal model updated: {m['weight_you']:.0%} you "
                  f"({m['n_words']:,} words, {m['sessions']} sessions)")
    except Exception as e:
        print(f"(could not update personal model: {e})")

    print(f"\nsession {sid} saved ({n_frames} frames at {fps:.0f} fps)")
    print("history for this user:")
    for row in db.user_progress(conn, uid):
        # a session that captured no frames has NULL summary stats; formatting
        # None with :.0f raises, which used to crash the run AFTER the data was
        # already safely saved -- the worst kind of cosmetic bug.
        dur = f"{row[4]:.0f}s" if row[4] is not None else "  -  "
        face = f"{row[5]:.0%}" if row[5] is not None else " - "
        print(f"  session {row[0]}  {row[1]}  feel={row[2]}  {dur}  "
              f"face {face}  {row[6] or 0} words  median dwell {row[7] or 0:.0f} ms")


if __name__ == "__main__":
    main()
