import cv2
import time
import csv
import os
import mediapipe as mp
import numpy as np
import pyautogui
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 1. SETUP OUTPUT DIRECTORY
output_dir = "/Users/emmamaenhout/Desktop/Channel Islands/Spring 2026/Machine Learning/mlclass/gaze_data_notebooks"
os.makedirs(output_dir, exist_ok=True)

user_id = input("Enter username: ")
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"gaze_data_{user_id}_{timestamp_str}.csv")

# 2. BROWSER & WORD MAPPING SETUP
def setup_browser_and_map(url):
    print(f"Opening browser to: {url}")
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized") # Essential for coordinate alignment
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3) # Wait for page to fully render
    
    # JavaScript to wrap every word in a span to get individual word coordinates
    # This is much more accurate for "word difficulty" tracking
    script = """
    var wordMap = [];
    var textNodes = document.querySelectorAll('p, h1, h2, li');
    textNodes.forEach(node => {
        var text = node.innerText;
        var words = text.split(/\s+/);
        node.innerHTML = words.map(word => `<span>${word}</span>`).join(' ');
        
        var spans = node.querySelectorAll('span');
        spans.forEach(span => {
            var rect = span.getBoundingClientRect();
            if (span.innerText.trim().length > 0) {
                wordMap.push({
                    "text": span.innerText.trim(),
                    "left": rect.left,
                    "top": rect.top,
                    "right": rect.right,
                    "bottom": rect.bottom,
                    "length": span.innerText.trim().length
                });
            }
        });
    });
    return wordMap;
    """
    word_map = driver.execute_script(script)
    
    # Get the browser's position on your monitor to align eye-tracking pixels
    window_rect = driver.get_window_rect()
    # Calculate height of tabs/URL bar (usually window height - viewport height)
    viewport_height = driver.execute_script("return window.innerHeight;")
    header_height = window_rect['height'] - viewport_height
    
    return driver, word_map, window_rect['x'], window_rect['y'] + header_height

# --- Gaze Tracker Initializations ---
SCREEN_W, SCREEN_H = pyautogui.size()
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 133]
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_EYE = [362, 263]

def get_iris_center(landmarks, indices, shape):
    points = np.array([(int(landmarks[i].x * shape[1]), int(landmarks[i].y * shape[0])) for i in indices])
    return np.mean(points, axis=0)

def get_eye_bounds(landmarks, indices, shape):
    points = np.array([(int(landmarks[i].x * shape[1]), int(landmarks[i].y * shape[0])) for i in indices])
    return np.min(points[:,0]), np.max(points[:,0]), np.min(points[:,1]), np.max(points[:,1])

def get_word_at_gaze(gaze_x, gaze_y, word_map, scroll_y, offset_x, offset_y):
    # Convert monitor-relative pixels to browser-viewport pixels
    target_x = gaze_x - offset_x
    target_y = gaze_y - offset_y
    
    for entry in word_map:
        # Check against the word boxes (adjusted for current scroll)
        # Note: If wordMap is generated ONCE, entry['top'] is fixed. 
        # We check if (gaze + scroll) matches the original document position.
        if entry['left'] <= target_x <= entry['right'] and \
           entry['top'] <= (target_y + scroll_y) <= entry['bottom']:
            return entry['text'], entry['length']
    return "None", 0

# Start Browser and Map
target_url = "https://gsas.harvard.edu/news/reading-between-leaves-importance-being-good-reader"
browser, active_word_map, off_x, off_y = setup_browser_and_map(target_url)

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "face_detected", "avg_norm_x", "avg_norm_y", "screen_x", "screen_y", "current_text", "text_length"])

    while True:
        ret, frame = cap.read()
        if not ret: break

        face_detected = 0
        avg_norm_x, avg_norm_y = 0.5, 0.5
        screen_x, screen_y = -1, -1
        current_text, text_len = "None", 0

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_detected = 1
            landmarks = results.multi_face_landmarks[0].landmark
            shape = frame.shape

            l_iris = get_iris_center(landmarks, LEFT_IRIS, shape)
            l_min_x, l_max_x, l_min_y, l_max_y = get_eye_bounds(landmarks, LEFT_EYE, shape)
            l_norm_x = (l_iris[0] - l_min_x) / (l_max_x - l_min_x + 1e-6)
            l_norm_y = (l_iris[1] - l_min_y) / (l_max_y - l_min_y + 1e-6)

            r_iris = get_iris_center(landmarks, RIGHT_IRIS, shape)
            r_min_x, r_max_x, r_min_y, r_max_y = get_eye_bounds(landmarks, RIGHT_EYE, shape)
            r_norm_x = (r_iris[0] - r_min_x) / (r_max_x - r_min_x + 1e-6)
            r_norm_y = (r_iris[1] - r_min_y) / (r_max_y - r_min_y + 1e-6)

            avg_norm_x = (l_norm_x + r_norm_x) / 2
            avg_norm_y = (l_norm_y + r_norm_y) / 2

            screen_x = int(avg_norm_x * SCREEN_W)
            screen_y = int(avg_norm_y * SCREEN_H)

            # Get current scroll position
            scroll_y = browser.execute_script("return window.scrollY;")
            current_text, text_len = get_word_at_gaze(screen_x, screen_y, active_word_map, scroll_y, off_x, off_y)

        writer.writerow([time.time(), face_detected, avg_norm_x, avg_norm_y, screen_x, screen_y, current_text, text_len])

        cv2.imshow('Selenium Gaze Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

browser.quit()
cap.release()
cv2.destroyAllWindows()