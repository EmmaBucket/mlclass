import cv2
import time
import csv
import os
import mediapipe as mp
import numpy as np

user_id = input("Enter username: ")
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
base_filename = f"gaze_data_{user_id}_{timestamp_str}.csv"
filename = base_filename
counter = 1

GRID_SIZE = 3
heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
gaze_events_filename = f"gaze_events_{user_id}_{timestamp_str}.csv"
gaze_events_file = open(gaze_events_filename, "w", newline="")
gaze_events_writer = csv.writer(gaze_events_file)
gaze_events_writer.writerow(["timestamp", "gaze_x", "gaze_y", "norm_x", "norm_y"])

while os.path.exists(filename):
    filename = f"gaze_data_{user_id}_{timestamp_str}_{counter}.csv"
    counter += 1

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 133]
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_EYE = [362, 263]

GRID_SIZE = 3  # 3x3 grid

def get_iris_center(landmarks, indices, shape):
    points = np.array([(int(landmarks[i].x * shape[1]), int(landmarks[i].y * shape[0])) for i in indices])
    center = np.mean(points, axis=0)
    return center

def get_eye_bounds(landmarks, indices, shape):
    points = np.array([(int(landmarks[i].x * shape[1]), int(landmarks[i].y * shape[0])) for i in indices])
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
    return min_x, max_x, min_y, max_y

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "user_id", "timestamp", "frame_number", "face_detected",
        "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
        "left_norm_x", "left_norm_y", "right_norm_x", "right_norm_y",
        "gaze_x", "gaze_y"
    ])

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        gaze_x, gaze_y = -1, -1
        left_iris_center = right_iris_center = [np.nan, np.nan]
        left_norm_x = left_norm_y = right_norm_x = right_norm_y = np.nan
        face_detected = 0

        if results.multi_face_landmarks:
            face_detected = 1
            landmarks = results.multi_face_landmarks[0].landmark
            shape = frame.shape

            left_iris_center = get_iris_center(landmarks, LEFT_IRIS, shape)
            left_min_x, left_max_x, left_min_y, left_max_y = get_eye_bounds(landmarks, LEFT_EYE, shape)
            left_norm_x = (left_iris_center[0] - left_min_x) / (left_max_x - left_min_x + 1e-6)
            left_norm_y = (left_iris_center[1] - left_min_y) / (left_max_y - left_min_y + 1e-6)

            right_iris_center = get_iris_center(landmarks, RIGHT_IRIS, shape)
            right_min_x, right_max_x, right_min_y, right_max_y = get_eye_bounds(landmarks, RIGHT_EYE, shape)
            right_norm_x = (right_iris_center[0] - right_min_x) / (right_max_x - right_min_x + 1e-6)
            right_norm_y = (right_iris_center[1] - right_min_y) / (right_max_y - right_min_y + 1e-6)

            avg_norm_x = (left_norm_x + right_norm_x) / 2
            avg_norm_y = (left_norm_y + right_norm_y) / 2

            gaze_x = int(avg_norm_x * GRID_SIZE)
            gaze_y = int(avg_norm_y * GRID_SIZE)
            gaze_x = min(max(gaze_x, 0), GRID_SIZE - 1)
            gaze_y = min(max(gaze_y, 0), GRID_SIZE - 1)

            

            cv2.circle(frame, (int(left_iris_center[0]), int(left_iris_center[1])), 2, (0,255,0), -1)
            cv2.rectangle(frame, (left_min_x, left_min_y), (left_max_x, left_max_y), (255,0,0), 1)
            cv2.circle(frame, (int(right_iris_center[0]), int(right_iris_center[1])), 2, (0,255,0), -1)
            cv2.rectangle(frame, (right_min_x, right_min_y), (right_max_x, right_max_y), (255,0,0), 1)

        timestamp = time.time()
        writer.writerow([
            user_id, timestamp, frame_number, face_detected,
            left_iris_center[0], left_iris_center[1],
            right_iris_center[0], right_iris_center[1],
            left_norm_x, left_norm_y, right_norm_x, right_norm_y,
            gaze_x, gaze_y
        ])

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        gaze_x, gaze_y = -1, -1  # Default: not detected

        if 0 <= gaze_x < GRID_SIZE and 0 <= gaze_y < GRID_SIZE:
            heatmap[gaze_y, gaze_x] += 1
            gaze_events_writer.writerow([time.time(), gaze_x, gaze_y, avg_norm_x, avg_norm_y])


        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            shape = frame.shape

            # Left eye
            left_iris_center = get_iris_center(landmarks, LEFT_IRIS, shape)
            left_min_x, left_max_x, left_min_y, left_max_y = get_eye_bounds(landmarks, LEFT_EYE, shape)
            left_norm_x = (left_iris_center[0] - left_min_x) / (left_max_x - left_min_x + 1e-6)
            left_norm_y = (left_iris_center[1] - left_min_y) / (left_max_y - left_min_y + 1e-6)

            # Right eye
            right_iris_center = get_iris_center(landmarks, RIGHT_IRIS, shape)
            right_min_x, right_max_x, right_min_y, right_max_y = get_eye_bounds(landmarks, RIGHT_EYE, shape)
            right_norm_x = (right_iris_center[0] - right_min_x) / (right_max_x - right_min_x + 1e-6)
            right_norm_y = (right_iris_center[1] - right_min_y) / (right_max_y - right_min_y + 1e-6)

            # Average normalized positions
            avg_norm_x = (left_norm_x + right_norm_x) / 2
            avg_norm_y = (left_norm_y + right_norm_y) / 2

            # Map to grid
            gaze_x = int(avg_norm_x * GRID_SIZE)
            gaze_y = int(avg_norm_y * GRID_SIZE)
            gaze_x = min(max(gaze_x, 0), GRID_SIZE - 1)
            gaze_y = min(max(gaze_y, 0), GRID_SIZE - 1)

            # Visualization
            cv2.circle(frame, (int(left_iris_center[0]), int(left_iris_center[1])), 2, (0,255,0), -1)
            cv2.rectangle(frame, (left_min_x, left_min_y), (left_max_x, left_max_y), (255,0,0), 1)
            cv2.circle(frame, (int(right_iris_center[0]), int(right_iris_center[1])), 2, (0,255,0), -1)
            cv2.rectangle(frame, (right_min_x, right_min_y), (right_max_x, right_max_y), (255,0,0), 1)

        timestamp = time.time()
        writer.writerow([user_id, timestamp, "frame_captured", gaze_x, gaze_y])

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting webcam...")
            break

cap.release()
cv2.destroyAllWindows()

heatmap_filename = "gaze_heatmap.csv"
with open(heatmap_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Heatmap of gaze grid. Each cell shows fixation count. Try to guess the reading pattern!"])
    header = [f"C{col+1}" for col in range(GRID_SIZE)]
    writer.writerow([""] + header)
    for row in range(GRID_SIZE):
        row_label = f"R{row+1}"
        writer.writerow([row_label] + list(heatmap[row]))

gaze_events_file.close()