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

while os.path.exists(filename):
    filename = f"gaze_data_{user_id}_{timestamp_str}_{counter}.csv"
    counter += 1

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Indices for left and right iris and eye contours
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
    writer.writerow(["user_id", "timestamp", "frame_status", "gaze_x", "gaze_y"])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        gaze_x, gaze_y = -1, -1  # Default: not detected

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            shape = frame.shape

            # Left eye
            iris_center = get_iris_center(landmarks, LEFT_IRIS, shape)
            min_x, max_x, min_y, max_y = get_eye_bounds(landmarks, LEFT_EYE, shape)
            # Normalize iris x position within the eye bounds (0 = left, 1 = right)
            norm_x = (iris_center[0] - min_x) / (max_x - min_x + 1e-6)
            norm_y = (iris_center[1] - min_y) / (max_y - min_y + 1e-6)

            # Map normalized position to grid
            gaze_x = int(norm_x * GRID_SIZE)
            gaze_y = int(norm_y * GRID_SIZE)

            # Clamp to grid
            gaze_x = min(max(gaze_x, 0), GRID_SIZE - 1)
            gaze_y = min(max(gaze_y, 0), GRID_SIZE - 1)

            # Optional: Draw eye and iris for visualization
            cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 2, (0,255,0), -1)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255,0,0), 1)

        timestamp = time.time()
        writer.writerow([user_id, timestamp, "frame_captured", gaze_x, gaze_y])

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting webcam...")
            break

cap.release()
cv2.destroyAllWindows()