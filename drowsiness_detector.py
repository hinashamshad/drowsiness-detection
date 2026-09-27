import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading

# ============ CONFIG ============
EAR_THRESHOLD = 0.21
DROWSY_TIME = 2.0
MAR_THRESHOLD = 0.6          # Mouth open threshold
YAWN_TIME = 1.5              # Yawn kitne sec chale
ALARM_FREQ = 2500
ALARM_DURATION = 300

# ============ MEDIAPIPE ============
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks (top, bottom, left, right)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# ============ FUNCTIONS ============
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_landmarks):
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    h = euclidean(p1, p4)
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_landmarks):
    """MAR = vertical / horizontal"""
    top, bottom, left, right = mouth_landmarks
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / horizontal

def play_alarm():
    for _ in range(3):
        winsound.Beep(ALARM_FREQ, ALARM_DURATION)
        time.sleep(0.05)

# ============ MAIN LOOP ============
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(3))
cam_h = int(cap.get(4))

eyes_closed_start = None
yawn_start = None
alarm_active = False
prev_time = 0
drowsy_count = 0
yawn_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NO FACE"
    status_color = (128, 128, 128)
    ear_value = 0.0
    mar_value = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * cam_w)
                y = int(lm.y * cam_h)
                landmarks.append((x, y))

            # Calculate EAR
            left_pts = [landmarks[i] for i in LEFT_EYE]
            right_pts = [landmarks[i] for i in RIGHT_EYE]
            ear_value = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

            # Calculate MAR
            mouth_pts = [
                landmarks[MOUTH_TOP],
                landmarks[MOUTH_BOTTOM],
                landmarks[MOUTH_LEFT],
                landmarks[MOUTH_RIGHT]
            ]
            mar_value = calculate_mar(mouth_pts)

            # Draw eye landmarks
            for pt in left_pts + right_pts:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # Draw mouth landmarks
            for pt in mouth_pts:
                cv2.circle(frame, pt, 3, (255, 0, 255), -1)

            current_time = time.time()

            # ============ DROWSINESS (EYE) ============
            if ear_value < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time
                closed_duration = current_time - eyes_closed_start
                if closed_duration >= DROWSY_TIME:
                    status = "DROWSY! WAKE UP!"
                    status_color = (0, 0, 255)
                    if not alarm_active:
                        alarm_active = True
                        drowsy_count += 1
                        threading.Thread(target=play_alarm, daemon=True).start()
                else:
                    status = f"EYES CLOSED: {closed_duration:.1f}s"
                    status_color = (0, 165, 255)
            else:
                eyes_closed_start = None
                alarm_active = False
                status = "AWAKE"
                status_color = (0, 255, 0)

            # ============ YAWN (MOUTH) ============
            if mar_value > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = current_time
                yawn_duration = current_time - yawn_start
                if yawn_duration >= YAWN_TIME:
                    yawn_count += 1
                    yawn_start = None  # Reset to avoid double count
                    cv2.putText(frame, "YAWNING!", (30, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    cv2.putText(frame, f"MOUTH OPEN: {yawn_duration:.1f}s", (30, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            else:
                yawn_start = None

    # ============ UI ============
    cv2.putText(frame, f"EAR: {ear_value:.3f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar_value:.3f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(frame, status, (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.putText(frame, f"Alerts: {drowsy_count}  Yawns: {yawn_count}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (cam_w - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if status == "DROWSY! WAKE UP!":
        cv2.rectangle(frame, (5, 5), (cam_w - 5, cam_h - 5), (0, 0, 255), 8)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()