import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks (MediaPipe FaceMesh indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_landmarks):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    # Vertical distances
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    # Horizontal distance
    h = euclidean(p1, p4)
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

cap = cv2.VideoCapture(0)
cam_w = int(cap.get(3))
cam_h = int(cap.get(4))

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmarks ko pixel coordinates mei convert karo
            landmarks = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * cam_w)
                y = int(lm.y * cam_h)
                landmarks.append((x, y))

            # Left eye EAR
            left_eye_pts = [landmarks[i] for i in LEFT_EYE]
            left_ear = calculate_ear(left_eye_pts)

            # Right eye EAR
            right_eye_pts = [landmarks[i] for i in RIGHT_EYE]
            right_ear = calculate_ear(right_eye_pts)

            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0

            # Eye landmarks pe circles draw karo
            for pt in left_eye_pts + right_eye_pts:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # Screen pe EAR value dikhao
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Status: Open ya Closed
            if avg_ear < 0.20:
                status = "EYES CLOSED"
                color = (0, 0, 255)  # Red
            else:
                status = "EYES OPEN"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("EAR Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()