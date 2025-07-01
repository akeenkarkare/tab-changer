import cv2
import mediapipe as mp
import subprocess
import numpy as np
import time

# — CONFIG —
BROWSER     = "Google Chrome"   # Or "Safari"
OPEN_THRESH = 0.25              # normalized distance threshold; tweak as needed
COOLDOWN    = 1                 # seconds between allowed tab actions

# AppleScript snippets
AS_OPEN = f'''
tell application "{BROWSER}"
  tell front window to set active tab index to (active tab index + 1)
end tell
'''
AS_CLOSE = f'''
tell application "{BROWSER}"
  tell front window to set active tab index to (active tab index - 1)
end tell
'''

def open_tab():
    subprocess.run(["osascript", "-e", AS_OPEN], check=False)

def close_tab():
    subprocess.run(["osascript", "-e", AS_CLOSE], check=False)

# — Mediapipe setup —
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(min_detection_confidence=0.8,
                         min_tracking_confidence=0.8,
                         max_num_hands=1)
draw     = mp.solutions.drawing_utils

# indices of landmarks
WRIST   = mp_hands.HandLandmark.WRIST
TIP_IDS = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

last_action = 0.0

print("Starting Open/Close gesture tab control. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm_list = res.multi_hand_landmarks[0].landmark

        # draw the hand skeleton
        draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # compute normalized distances fingertip→wrist
        wrist_pt = np.array([lm_list[WRIST].x, lm_list[WRIST].y])
        dists = []
        for tip_id in TIP_IDS:
            tip_pt = np.array([lm_list[tip_id].x, lm_list[tip_id].y])
            dists.append(np.linalg.norm(tip_pt - wrist_pt))
        avg_dist = float(np.mean(dists))

        # check cooldown
        now = time.time()
        if now - last_action > COOLDOWN:
            if avg_dist > OPEN_THRESH:
                open_tab()
                state, color = "Open Hand → Next Tab", (0, 255, 0)
            else:
                close_tab()
                state, color = "Closed Fist → Previous Tab", (0, 0, 255)
            last_action = now
        else:
            # still cooling down
            state, color = f"Cooling… ({avg_dist:.2f})", (200, 200, 200)

        # overlay state & distance
        cv2.putText(frame,
                    state,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Open/Close Gesture Tab Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
