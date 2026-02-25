"""
camera.py
─────────
Step 3: MediaPipe hand landmark detection + C feature extraction

Opens your webcam, detects hand landmarks in real time using
MediaPipe Tasks API (0.10+), feeds them into libhandmath.so
via the bridge, and draws everything on screen.

Run with:
    python camera.py

Controls:
    Q — quit
    S — save a snapshot of current features to features_snapshot.npy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import pathlib
import urllib.request

try:
    from python import HandMathLib
    #remove at some point try excepeepet
except ImportError:
    print("Failed to load C")

else:
    print("C loaded fucking perfectlyllllla")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ─────────────────────────────────────────────────────────────
# Model download (runs once, cached locally)
# ─────────────────────────────────────────────────────────────

MODEL_PATH = pathlib.Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading MediaPipe hand model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded")
    else:
        print(f"✅ Model found: {MODEL_PATH}")


# red green azul

COLOR_LABEL = (255, 255, 255)
COLOR_ANGLES = (100, 220, 255)
COLOR_DISTS = (255, 180, 100)
COLOR_FPS = (180, 255, 100)
COLOR_WARNING = (0, 100, 255)
COLOR_TIP = (0, 255, 180)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
FINGERTIPS = [4, 8, 12, 16, 20]


# Draw helpers
def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (120, 120, 120), 2)
    for i, (x, y) in enumerate(pts):
        color = COLOR_TIP if i in FINGERTIPS else (80, 200, 80)
        radius = 8 if i in FINGERTIPS else 5
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)


def draw_feature_panel(frame, features: np.ndarray, hand_label: str):
    h, w = frame.shape[:2]
    panel_x = w - 255
    angles = np.degrees(features[:15])
    dists = features[15:30]

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 10), (w - 5, h - 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 30
    cv2.putText(frame, f"Hand: {hand_label}", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1)

    y += 25
    cv2.putText(frame, "Joint Angles (deg)", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_ANGLES, 1)

    joint_names = [
        "Th-CMC","Th-MCP","Th-IP",
        "Ix-MCP","Ix-PIP","Ix-DIP",
        "Md-MCP","Md-PIP","Md-DIP",
        "Rg-MCP","Rg-PIP","Rg-DIP",
        "Pk-MCP","Pk-PIP","Pk-DIP",
    ]
    for name, ang in zip(joint_names, angles):
        y += 17
        bar = int((ang / 180.0) * 80)
        cv2.rectangle(frame, (panel_x, y-10), (panel_x+bar, y-2), COLOR_ANGLES, -1)
        cv2.putText(frame, f"{name}: {ang:5.1f}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_LABEL, 1)

    y += 22
    cv2.putText(frame, "Distances (norm)", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DISTS, 1)

    dist_names = [
        "Th-Ix","Th-Md","Th-Rg","Th-Pk",
        "Ix-Md","Ix-Rg","Ix-Pk",
        "Md-Rg","Md-Pk","Rg-Pk",
        "Wr-Th","Wr-Ix","Wr-Md","Wr-Rg","Wr-Pk",
    ]
    max_d = dists.max() if dists.max() > 0 else 1.0
    for name, d in zip(dist_names, dists):
        y += 17
        bar = int((d / max_d) * 80)
        cv2.rectangle(frame, (panel_x, y-10), (panel_x+bar, y-2), COLOR_DISTS, -1)
        cv2.putText(frame, f"{name}: {d:.3f}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_LABEL, 1)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FPS, 2)


# Landmark adapter: Tasks API list → bridge-compatible object
class LandmarkAdapter:
    """Makes a Tasks API landmark list look like the old solutions
    API so hand_math_bridge._to_c_landmarks() works unchanged."""
    def __init__(self, lm_list):
        self.landmark = lm_list


# Main loop 4 camera

def run(camera_index: int = 0):
    ensure_model()
    lib = HandMathLib()

    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("─" * 50)
    print(" Camera running.")
    print(" Q — quit")
    print(" S — snapshot features → features_snapshot.npy")
    print("─" * 50)

    last_features = None
    prev_time = time.time()
    fps = 0.0
    frame_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_ms += 33
        result = detector.detect_for_video(mp_image, frame_ms)

        if result.hand_landmarks:
            for hand_idx, lm_list in enumerate(result.hand_landmarks):
                hand_label = "?"
                if result.handedness:
                    hand_label = result.handedness[hand_idx][0].display_name

                draw_skeleton(frame, lm_list)

                features = lib.extract_features(LandmarkAdapter(lm_list))
                last_features = features

                draw_feature_panel(frame, features, hand_label)
        else:
            h, w = frame.shape[:2]
            cv2.putText(frame, "Show your hand to the camera",
                        (w//2 - 170, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WARNING, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        cv2.putText(frame, "Q: quit S: snapshot",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.imshow("Sign Language — Step 3", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('s') and last_features is not None:
            path = pathlib.Path("features_snapshot.npy")
            np.save(path, last_features)
            print(f"✅ (-> see that is a emojijijijij) (dont ask i was looking online to tell me somthign and i found a cool emoji website that e=needs to be kept in my locer )Snapshot → {path.resolve()}")
            print(f" Angles (deg): {np.degrees(last_features[:15]).round(1)}")
            print(f" Dists: {last_features[15:30].round(3)}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    run(camera_index=0)