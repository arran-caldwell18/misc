"""
collect_data.py
────────────────
Step 4: Collect labeled hand landmark features for each ASL letter.

This script opens your camera, judges your hand,
and stores 30 tiny floating-point secrets per letter.

Hold a letter sign.
Press that letter key.
Repeat until you question your life choices.

All samples saved to:
    data/training_data.csv

Run with:
    python collect_data.py

Controls:
    A-Z       → save sample for that letter
    BACKSPACE → undo last mistake
    SPACE     → freeze frame (spam the same pose like a maniac)
    Q         → quit and face your dataset
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import pathlib
import urllib.request
from collections import Counter

from python import HandMathLib
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision



# Config (numbers that determine your destiny)

DATA_DIR = pathlib.Path("data")  # where your precious CSV lives
CSV_PATH = DATA_DIR / "training_data.csv"

MODEL_PATH = pathlib.Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

SAMPLES_GOAL = 30   # per letter. yes all 26. yes you signed up for this.
NUM_FEATURES = 30   # 15 angles + 15 distances (active features only)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # the alphabet. all of it.

# Colors (because raw vision is boring)
C_WHITE = (255, 255, 255)
C_GREEN = (100, 255, 150)
C_YELLOW = (0, 220, 255)
C_RED = (60, 60, 220)
C_CYAN = (255, 220, 100)
C_GRAY = (160, 160, 160)
C_DARK = (30, 30, 30)

# which dots connect to which other dots (hand skeleton conspiracy map)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),#it feels like somthings ment to go herere
]

FINGERTIPS = [4, 8, 12, 16, 20]  # dramatic dots


# CSV helpers (spreadsheet wizardry)
def load_existing(path: pathlib.Path) -> list:
    """Load existing samples so we don't pretend progress vanished."""
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header because headers are boring
            for row in reader:
                rows.append(row)
    return rows


def append_row(path: pathlib.Path, label: str, features: np.ndarray):
    """
    Append one labeled feature vector to the CSV.
    Aka: immortalize this exact hand pose forever.
    """
    DATA_DIR.mkdir(exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            header = ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]
            writer.writerow(header)

        writer.writerow([label] + features[:NUM_FEATURES].tolist())


def delete_last_row(path: pathlib.Path) -> str | None:
    """
    Remove the last saved row.
    Because mistakes happen.
    And BACKSPACE exists for a reason.
    """
    if not path.exists():
        return None

    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return None

    deleted_label = rows[-1][0]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows[:-1])

    return deleted_label



# Draw helpers (UI that pretends this is polished software)

def draw_skeleton(frame, landmarks):
    """
    Draw 21 points and connect them like a tiny bone diagram.
    Makes you feel like a computer vision wizard.
    """
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (100, 100, 100), 2)

    for i, (x, y) in enumerate(pts):
        color = C_CYAN if i in FINGERTIPS else C_GREEN
        radius = 8 if i in FINGERTIPS else 5
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, C_WHITE, 1)


def draw_big_letter(frame, letter: str):
    """
    Display the letter you just saved.
    Dramatic. Triumphant. Slightly aggressive.
    """
    h, w = frame.shape[:2]
    cv2.putText(frame, letter, (w // 2 - 200, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 150), 12)
    cv2.putText(frame, letter, (w // 2 - 200, h // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 255, 255), 3)



# Main loop (where everything either works or collapses)

def runtsbackkkkkkkkkkkkkkkkk(camera_index: int = 0): #lil tt referebcea
    """
    Opens camera.
    Detects hand.
    Extracts features.
    Waits for keyboard input.
    Repeats until Q or existential crisis.
    """

    # download model if missing because we believe in automation
    if not MODEL_PATH.exists():
        print(f"Downloading model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    lib = HandMathLib()  # the C bridge wizard

    existing = load_existing(CSV_PATH)
    counts = Counter(row[0] for row in existing)

    print(f"Loaded {len(existing)} existing samples.")
    print("Hold a letter sign and press that key.")
    print("SPACE = freeze. BACKSPACE = undo. Q = quit.")

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
        raise RuntimeError("Camera refused to cooperate.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frozen = False
    frozen_frame = None
    current_features = None

    frame_ms = 0

    while True:
        if frozen and frozen_frame is not None:
            frame = frozen_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_ms += 33
            result = detector.detect_for_video(mp_image, frame_ms)

            current_features = None

            if result.hand_landmarks:
                lm_list = result.hand_landmarks[0]
                draw_skeleton(frame, lm_list)
                current_features = lib.extract_features(lm_list)
            else:
                cv2.putText(frame, "No hand detected",
                            (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 80, 255), 2)

        cv2.imshow("ASL Data Collector — Step 4", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit immediately before things escalate
        if key == ord('q'):
            print("Exiting. Dataset preserved.")
            break

        # save letter
        elif key < 128 and chr(key).upper() in LETTERS:
            letter = chr(key).upper()

            if current_features is not None:
                append_row(CSV_PATH, letter, current_features)
                counts[letter] += 1
                print(f"Saved '{letter}' → total {counts[letter]}")
            else:
                print("No hand detected. Try again.")

        # undo
        elif key == 8:
            deleted = delete_last_row(CSV_PATH)
            if deleted:
                counts[deleted] -= 1
                print(f"Deleted last sample ('{deleted}')")
            else:
                print("Nothing to delete.")

        # freeze toggle
        elif key == ord(' '):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()
                print("Frozen. Spam keys responsibly.")
            else:
                frozen_frame = None
                print("Unfrozen.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    runtsbackkkkkkkkkkkkkkkkk(camera_index=0)
    #i dearly regret naming the function that