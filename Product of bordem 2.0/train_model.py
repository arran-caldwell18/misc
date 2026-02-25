"""
train.py
────────
Step 5: Train a KNN classifier from data/training_data.csv
        and export it into a binary blob the C code can inhale.

This script:
    • Reads your lovingly collected hand angles
    • Normalizes them so geometry behaves
    • Trains a KNN classifier (no gradients, just vibes)
    • Evaluates it with leave-one-out (painfully)
    • Writes everything into knn_model.bin
    • Writes a label_map.txt so C knows what 0 means

Run:
    python train.py

Output:
    data/knn_model.bin  → binary model for C
    data/label_map.txt  → index → letter decoder ring
"""

import csv
import struct
import pathlib
import numpy as np
from collections import Counter

DATA_DIR = pathlib.Path("data")  # where all your sins are stored
CSV_PATH = DATA_DIR / "training_data.csv"
MODEL_PATH = DATA_DIR / "knn_model.bin"
LABEL_PATH = DATA_DIR / "label_map.txt"

K = 5         # how many neighbors get to vote in democracy
NUM_FEAT = 30 # 15 angles + 15 distances (the chosen ones)


# ─────────────────────────────────────────────────────────────
# Load CSV (resurrect the dataset)
# ─────────────────────────────────────────────────────────────

def load_csv(path: pathlib.Path):
    """
    Load training_data.csv into:
        X → (N, 30) float32
        labels → list[str]

    If the file doesn't exist?
    That's between you and collect_data.py.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No training data found at {path}\n"
            f"Run python collect_data.py first."
        )

    labels, features = [], []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header because headers are ceremonial

        for row in reader:
            labels.append(row[0])
            features.append([float(x) for x in row[1:NUM_FEAT + 1]])

    return np.array(features, dtype=np.float32), labels


# ─────────────────────────────────────────────────────────────
# Label mapping (letters → integers → civilization)
# ─────────────────────────────────────────────────────────────

def build_label_map(labels):
    """
    Convert letters into integer class indices.

    Because C does not understand vibes.
    C understands integers.
    """
    unique = sorted(set(labels))
    letter_to_idx = {l: i for i, l in enumerate(unique)}
    idx_to_letter = {i: l for l, i in letter_to_idx.items()}
    return letter_to_idx, idx_to_letter


# ─────────────────────────────────────────────────────────────
# Math zone (statistics but approachable)
# ─────────────────────────────────────────────────────────────

def compute_norm_params(X: np.ndarray):
    """
    Compute feature-wise mean and std.

    We normalize because distance-based classifiers
    get emotional when scales are inconsistent.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # if a feature never changes, don't divide by zero and implode
    std[std < 1e-6] = 1.0

    return mean, std


def normalize(X, mean, std):
    """Classic z-score normalization. No magic. Just math."""
    return (X - mean) / std



# KNN logic (no training phase, just stored memories)
# ───────────────────────────────────────────────────

def knn_predict(X_train, y_train, x_query, k):
    """
    Predict label of x_query using:
        • Euclidean distance
        • k nearest neighbors
        • majority vote

    KNN is just:
        "who are your closest friends?"
    """
    dists = np.linalg.norm(X_train - x_query, axis=1)
    top_k = np.argsort(dists)[:k]
    votes = Counter(y_train[i] for i in top_k)
    return votes.most_common(1)[0][0]


def evaluate(X_norm, y_int, idx_to_letter, k):
    """
    Leave-One-Out Cross Validation.

    For each sample:
        Remove it.
        Predict it.
        Judge it.
        Repeat N times.

    Yes, this is O(N²).
    Yes, we accept that.
    """
    correct = 0
    n = len(X_norm)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        pred = knn_predict(
            X_norm[mask],
            y_int[mask],
            X_norm[i],
            k
        )

        if pred == y_int[i]:
            correct += 1

    return correct / n


# Binary export format (C-compatibilty ritual)

#
# Format (little-endian):
# [4]  magic "KNN\0"
# [4]  n_samples int32
# [4]  n_features int32
# [4]  n_classes int32
# [4]  k int32
# [f*4] mean float32[]
# [f*4] std float32[]
# [n*f*4] X_norm float32[]
# [n*4] y_int int32[]
#
# If this layout changes and C doesn't know?
# Catastrophic sadn:(ess.
# ─────────────────────────────────────────────────────────────

def export_model(path, X_norm, y_int, mean, std, n_classes, k):
    """
    Write model into a binary format the C classifier understands.

    We are manually controlling memory layout.
    This is power.
    """
    n, f = X_norm.shape

    with open(path, "wb") as fp:
        fp.write(b"KNN\x00")
        fp.write(struct.pack("<iiii", n, f, n_classes, k))
        fp.write(mean.astype(np.float32).tobytes())
        fp.write(std.astype(np.float32).tobytes())
        fp.write(X_norm.astype(np.float32).tobytes())
        fp.write(np.array(y_int, dtype=np.int32).tobytes())

    print(f"🔥 Model saved → {path} ({path.stat().st_size} bytes)")


def export_label_map(path, idx_to_letter):
    """
    Write index → letter mapping.

    Because when C predicts class 7,
    someone needs to know that means 'H'.
    """
    with open(path, "w") as f:
        for i in sorted(idx_to_letter):
            f.write(f"{i} {idx_to_letter[i]}\n")

    print(f"🧠 Label map saved → {path}")


# Main 
# ──────

def main():
    print("═" * 50)
    print(" Step 5 — Training KNN Classifier")
    print("═" * 50)

    # Load dataset
    X_raw, labels = load_csv(CSV_PATH)
    counts = Counter(labels)

    print(f"\nLoaded {len(labels)} samples across {len(counts)} classes")
    print(f"Samples per letter: {dict(sorted(counts.items()))}")

    if len(counts) < 2:
        print("⚠ Need at least 2 letters to train. One letter is just vibes.")
        return

    # Encode labels into integers
    letter_to_idx, idx_to_letter = build_label_map(labels)
    y_int = np.array([letter_to_idx[l] for l in labels], dtype=np.int32)

    # Normalize features
    mean, std = compute_norm_params(X_raw)
    X_norm = normalize(X_raw, mean, std)

    # Evaluate model
    print(f"\nRunning leave-one-out evaluation (k={K}) ...")
    acc = evaluate(X_norm, y_int, idx_to_letter, K)
    print(f" Leave-one-out accuracy: {acc * 100:.1f}%")

    if acc < 0.7:
        print("Accuracy below 70% — collect more samples, warrior.")
    elif acc < 0.9:
        print("Decent accuracy — but we hunger for more data.")
    else:
        print(" Great accuracy. Geometry obeys you.")

    # Per-letter breakdown
    print("\n Per-letter accuracy:")
    for letter in sorted(counts):
        idx = letter_to_idx[letter]
        mask = y_int == idx

        X_l = X_norm[mask]
        y_l = y_int[mask]

        correct = 0

        X_others = X_norm[~mask]
        y_others = y_int[~mask]

        for i in range(len(X_l)):
            X_train = np.vstack([
                X_others,
                X_l[np.arange(len(X_l)) != i]
            ])
            y_train = np.concatenate([
                y_others,
                y_l[np.arange(len(y_l)) != i]
            ])

            pred = knn_predict(X_train, y_train, X_l[i], K)

            if pred == idx:
                correct += 1

        la = correct / len(X_l) if len(X_l) > 0 else 0
        bar = "█" * int(la * 20)
        print(f" {letter}: {la*100:5.1f}% {bar}")

    # Export model
    print()
    DATA_DIR.mkdir(exist_ok=True)

    export_model(
        MODEL_PATH,
        X_norm,
        y_int,
        mean,
        std,
        n_classes=len(idx_to_letter),
        k=K
    )

    export_label_map(LABEL_PATH, idx_to_letter)

    print("\nReady for Step 6 — C classifier will consume this binary.")
    print("═" * 50)


if __name__ == "__main__":
    main()