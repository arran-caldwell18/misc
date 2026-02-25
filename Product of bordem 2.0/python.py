# ik """""" is like bad practise or somthing but yeah im doing itttttttttttttttttttttttagggggggggggggggf help its MIDNIGHHHHHT
"""
hand_math_bridge.py
───────────────────
Python ↔ C bridge for libhandmath.so

This loads the scary .so file with ctypes so the rest of
the project never has to look at ctypes ever again
because ctypes is pain.

Usage:
    from hand_math_bridge import HandMathLib

    lib = HandMathLib()  # loads the .so (pray it exists)
    features = lib.extract_features(landmarks)
    print(features)  # numpy array, shape (63,) very fancy
"""

import ctypes
import pathlib
import numpy as np


# ─────────────────────────────────────────────────────────────
# C struct mirrors (THESE MUST MATCH THE HEADER OR EVERYTHING DIES)
# ─────────────────────────────────────────────────────────────

class CLandmark(ctypes.Structure):
    """typedef struct { float x, y, z; } Landmark; but in python cosplay"""
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class CHandLandmarks(ctypes.Structure):
    """typedef struct { Landmark pts[21]; } HandLandmarks; aka 21 little dudes"""
    _fields_ = [
        ("pts", CLandmark * 21),
    ]


class CFeatureVec(ctypes.Structure):
    """typedef struct { float features[63]; } FeatureVec; big float bucket"""
    _fields_ = [
        ("features", ctypes.c_float * 63),
    ]


# ─────────────────────────────────────────────────────────────
# Bridge Class (python pretending to be C)
# ─────────────────────────────────────────────────────────────

class HandMathLib:
    """
    Wraps libhandmath.so so we can pretend everything is clean
    and object-oriented and not terrifying.

    Parame:ters::


    lib_path : optional path to libhandmath.so
        If you don't give one it looks next to this file.
        If it isn't there? skill issue.
    """

    def __init__(self, lib_path: str | pathlib.Path | None = None):
        if lib_path is None:
            # look next to this file because that is convenient
            here = pathlib.Path(__file__).parent
            lib_path = here / "libhandmath.so"

        lib_path = pathlib.Path(lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Could not find {lib_path}\n"
                f"Run 'make' first before crying."
            )

        # summon the shared object from the abyss
        self._lib = ctypes.CDLL(str(lib_path))
        self._bind_functions()
        print(f"✅ Loaded {lib_path} (no explosions detected)")

    # ── tell ctypes what the C functions look like or chaos happens ──

    def _bind_functions(self):
        lib = self._lib

        # void normalize_landmarks(const HandLandmarks* in, HandLandmarks* out)
        # moves wrist to origin, scales hand, makes ML less angy
        lib.normalize_landmarks.restype = None
        lib.normalize_landmarks.argtypes = [
            ctypes.POINTER(CHandLandmarks),
            ctypes.POINTER(CHandLandmarks),
        ]

        # void compute_angles(...)
        # fills 15 angles into an array. radians. NOT degrees.
        lib.compute_angles.restype = None
        lib.compute_angles.argtypes = [
            ctypes.POINTER(CHandLandmarks),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]

        # void compute_distances(...)
        # fills 15 distances. 3D pythag moment.
        lib.compute_distances.restype = None
        lib.compute_distances.argtypes = [
            ctypes.POINTER(CHandLandmarks),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]

        # void extract_features(...)
        # big pipeline. raw in. 63 floats out. ML happy.
        lib.extract_features.restype = None
        lib.extract_features.argtypes = [
            ctypes.POINTER(CHandLandmarks),
            ctypes.POINTER(CFeatureVec),
        ]

        # float dot3(...)
        # multiply stuff add stuff done
        lib.dot3.restype = ctypes.c_float
        lib.dot3.argtypes = [ctypes.c_float] * 6

        # float mag3(...)
        # sqrt time. hope sqrt behaves.
        lib.mag3.restype = ctypes.c_float
        lib.mag3.argtypes = [ctypes.c_float] * 3

        # float angle_between(...)
        # radians. again. do not mess it up.
        lib.angle_between.restype = ctypes.c_float
        lib.angle_between.argtypes = [ctypes.c_float] * 6

    # ── convert python landmarks into C struct (ritual transformation) ──

    @staticmethod
    def _to_c_landmarks(landmarks) -> CHandLandmarks:
        """
        Turns whatever you passed in into the sacred CHandLandmarks struct.

        Accepts:
          • list of 21 (x,y,z)
          • numpy array shape (21,3)
          • MediaPipe object (has .landmark)
        """

        # MediaPipe object (has .landmark attribute)
        if hasattr(landmarks, "landmark"):
            landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

        # numpy array? convert to list so ctypes doesnt cry
        if hasattr(landmarks, "shape"):
            landmarks = landmarks.tolist()

        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)} bruh")

        c_hand = CHandLandmarks()

        # copy each tiny xyz into the struct manually because python cant chill
        for i, pt in enumerate(landmarks):
            c_hand.pts[i].x = float(pt[0])
            c_hand.pts[i].y = float(pt[1])
            c_hand.pts[i].z = float(pt[2])

        return c_hand

    # ── PUBLIC API (the only part the rest of the project should see) ──

    def extract_features(self, landmarks) -> np.ndarray:
        """
        Full pipeline:
        raw landmarks → normalize → angles + distances → 63 floats

        Returns numpy array (63,) float32

        [0:15]  angles
        [15:30] distances
        [30:63] empty for future suffering
        """

        c_hand = self._to_c_landmarks(landmarks)
        c_fv = CFeatureVec()

        # call into C land
        self._lib.extract_features(
            ctypes.byref(c_hand),
            ctypes.byref(c_fv),
        )

        # convert C float array back into numpy like nothing happened
        return np.array(list(c_fv.features), dtype=np.float32)

    def normalize(self, landmarks) -> np.ndarray:
        """
        Only normalize. No angles. No distances.
        Just wrist to origin and scale magic.
        """

        c_in = self._to_c_landmarks(landmarks)
        c_out = CHandLandmarks()

        self._lib.normalize_landmarks(
            ctypes.byref(c_in),
            ctypes.byref(c_out),
        )

        result = np.zeros((21, 3), dtype=np.float32)

        # copy back into numpy because ctypes cant stay in its lane
        for i in range(21):
            result[i] = [
                c_out.pts[i].x,
                c_out.pts[i].y,
                c_out.pts[i].z,
            ]

        return result

    def compute_angles(self, landmarks) -> np.ndarray:
        """
        Returns 15 joint angles (radians).
        NOT degrees.
        I am repeating this on purpose.
        """

        c_hand = self._to_c_landmarks(landmarks)
        arr = (ctypes.c_float * 15)()
        count = ctypes.c_int(0)

        self._lib.compute_angles(
            ctypes.byref(c_hand),
            arr,
            ctypes.byref(count),
        )

        return np.array(list(arr[:count.value]), dtype=np.float32)

    def compute_distances(self, landmarks) -> np.ndarray:
        """
        Returns 15 distances between important landmark pairs.
        Shape matters apparently.
        """

        c_hand = self._to_c_landmarks(landmarks)
        arr = (ctypes.c_float * 15)()
        count = ctypes.c_int(0)

        self._lib.compute_distances(
            ctypes.byref(c_hand),
            arr,
            ctypes.byref(count),
        )

        return np.array(list(arr[:count.value]), dtype=np.float32)

    # ── passthrough math helpers ──

    def dot3(self, a, b) -> float:
        """multiply stuff add stuff done (python edition)"""
        return self._lib.dot3(*a, *b)

    def mag3(self, v) -> float:
        """vector length moment"""
        return self._lib.mag3(*v)

    def angle_between(self, a, b) -> float:
        """returns radians. yes still radians."""
        return self._lib.angle_between(*a, *b)