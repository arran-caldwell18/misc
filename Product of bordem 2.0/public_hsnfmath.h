#ifndef HAND_MATH_H
#define HAND_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// 21 tiny landmarks floating in space
#define NUM_LANDMARKS 21

// 63 numbers come out eventually
// angles, distances, maybe future chaos
#define NUM_FEATURES  63

// one point in 3D. x,y,z because life is 3D
typedef struct {
    float x, y, z;
} Landmark;

// 21 points make a hand (probably)
typedef struct {
    Landmark pts[NUM_LANDMARKS];  // wrist is 0, fingers go wild with other nums
} HandLandmarks;

// the sacred 63-float feature vector
typedef struct {
    float features[NUM_FEATURES];  // angles + distances + ghosts buaters
} FeatureVec;

// ---------------- NORMALIZE THE HAND OO DTEAH ------------------
// move wrist to origin
// scale so hand isnt huge or tiny
// ML gets less angry
void normalize_landmarks(const HandLandmarks* in, HandLandmarks* out);

// -------------------- ANGLES TIME ----------------------------
// parent, joint, child triplets
// returns radians. NOT degrees. do not convert. seriously.
void compute_angles(const HandLandmarks* hand, float* angles, int* count);

// ---------------- DISTANCES ------------------------------
// fingertips and wrist and other magic pairs
// 3D pythag moment
void compute_distances(const HandLandmarks* hand, float* dists, int* count);

// ---------------- BIG FEATURE PIPELINE ------------------
// raw landmarks -> normalized -> angles + distances -> 63 floats
// ML happy. ghosts removed. future you will hate me
void extract_features(const HandLandmarks* raw, FeatureVec* features);

// ---------------- MATH UTILITIES ------------------------

// multiply stuff, add stuff, done. (dot product)
float dot3(float ax, float ay, float az,
           float bx, float by, float bz);

// sqrt(x*x+y*y+z*z) kinda works
float mag3(float x, float y, float z);

// angle between 2 vectors in radians. NOT degrees. AGAIN.
float angle_between(float ax, float ay, float az,
                    float bx, float by, float bz);

#ifdef __cplusplus
}
#endif

#endif // HAND_MATH_H
