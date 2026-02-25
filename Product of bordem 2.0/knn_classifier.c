/*
    hand_math.c
    ────────────────────────────────────────────────
    Ultra Serious Scientific Hand Geometry Engine™

    Turns 21 tiny float triplets into:
        - 15 joint angles
        - 15 normalized distances

    Because computers deserve to understand fingers too.

    Compile:
        gcc -shared -fPIC -O3 hand_math.c -o libhandmath.so -lm
*/

#include <math.h>
#include <stdio.h>

/* ────────────────────────────────────────────────
   Tiny vector struct (because we are civilized)
   ──────────────────────────────────────────────── */
typedef struct {
    float x;
    float y;
    float z;
} Vec3;


/* ────────────────────────────────────────────────
   Vector utilities (a.k.a. finger physics)
   ──────────────────────────────────────────────── */

// Subtract two vectors: a - b
static Vec3 v_sub(Vec3 a, Vec3 b) {
    Vec3 r = {a.x - b.x, a.y - b.y, a.z - b.z};
    return r;
}

// Dot product (how much two vectors vibe together)
static float v_dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Vector length (aka emotional magnitude)
static float v_len(Vec3 v) {
    return sqrtf(v_dot(v, v));
}

// Safe normalize (avoids existential division by zero)
static Vec3 v_norm(Vec3 v) {
    float len = v_len(v);
    if (len < 1e-6f) {
        Vec3 zero = {0,0,0};
        return zero; // this vector has chosen peace
    }
    Vec3 r = {v.x/len, v.y/len, v.z/len};
    return r;
}

// Angle between two vectors (in radians)
static float angle_between(Vec3 a, Vec3 b) {
    Vec3 na = v_norm(a);
    Vec3 nb = v_norm(b);
    float d = v_dot(na, nb);

    // Clamp to stop acos from having a meltdown
    if (d > 1.0f) d = 1.0f;
    if (d < -1.0f) d = -1.0f;

    return acosf(d);
}


/* ────────────────────────────────────────────────
   Finger Joint Definitions
   Because fingers are organized chaos
   ──────────────────────────────────────────────── */

/*
Landmark index map (MediaPipe style):

0  = wrist
1-4  = thumb
5-8  = index
9-12 = middle
13-16= ring
17-20= pinky
*/


// Each angle defined by three indices: (a, b, c)
// Angle at point b between segments a-b and c-b
static const int ANGLE_TRIPLETS[15][3] = {
    // Thumb
    {1, 2, 3},
    {2, 3, 4},
    {0, 1, 2},

    // Index
    {5, 6, 7},
    {6, 7, 8},
    {0, 5, 6},

    // Middle
    {9,10,11},
    {10,11,12},
    {0, 9,10},

    // Ring
    {13,14,15},
    {14,15,16},
    {0,13,14},

    // Pinky
    {17,18,19},
    {18,19,20},
    {0,17,18}
};


// Distance pairs (because fingers like measuring things)
static const int DIST_PAIRS[15][2] = {
    {4,8}, {4,12}, {4,16}, {4,20},
    {8,12}, {8,16}, {8,20},
    {12,16}, {12,20}, {16,20},
    {0,4}, {0,8}, {0,12}, {0,16}, {0,20}
};


/* ────────────────────────────────────────────────
   Main Feature Extraction Function
   The Star of the Show
   ──────────────────────────────────────────────── */

/*
Input:
    landmarks[63] = 21 * (x,y,z)

Output:
    out_features[30]
        0-14  = angles (radians)
        15-29 = normalized distances
*/

void extract_features(float landmarks[63], float out_features[30]) {

    // Step 1: Convert raw float soup into Vec3s
    Vec3 pts[21];
    for (int i = 0; i < 21; i++) {
        pts[i].x = landmarks[i*3 + 0];
        pts[i].y = landmarks[i*3 + 1];
        pts[i].z = landmarks[i*3 + 2];
    }

    /* ───────────────────────────────────────────
       Compute 15 joint angles
       ─────────────────────────────────────────── */
    for (int i = 0; i < 15; i++) {

        int a = ANGLE_TRIPLETS[i][0];
        int b = ANGLE_TRIPLETS[i][1];
        int c = ANGLE_TRIPLETS[i][2];

        // Build two bone vectors meeting at joint b
        Vec3 ba = v_sub(pts[a], pts[b]);
        Vec3 bc = v_sub(pts[c], pts[b]);

        float ang = angle_between(ba, bc);

        out_features[i] = ang;  // radians of glory
    }

    /* ───────────────────────────────────────────
       Compute 15 distances
       ─────────────────────────────────────────── */
    float max_dist = 0.0f;

    for (int i = 0; i < 15; i++) {
        int a = DIST_PAIRS[i][0];
        int b = DIST_PAIRS[i][1];

        Vec3 d = v_sub(pts[a], pts[b]);
        float dist = v_len(d);

        out_features[15 + i] = dist;

        if (dist > max_dist)
            max_dist = dist; // the alpha distance
    }

    // Normalize distances so giant hands don’t flex unfairly
    if (max_dist < 1e-6f)
        max_dist = 1.0f; // avoid divide-by-zero apocalypse

    for (int i = 0; i < 15; i++) {
        out_features[15 + i] /= max_dist;
    }

    /*
        At this point:
        - out_features[0..14]  = angles (radians)
        - out_features[15..29] = normalized distances

        You may now teach your computer sign language.
        Or confuse it.
        Both are valid.
    */
}