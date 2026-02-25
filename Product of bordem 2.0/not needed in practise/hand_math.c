#include "public_hsnfmath.h"   // not touching this ok
#include <math.h>
#include <string.h>
#include <stdio.h>

// basci 3d dot thing
// multiply stuff add stuff done
float dot3(float ax,float ay,float az,
           float bx, float by , float bz){
    return ax*bx+ ay*by +az*bz; // mathemaicas heheheeeg    (didnt know what to do so gpt helped with that mthemeacitcasssss)
}

// lenght of 3d vector (prob right)
float mag3(float x,float y,float z){
    return sqrtf(x*x + y*y + z*z);  // hope sqrtf doesnt betray me
}

///angle between two vecotrs
// returns radains NOT degrees dont mess it up
float angle_between(float ax,float ay,float az,
                    float bx,float by,float bz){

    float d = dot3(ax,ay,az,bx,by,bz);
    float ma = mag3(ax,ay,az);
    float mb = mag3(bx,by,bz);

    // if one vec is basically dead
    if(ma < 1e-6f || mb<1e-6f)
        return 0.0f;  // shrug it offfffffffffffffffffffffffffffffffffffffffffffffffffffffffffff

    float cosine = d/(ma*mb);

    // floating point gets funky and then acos crys
    if(cosine > 1.0f) cosine = 1.0f;
    if(cosine < -1.0f) cosine = -1.0f;

    return acosf(cosine); // radains. again. . . . . . . . . 
}


// ---------------- NORMALIZING HAND OR WTEVER ----------------
// move wrist to 0,0,0
// scale so hand isnt giant or tiny
// makes ML less angy
// ---------------------------------------------------------------

void normalize_landmarks(const HandLandmarks* in, HandLandmarks* out){

    // wrist index is 0 (i think? yes.UR CODE SAID SO I TINNK))
    float wx=in->pts[0].x;
    float wy=in->pts[0].y;
    float wz=in->pts[0].z;

    // wrist to middle finger mcp (9)
    // random but works decent
    float hx = in->pts[9].x - wx;
    float hy = in->pts[9].y - wy;
    float hz = in->pts[9].z - wz;

    float scale = mag3(hx,hy,hz);

    // if scale explodes or implodes
    if(scale < 1e-6f)
        scale = 1.0f;  // fine whatever

    for(int i=0;i<NUM_LANDMARKS;i++){

        // shift + scale (magic)
        out->pts[i].x = (in->pts[i].x - wx)/scale;
        out->pts[i].y = (in->pts[i].y - wy)/scale;
        out->pts[i].z = (in->pts[i].z - wz)/scale;
    }
}



// parent joint child thing
static const int JOINT_TRIPLETS[][3]={

    // thumb chain (the weird one)
    {0,1,2},{1,2,3},{2,3,4},

    // index fingr
    {0,5,6},{5,6,7},{6,7,8},

    // middle fingre
    {0,9,10},{9,10,11},{10,11,12},

    // the other one idk ring?? yea ring
    {0,13,14},{13,14,15},{14,15,16},

    // pinky tiny dude
    {0,17,18},{17,18,19},{18,19,20},
};

#define NUM_JOINT_TRIPLETS 15  // counted twice to be sure


// computes angles
// returns nothing but fills array
// very cool very legal
void compute_angles(const HandLandmarks* hand,
                    float* angles,
                    int* count){

    *count=NUM_JOINT_TRIPLETS;

    for(int i=0;i<NUM_JOINT_TRIPLETS;i++){

        int p=JOINT_TRIPLETS[i][0];
        int j=JOINT_TRIPLETS[i][1];
        int c=JOINT_TRIPLETS[i][2];

        // vector backwords
        float v1x=hand->pts[p].x - hand->pts[j].x;
        float v1y=hand->pts[p].y - hand->pts[j].y;
        float v1z=hand->pts[p].z - hand->pts[j].z;

        // vector forwrd
        float v2x=hand->pts[c].x - hand->pts[j].x;
        float v2y=hand->pts[c].y - hand->pts[j].y;
        float v2z=hand->pts[c].z - hand->pts[j].z;

        angles[i]=angle_between(v1x,v1y,v1z,
                                v2x,v2y,v2z);
    }
}




// distance pairs becuse shape matters apperently
static const int DIST_PAIRS[][2]={

    {4,8},{4,12},{4,16},{4,20},
    {8,12},{8,16},{8,20},
    {12,16},{12,20},{16,20},

    // wrist to tips for spready hands
    {0,4},{0,8},{0,12},{0,16},{0,20}
};

#define NUM_DIST_PAIRS 15  // hope so


void compute_distances(const HandLandmarks* hand,
                       float* dists,
                       int* count){

    *count = NUM_DIST_PAIRS;

    for(int i=0;i<NUM_DIST_PAIRS;i++){

        int a=DIST_PAIRS[i][0];
        int b=DIST_PAIRS[i][1];

        float dx=hand->pts[a].x - hand->pts[b].x;
        float dy=hand->pts[a].y - hand->pts[b].y;
        float dz=hand->pts[a].z - hand->pts[b].z;

        dists[i]=mag3(dx,dy,dz); // pythag but 3d edition
    }
}



// ------------ BIG FEATURE THING --------------
// raw in
// numbers out
// ML happy
// ----------------------------------------------

void extract_features(const HandLandmarks* raw,
                      FeatureVec* out){

    memset(out->features,0,sizeof(out->features)); // delete ghosts

    HandLandmarks norm;
    normalize_landmarks(raw,&norm);  // must normalize or bad things

    int angle_count=0;
    compute_angles(&norm,&out->features[0],&angle_count);

    int dist_count=0;
    compute_distances(&norm,&out->features[15],&dist_count);

    // rest empty for now (future me issue)
    // maybe palm normal
    // maybe velocity
    // maybe i HAAAAAAAAAAAAAAAATTTTTTTTEEEEEEEEEEEEEE CCCCCCCCCCCCCCCCCCCCCCc
}