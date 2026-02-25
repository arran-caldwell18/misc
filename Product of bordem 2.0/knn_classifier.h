#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#ifdef __cplusplus
extern "C" {
#endif

// Hard limits. These are law.
// If your model exceeds these, it shall not pass.
#define KNN_MAX_SAMPLES   5000
#define KNN_MAX_FEATURES  30
#define KNN_MAX_CLASSES   26
#define KNN_LABEL_LEN     4

// Entire model stored in static buffers.
// No malloc.
// No drama.
typedef struct {
    int   n_samples;
    int   n_features;
    int   n_classes;
    int   k;

    float mean[KNN_MAX_FEATURES];
    float std [KNN_MAX_FEATURES];

    // Normalized training data
    float X[KNN_MAX_SAMPLES][KNN_MAX_FEATURES];

    // Class indices
    int   y[KNN_MAX_SAMPLES];

    // index → string label (e.g. "A")
    char  labels[KNN_MAX_CLASSES][KNN_LABEL_LEN];

} KNNModel;


// Output of classification
typedef struct {
    char  label[KNN_LABEL_LEN];   // predicted letter
    int   class_idx;              // numeric class
    float confidence;             // fraction of votes
    float distance;               // nearest neighbor distance
} KNNResult;


// Load model from disk.
// Returns 1 on success, 0 on failure.
// If this fails, your binary file is suspicious.
int  knn_load(KNNModel* model,
              const char* bin_path,
              const char* label_path);


// Classify raw (unnormalized) features.
// Internally normalizes using stored mean/std.
KNNResult knn_predict(const KNNModel* model,
                      const float* features,
                      int n_features);


// Free resources (currently nothing).
void knn_free(KNNModel* model);

#ifdef __cplusplus
}
#endif

#endif // KNN_CLASSIFIER_H