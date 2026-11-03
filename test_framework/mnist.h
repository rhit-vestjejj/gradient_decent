#ifndef MNIST_H
#define MNIST_H

#define MNIST_PIXELS 784   // 28 x 28
#define MNIST_CLASSES 10

typedef struct {
    int count;         // number of samples
    float *images;     // count x 784, pixels scaled to [0, 1]
    int *labels;       // count, values 0..9
} MnistSet;

// Load an IDX image file and its matching IDX label file.
// Returns NULL (and prints why) if the files are missing or malformed.
// max_count <= 0 loads every sample.
MnistSet *mnist_load(const char *image_path, const char *label_path, int max_count);
void mnist_free(MnistSet *set);

// Default locations (populated by `make data`)
#define MNIST_TRAIN_IMAGES "data/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS "data/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGES  "data/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS  "data/t10k-labels-idx1-ubyte"

#endif // MNIST_H
