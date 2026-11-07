#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "ffnn.h"
#include "mnist.h"

/* Checks a parallel implementation against the serial baseline: same seed,
 * same samples, many SGD steps, at several thread counts. */

static const NNImpl SERIAL = {"serial", forward_serial, backward_serial, update_serial};

static unsigned int g_state = 1;
static float rand_uniform(void) {
    g_state ^= g_state << 13; g_state ^= g_state >> 17; g_state ^= g_state << 5;
    return (g_state >> 8) / 8388608.0f - 1.0f;
}

static double max_param_diff(const Network *a, const Network *b) {
    double m = 0;
    for (int l = 0; l < a->num_layers; l++) {
        long n = (long)a->sizes[l] * a->sizes[l + 1];
        for (long k = 0; k < n; k++) m = fmax(m, fabs(a->W[l][k] - b->W[l][k]));
        for (int j = 0; j < a->sizes[l + 1]; j++) m = fmax(m, fabs(a->b[l][j] - b->b[l][j]));
    }
    return m;
}

static double max_transpose_diff(const Network *net) {
    if (!net->WT) return 0;
    double m = 0;
    for (int l = 0; l < net->num_layers; l++) {
        int in = net->sizes[l], out = net->sizes[l + 1];
        for (int j = 0; j < out; j++)
            for (int i = 0; i < in; i++)
                m = fmax(m, fabs(net->W[l][(size_t)j * in + i] - net->WT[l][(size_t)i * out + j]));
    }
    return m;
}

static int test_vs_serial(const NNImpl *impl, int num_layers, const int *sizes,
                          int threads, unsigned int seed) {
    const int steps = 30;
    const float lr = 0.05f;
    omp_set_num_threads(threads);

    Network *ref = network_create(num_layers, sizes, seed);
    Network *net = network_create(num_layers, sizes, seed);
    float *x = malloc(sizes[0] * sizeof(float));
    int n_out = sizes[num_layers];

    g_state = seed;
    double max_prob = 0, max_delta = 0;
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < sizes[0]; i++) x[i] = rand_uniform();
        int label = s % n_out;

        SERIAL.forward(ref, x);
        impl->forward(net, x);
        for (int j = 0; j < n_out; j++)
            max_prob = fmax(max_prob, fabs(ref->a[num_layers][j] - net->a[num_layers][j]));

        SERIAL.backward(ref, label);
        impl->backward(net, label);
        for (int l = 1; l <= num_layers; l++)
            for (int i = 0; i < sizes[l]; i++)
                max_delta = fmax(max_delta, fabs(ref->delta[l][i] - net->delta[l][i]));

        SERIAL.update(ref, lr);
        impl->update(net, lr);
    }
    double max_w = max_param_diff(ref, net);
    double max_t = max_transpose_diff(net);

    int passed = max_prob < 1e-4 && max_delta < 1e-4 && max_w < 1e-4 && max_t < 1e-5;
    printf("    %2d threads: max diff p %.1e, delta %.1e, W %.1e%s  %s\n",
           threads, max_prob, max_delta, max_w,
           net->WT ? (max_t < 1e-5 ? ", WT in sync" : ", WT OUT OF SYNC") : "",
           passed ? "PASS" : "FAIL");

    free(x);
    network_free(ref);
    network_free(net);
    return passed;
}

static int test_mnist(const NNImpl *impl, int threads) {
    MnistSet *train = mnist_load(MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, 10000);
    MnistSet *test = mnist_load(MNIST_TEST_IMAGES, MNIST_TEST_LABELS, 2000);
    if (!train || !test) {
        printf("    MNIST files not found - SKIPPED (run `make data`)\n");
        mnist_free(train);
        mnist_free(test);
        return 1;
    }
    omp_set_num_threads(threads);
    int sizes[] = {MNIST_PIXELS, 64, MNIST_CLASSES};
    Network *net = network_create(2, sizes, 42);
    for (int s = 0; s < train->count; s++)
        train_step(impl, net, train->images + (size_t)s * MNIST_PIXELS, train->labels[s], 0.01f);
    int correct = 0;
    for (int s = 0; s < test->count; s++) {
        impl->forward(net, test->images + (size_t)s * MNIST_PIXELS);
        correct += network_predict(net) == test->labels[s];
    }
    double acc = 100.0 * correct / test->count;
    int passed = acc > 85.0;
    printf("    784-64-10, 10k steps, %d threads: test accuracy %.1f%%  %s\n",
           threads, acc, passed ? "PASS" : "FAIL");
    network_free(net);
    mnist_free(train);
    mnist_free(test);
    return passed;
}

int main(int argc, char *argv[]) {
    NNImpl impls[] = {
        {"partial", forward_partial, backward_partial, update_partial},
        {"transposed", forward_transposed, backward_transposed, update_transposed},
    };
    int num_impls = 2;
    const char *only = argc > 1 ? argv[1] : NULL;

    int s0[] = {4, 3, 2};
    int s1[] = {10, 1, 7, 3};
    int s2[] = {6, 5, 4, 3};
    int s3[] = {8, 16, 16, 16, 16, 5};
    int s4[] = {784, 32, 32, 10};
    int s5[] = {784, 100, 50, 10};
    struct { int num_layers; int *sizes; const char *name; } shapes[] = {
        {2, s0, "4-3-2"},
        {3, s1, "10-1-7-3"},
        {3, s2, "6-5-4-3"},
        {5, s3, "8-16-16-16-16-5"},
        {3, s4, "784-32-32-10"},
        {3, s5, "784-100-50-10"},
    };
    int num_shapes = 6;
    int thread_counts[] = {1, 2, 3, 4, 7, 8};
    int num_thread_counts = 6;

    int total = 0, passed = 0;
    for (int m = 0; m < num_impls; m++) {
        if (only && strcmp(only, impls[m].name) != 0) continue;
        printf("=== %s vs serial baseline ===\n\n", impls[m].name);
        for (int s = 0; s < num_shapes; s++) {
            printf("  Network %s:\n", shapes[s].name);
            for (int t = 0; t < num_thread_counts; t++) {
                int r = test_vs_serial(&impls[m], shapes[s].num_layers, shapes[s].sizes,
                                       thread_counts[t], 2000 + 10 * s + t);
                total++; passed += r;
            }
        }
        printf("  MNIST sanity run:\n");
        int r = test_mnist(&impls[m], 4);
        total++; passed += r;
        printf("\n");
    }

    printf("=== Final Results ===\n");
    printf("Total tests: %d\n", total);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", total - passed);
    if (passed == total) {
        printf("\nAll tests PASSED!\n");
        return 0;
    }
    printf("\nSome tests FAILED!\n");
    return 1;
}
