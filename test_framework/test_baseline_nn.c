#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ffnn.h"
#include "mnist.h"

/******************************************************************************
 * Correctness tests for the baseline serial FFNN.
 *
 *   1. Forward pass matches an independent double-precision reference.
 *   2. Backprop gradients match central finite differences of the loss
 *      (finite differences computed in double precision from the same
 *      float weights, so float round-off does not swamp the check).
 *   3. update() applies exactly W -= lr * delta (x) a_prev.
 *   4. Repeated SGD on a fixed small batch drives the loss down.
 *   5. (If data/ exists) short MNIST run reaches sensible accuracy.
 *****************************************************************************/

static const NNImpl IMPL = {"serial", forward_serial, backward_serial, update_serial};

static unsigned int g_state = 1;
static float rand_uniform(void) {  // in [-1, 1)
    g_state ^= g_state << 13; g_state ^= g_state >> 17; g_state ^= g_state << 5;
    return (g_state >> 8) / 8388608.0f - 1.0f;
}

/* Independent double-precision forward pass. Optionally overrides one weight
 * (layer pl, flat index pk) with value pv. Returns cross-entropy loss and
 * (if probs != NULL) writes the output probabilities. */
static double reference_loss(const Network *net, const float *x, int label,
                             int pl, long pk, double pv, double *probs) {
    int L = net->num_layers;
    int max_size = 0;
    for (int l = 0; l <= L; l++) if (net->sizes[l] > max_size) max_size = net->sizes[l];
    double *cur = malloc(max_size * sizeof(double));
    double *nxt = malloc(max_size * sizeof(double));
    for (int i = 0; i < net->sizes[0]; i++) cur[i] = x[i];

    for (int l = 0; l < L; l++) {
        int in = net->sizes[l], out = net->sizes[l + 1];
        for (int j = 0; j < out; j++) {
            double s = net->b[l][j];
            for (int i = 0; i < in; i++) {
                long k = (long)j * in + i;
                double w = (l == pl && k == pk) ? pv : (double)net->W[l][k];
                s += w * cur[i];
            }
            nxt[j] = (l < L - 1) ? (s > 0 ? s : 0) : s;
        }
        double *t = cur; cur = nxt; nxt = t;
    }

    int n = net->sizes[L];
    double m = cur[0];
    for (int j = 1; j < n; j++) if (cur[j] > m) m = cur[j];
    double total = 0;
    for (int j = 0; j < n; j++) total += exp(cur[j] - m);
    double loss = -((cur[label] - m) - log(total));
    if (probs) for (int j = 0; j < n; j++) probs[j] = exp(cur[j] - m) / total;

    free(cur);
    free(nxt);
    return loss;
}

static int test_forward(int num_layers, const int *sizes, unsigned int seed) {
    Network *net = network_create(num_layers, sizes, seed);
    float *x = malloc(sizes[0] * sizeof(float));
    for (int i = 0; i < sizes[0]; i++) x[i] = rand_uniform();
    int label = seed % sizes[num_layers];

    IMPL.forward(net, x);
    double *probs = malloc(sizes[num_layers] * sizeof(double));
    reference_loss(net, x, label, -1, -1, 0, probs);

    double max_err = 0, prob_sum = 0;
    for (int j = 0; j < sizes[num_layers]; j++) {
        double e = fabs(net->a[num_layers][j] - probs[j]);
        if (e > max_err) max_err = e;
        prob_sum += net->a[num_layers][j];
    }
    int passed = max_err < 1e-5 && fabs(prob_sum - 1.0) < 1e-5;
    printf("    forward vs reference: max err %.2e, sum(p) = %.6f  %s\n",
           max_err, prob_sum, passed ? "PASS" : "FAIL");

    free(probs);
    free(x);
    network_free(net);
    return passed;
}

/* Gradient check on every weight (or a sample of them for big layers). */
static int test_gradients(int num_layers, const int *sizes, unsigned int seed) {
    Network *net = network_create(num_layers, sizes, seed);
    float *x = malloc(sizes[0] * sizeof(float));
    for (int i = 0; i < sizes[0]; i++) x[i] = rand_uniform();
    int label = seed % sizes[num_layers];

    IMPL.forward(net, x);
    IMPL.backward(net, label);

    const double h = 1e-3;
    double max_rel = 0;
    int checked = 0, bad = 0;

    for (int l = 0; l < num_layers; l++) {
        int in = sizes[l], out = sizes[l + 1];
        long total = (long)in * out;
        long stride = total > 400 ? total / 400 : 1;   // sample big layers
        for (long k = 0; k < total; k += stride) {
            int j = (int)(k / in), i = (int)(k % in);
            double analytic = (double)net->delta[l + 1][j] * net->a[l][i];
            double w = net->W[l][k];
            double numeric = (reference_loss(net, x, label, l, k, w + h, NULL) -
                              reference_loss(net, x, label, l, k, w - h, NULL)) / (2 * h);
            double err = fabs(analytic - numeric);
            double rel = err / fmax(1e-3, fabs(analytic) + fabs(numeric));
            if (rel > max_rel) max_rel = rel;
            // ReLU kinks can make a finite difference straddle a corner, so
            // allow a tiny absolute slack as well as a relative tolerance.
            if (rel > 1e-2 && err > 1e-4) bad++;
            checked++;
        }
    }

    int passed = (bad == 0);
    printf("    gradient check: %d weights, max rel err %.2e, %d bad  %s\n",
           checked, max_rel, bad, passed ? "PASS" : "FAIL");

    free(x);
    network_free(net);
    return passed;
}

static int test_update(int num_layers, const int *sizes, unsigned int seed) {
    Network *net = network_create(num_layers, sizes, seed);
    Network *before = network_create(num_layers, sizes, seed);
    float *x = malloc(sizes[0] * sizeof(float));
    for (int i = 0; i < sizes[0]; i++) x[i] = rand_uniform();
    int label = seed % sizes[num_layers];
    const float lr = 0.05f;

    IMPL.forward(net, x);
    IMPL.backward(net, label);
    IMPL.update(net, lr);

    double max_err = 0;
    for (int l = 0; l < num_layers; l++) {
        int in = sizes[l], out = sizes[l + 1];
        for (int j = 0; j < out; j++) {
            float g = lr * net->delta[l + 1][j];
            for (int i = 0; i < in; i++) {
                float expect = before->W[l][(long)j * in + i] - g * net->a[l][i];
                double e = fabs(expect - net->W[l][(long)j * in + i]);
                if (e > max_err) max_err = e;
            }
            double e = fabs((before->b[l][j] - g) - net->b[l][j]);
            if (e > max_err) max_err = e;
        }
    }
    int passed = max_err < 1e-6;
    printf("    SGD update rule: max err %.2e  %s\n", max_err, passed ? "PASS" : "FAIL");

    free(x);
    network_free(before);
    network_free(net);
    return passed;
}

static int test_loss_decreases(int num_layers, const int *sizes, unsigned int seed) {
    Network *net = network_create(num_layers, sizes, seed);
    const int batch = 8;
    float *xs = malloc((size_t)batch * sizes[0] * sizeof(float));
    int labels[8];
    for (int k = 0; k < batch * sizes[0]; k++) xs[k] = rand_uniform();
    for (int s = 0; s < batch; s++) labels[s] = s % sizes[num_layers];

    double start = 0, end = 0;
    for (int s = 0; s < batch; s++) {
        IMPL.forward(net, xs + (size_t)s * sizes[0]);
        start += network_loss(net, labels[s]);
    }
    for (int epoch = 0; epoch < 200; epoch++)
        for (int s = 0; s < batch; s++)
            train_step(&IMPL, net, xs + (size_t)s * sizes[0], labels[s], 0.01f);
    for (int s = 0; s < batch; s++) {
        IMPL.forward(net, xs + (size_t)s * sizes[0]);
        end += network_loss(net, labels[s]);
    }
    // Require a clear drop, not a full fit: tiny bottleneck shapes like
    // 10-1-7-3 do not have the capacity to memorize 8 random samples.
    int passed = end < 0.9 * start;
    printf("    SGD on 8 fixed samples: loss %.4f -> %.4f  %s\n",
           start / batch, end / batch, passed ? "PASS" : "FAIL");

    free(xs);
    network_free(net);
    return passed;
}

static int test_mnist(void) {
    MnistSet *train = mnist_load(MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, 10000);
    MnistSet *test = mnist_load(MNIST_TEST_IMAGES, MNIST_TEST_LABELS, 2000);
    if (!train || !test) {
        printf("    MNIST files not found - SKIPPED (run `make data`)\n");
        mnist_free(train);
        mnist_free(test);
        return 1;
    }
    int sizes[] = {MNIST_PIXELS, 64, MNIST_CLASSES};
    Network *net = network_create(2, sizes, 42);
    for (int s = 0; s < train->count; s++)
        train_step(&IMPL, net, train->images + (size_t)s * MNIST_PIXELS,
                   train->labels[s], 0.01f);
    int correct = 0;
    for (int s = 0; s < test->count; s++) {
        IMPL.forward(net, test->images + (size_t)s * MNIST_PIXELS);
        correct += network_predict(net) == test->labels[s];
    }
    double acc = 100.0 * correct / test->count;
    int passed = acc > 85.0;
    printf("    784-64-10, 10k SGD steps: test accuracy %.1f%% (need > 85%%)  %s\n",
           acc, passed ? "PASS" : "FAIL");
    network_free(net);
    mnist_free(train);
    mnist_free(test);
    return passed;
}

int main(void) {
    printf("=== Baseline (serial) FFNN Tests ===\n\n");

    // Shapes: tiny, non-square, width-1 bottleneck, deep, MNIST-shaped
    int s0[] = {4, 3, 2};
    int s1[] = {6, 5, 4, 3};
    int s2[] = {10, 1, 7, 3};
    int s3[] = {8, 16, 16, 16, 16, 5};
    int s4[] = {784, 32, 32, 10};
    struct { int num_layers; int *sizes; const char *name; } shapes[] = {
        {2, s0, "4-3-2"},
        {3, s1, "6-5-4-3"},
        {3, s2, "10-1-7-3"},
        {5, s3, "8-16-16-16-16-5"},
        {3, s4, "784-32-32-10"},
    };
    int num_shapes = 5;
    int seeds_per_shape = 3;

    int total = 0, passed = 0;
    for (int s = 0; s < num_shapes; s++) {
        printf("Network %s:\n", shapes[s].name);
        for (int t = 0; t < seeds_per_shape; t++) {
            unsigned int seed = 1000 + 10 * s + t;
            g_state = seed;
            printf("  seed=%u\n", seed);
            int r;
            r = test_forward(shapes[s].num_layers, shapes[s].sizes, seed);   total++; passed += r;
            r = test_gradients(shapes[s].num_layers, shapes[s].sizes, seed); total++; passed += r;
            r = test_update(shapes[s].num_layers, shapes[s].sizes, seed);    total++; passed += r;
        }
        int r = test_loss_decreases(shapes[s].num_layers, shapes[s].sizes, 7 + s);
        total++; passed += r;
        printf("\n");
    }

    printf("MNIST sanity run:\n");
    int r = test_mnist();
    total++; passed += r;

    printf("\n=== Final Results ===\n");
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
