#include <math.h>
#include "ffnn.h"

/******************************************************************************
 * BASELINE: serial FFNN training step
 *
 * A simple serial forward pass, backpropagation, and SGD update with no
 * performance optimizations. This is the reference every other version is
 * checked against and timed against.
 *
 * Weight layout (see ffnn.h): W[l] is out x in, ROW-MAJOR, so
 *     W[l][j * in + i] = weight from input i to neuron j
 * Moving along a row (i++) is unit stride in memory.
 *****************************************************************************/

/* Forward pass: for each layer, z = W a_prev + b, then
 *   a = ReLU(z)      on hidden layers
 *   a = softmax(z)   on the output layer
 */
void forward_serial(Network *net, const float *x) {
    int L = net->num_layers;

    // Copy the input into a[0]
    for (int i = 0; i < net->sizes[0]; i++) {
        net->a[0][i] = x[i];
    }

    for (int l = 0; l < L; l++) {
        int in = net->sizes[l];
        int out = net->sizes[l + 1];
        const float *W = net->W[l];
        const float *b = net->b[l];
        const float *a_prev = net->a[l];
        float *z = net->z[l + 1];
        float *a = net->a[l + 1];

        // Matrix-vector product: one dot product per output neuron.
        // Each neuron j is independent -> this is the loop we split later.
        for (int j = 0; j < out; j++) {
            float sum = b[j];
            for (int i = 0; i < in; i++) {
                sum += W[j * in + i] * a_prev[i];
            }
            z[j] = sum;
        }

        if (l < L - 1) {
            // Hidden layer: ReLU
            for (int j = 0; j < out; j++) {
                a[j] = z[j] > 0.0f ? z[j] : 0.0f;
            }
        } else {
            // Output layer: numerically stable softmax
            float max_z = z[0];
            for (int j = 1; j < out; j++) {
                if (z[j] > max_z) max_z = z[j];
            }
            float total = 0.0f;
            for (int j = 0; j < out; j++) {
                a[j] = expf(z[j] - max_z);
                total += a[j];
            }
            for (int j = 0; j < out; j++) {
                a[j] /= total;
            }
        }
    }
}

/* Backward pass: compute delta[l] = dLoss/dz[l] for every layer.
 *
 * Output layer (softmax + cross-entropy):
 *     delta[L][j] = a[L][j] - (j == label)
 *
 * Hidden layers, from the top down:
 *     delta[l][i] = ReLU'(z[l][i]) * sum_j W[l][j][i] * delta[l+1][j]
 *
 * That sum is W^T * delta: it walks DOWN a column of the row-major W,
 * which is stride-`in` memory access. This is the step that needs a sum
 * across all neurons (and across all threads, once we parallelize).
 *
 * No weights are modified here, so every delta uses the pre-update weights.
 */
void backward_serial(Network *net, int label) {
    int L = net->num_layers;

    // Output layer
    int n_out = net->sizes[L];
    for (int j = 0; j < n_out; j++) {
        net->delta[L][j] = net->a[L][j] - (j == label ? 1.0f : 0.0f);
    }

    // Hidden layers: propagate delta from layer l+1 down to layer l
    for (int l = L - 1; l >= 1; l--) {
        int in = net->sizes[l];        // neurons in layer l
        int out = net->sizes[l + 1];   // neurons in layer l+1
        const float *W = net->W[l];    // out x in
        const float *d_next = net->delta[l + 1];
        const float *z = net->z[l];
        float *d = net->delta[l];

        for (int i = 0; i < in; i++) {
            float sum = 0.0f;
            // (Notice: stride-`in` memory access down column i.)
            for (int j = 0; j < out; j++) {
                sum += W[j * in + i] * d_next[j];
            }
            d[i] = z[i] > 0.0f ? sum : 0.0f;   // ReLU derivative is 1 or 0
        }
    }
}

/* SGD update for every layer:
 *     W[l][j][i] -= lr * delta[l+1][j] * a[l][i]
 *     b[l][j]    -= lr * delta[l+1][j]
 * Each neuron j updates only its own row, so neurons are independent.
 */
void update_serial(Network *net, float lr) {
    int L = net->num_layers;

    for (int l = 0; l < L; l++) {
        int in = net->sizes[l];
        int out = net->sizes[l + 1];
        float *W = net->W[l];
        float *b = net->b[l];
        const float *a_prev = net->a[l];
        const float *d = net->delta[l + 1];

        for (int j = 0; j < out; j++) {
            float g = lr * d[j];
            for (int i = 0; i < in; i++) {
                W[j * in + i] -= g * a_prev[i];
            }
            b[j] -= g;
        }
    }
}
