#include <math.h>
#include <omp.h>
#include "ffnn.h"

/* Transposed: each layer's neurons split across threads.
 * Backward: a transposed copy WT (row i = outgoing weights of neuron i) lets
 * each thread compute whole entries of W^T delta with no cross-thread sum.
 * The update keeps W and WT in sync. */

void forward_transposed(Network *net, const float *x) {
    int L = net->num_layers;
    if (!net->WT) network_alloc_transpose(net);
    for (int i = 0; i < net->sizes[0]; i++) net->a[0][i] = x[i];

    #pragma omp parallel
    {
        for (int l = 0; l < L; l++) {
            int in = net->sizes[l], out = net->sizes[l + 1];
            const float *W = net->W[l], *b = net->b[l], *a_prev = net->a[l];
            float *z = net->z[l + 1], *a = net->a[l + 1];
            int hidden = l < L - 1;

            #pragma omp for schedule(static)
            for (int j = 0; j < out; j++) {
                const float *row = W + (size_t)j * in;
                float sum = b[j];
                for (int i = 0; i < in; i++) sum += row[i] * a_prev[i];
                z[j] = sum;
                if (hidden) a[j] = sum > 0.0f ? sum : 0.0f;
            }
        }

        #pragma omp single
        {
            int n = net->sizes[L];
            float *z = net->z[L], *a = net->a[L];
            float m = z[0], total = 0.0f;
            for (int j = 1; j < n; j++) if (z[j] > m) m = z[j];
            for (int j = 0; j < n; j++) { a[j] = expf(z[j] - m); total += a[j]; }
            for (int j = 0; j < n; j++) a[j] /= total;
        }
    }
}

void backward_transposed(Network *net, int label) {
    int L = net->num_layers;
    if (!net->WT) network_alloc_transpose(net);
    for (int j = 0; j < net->sizes[L]; j++)
        net->delta[L][j] = net->a[L][j] - (j == label ? 1.0f : 0.0f);

    #pragma omp parallel
    for (int l = L - 1; l >= 1; l--) {
        int in = net->sizes[l], out = net->sizes[l + 1];
        const float *WT = net->WT[l], *d_next = net->delta[l + 1], *z = net->z[l];
        float *d = net->delta[l];

        #pragma omp for schedule(static)
        for (int i = 0; i < in; i++) {
            const float *row = WT + (size_t)i * out;
            float sum = 0.0f;
            for (int j = 0; j < out; j++) sum += row[j] * d_next[j];
            d[i] = z[i] > 0.0f ? sum : 0.0f;
        }
    }
}

void update_transposed(Network *net, float lr) {
    int L = net->num_layers;
    if (!net->WT) network_alloc_transpose(net);

    #pragma omp parallel
    for (int l = 0; l < L; l++) {
        int in = net->sizes[l], out = net->sizes[l + 1];
        float *W = net->W[l], *WT = net->WT[l], *b = net->b[l];
        const float *a_prev = net->a[l], *d = net->delta[l + 1];

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < out; j++) {
            float *row = W + (size_t)j * in;
            float g = lr * d[j];
            for (int i = 0; i < in; i++) row[i] -= g * a_prev[i];
            b[j] -= g;
        }

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < in; i++) {
            float *row = WT + (size_t)i * out;
            float ai = a_prev[i];
            for (int j = 0; j < out; j++) row[j] -= (lr * d[j]) * ai;
        }
    }
}
