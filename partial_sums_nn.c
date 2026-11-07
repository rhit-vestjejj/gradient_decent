#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "ffnn.h"

/* Partial sums: each layer's neurons split across threads.
 * Backward: each thread accumulates W^T delta over its own neurons into a
 * private buffer, then the buffers are summed. */

static float *partial = NULL;
static size_t partial_cap = 0;

void forward_partial(Network *net, const float *x) {
    int L = net->num_layers;
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

void backward_partial(Network *net, int label) {
    int L = net->num_layers;
    for (int j = 0; j < net->sizes[L]; j++)
        net->delta[L][j] = net->a[L][j] - (j == label ? 1.0f : 0.0f);
    if (L < 2) return;

    int max_in = 0;
    for (int l = 1; l < L; l++) if (net->sizes[l] > max_in) max_in = net->sizes[l];
    size_t stride = ((size_t)max_in + 15) & ~(size_t)15;
    size_t need = stride * omp_get_max_threads();
    if (need > partial_cap) {
        free(partial);
        partial = malloc(need * sizeof(float));
        partial_cap = need;
    }

    #pragma omp parallel
    {
        int nthr = omp_get_num_threads();
        float *mine = partial + stride * omp_get_thread_num();

        for (int l = L - 1; l >= 1; l--) {
            int in = net->sizes[l], out = net->sizes[l + 1];
            const float *W = net->W[l], *d_next = net->delta[l + 1], *z = net->z[l];
            float *d = net->delta[l];

            for (int i = 0; i < in; i++) mine[i] = 0.0f;

            #pragma omp for schedule(static)
            for (int j = 0; j < out; j++) {
                const float *row = W + (size_t)j * in;
                float dj = d_next[j];
                for (int i = 0; i < in; i++) mine[i] += row[i] * dj;
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < in; i++) {
                float sum = 0.0f;
                for (int t = 0; t < nthr; t++) sum += partial[stride * t + i];
                d[i] = z[i] > 0.0f ? sum : 0.0f;
            }
        }
    }
}

void update_partial(Network *net, float lr) {
    int L = net->num_layers;

    #pragma omp parallel
    for (int l = 0; l < L; l++) {
        int in = net->sizes[l], out = net->sizes[l + 1];
        float *W = net->W[l], *b = net->b[l];
        const float *a_prev = net->a[l], *d = net->delta[l + 1];

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < out; j++) {
            float *row = W + (size_t)j * in;
            float g = lr * d[j];
            for (int i = 0; i < in; i++) row[i] -= g * a_prev[i];
            b[j] -= g;
        }
    }
}
