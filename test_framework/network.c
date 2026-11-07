#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ffnn.h"

/* Small reproducible RNG (xorshift32) so every implementation starts from
 * exactly the same weights regardless of platform rand(). */
static unsigned int rng_next(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rng_uniform(unsigned int *state) {  // in (0, 1)
    return ((rng_next(state) >> 8) + 0.5f) / 16777216.0f;
}

static float rng_normal(unsigned int *state) {   // Box-Muller
    float u1 = rng_uniform(state);
    float u2 = rng_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "Error: out of memory allocating %zu bytes\n", bytes);
        exit(1);
    }
    return p;
}

Network *network_create(int num_layers, const int *sizes, unsigned int seed) {
    Network *net = xmalloc(sizeof(Network));
    net->num_layers = num_layers;
    net->sizes = xmalloc((num_layers + 1) * sizeof(int));
    memcpy(net->sizes, sizes, (num_layers + 1) * sizeof(int));

    net->W = xmalloc(num_layers * sizeof(float *));
    net->b = xmalloc(num_layers * sizeof(float *));
    net->a = xmalloc((num_layers + 1) * sizeof(float *));
    net->z = xmalloc((num_layers + 1) * sizeof(float *));
    net->delta = xmalloc((num_layers + 1) * sizeof(float *));
    net->WT = NULL;

    unsigned int state = seed ? seed : 1u;

    for (int l = 0; l < num_layers; l++) {
        int in = sizes[l], out = sizes[l + 1];
        net->W[l] = xmalloc((size_t)out * in * sizeof(float));
        net->b[l] = xmalloc((size_t)out * sizeof(float));

        // He initialization (good default for ReLU): N(0, 2/in)
        float scale = sqrtf(2.0f / (float)in);
        for (long k = 0; k < (long)out * in; k++) {
            net->W[l][k] = scale * rng_normal(&state);
        }
        memset(net->b[l], 0, (size_t)out * sizeof(float));
    }

    for (int l = 0; l <= num_layers; l++) {
        net->a[l] = xmalloc((size_t)sizes[l] * sizeof(float));
        memset(net->a[l], 0, (size_t)sizes[l] * sizeof(float));
        if (l == 0) {
            net->z[l] = NULL;
            net->delta[l] = NULL;
        } else {
            net->z[l] = xmalloc((size_t)sizes[l] * sizeof(float));
            net->delta[l] = xmalloc((size_t)sizes[l] * sizeof(float));
            memset(net->z[l], 0, (size_t)sizes[l] * sizeof(float));
            memset(net->delta[l], 0, (size_t)sizes[l] * sizeof(float));
        }
    }
    return net;
}

void network_free(Network *net) {
    if (!net) return;
    for (int l = 0; l < net->num_layers; l++) {
        free(net->W[l]);
        free(net->b[l]);
    }
    for (int l = 0; l <= net->num_layers; l++) {
        free(net->a[l]);
        free(net->z[l]);
        free(net->delta[l]);
    }
    if (net->WT) {
        for (int l = 0; l < net->num_layers; l++) free(net->WT[l]);
        free(net->WT);
    }
    free(net->W);
    free(net->b);
    free(net->a);
    free(net->z);
    free(net->delta);
    free(net->sizes);
    free(net);
}

void network_alloc_transpose(Network *net) {
    if (net->WT) return;
    net->WT = xmalloc(net->num_layers * sizeof(float *));
    for (int l = 0; l < net->num_layers; l++) {
        int in = net->sizes[l], out = net->sizes[l + 1];
        net->WT[l] = xmalloc((size_t)out * in * sizeof(float));
        for (int j = 0; j < out; j++)
            for (int i = 0; i < in; i++)
                net->WT[l][(size_t)i * out + j] = net->W[l][(size_t)j * in + i];
    }
}

long network_num_params(const Network *net) {
    long total = 0;
    for (int l = 0; l < net->num_layers; l++) {
        total += (long)net->sizes[l + 1] * net->sizes[l] + net->sizes[l + 1];
    }
    return total;
}

void network_copy_params(const Network *src, Network *dst) {
    for (int l = 0; l < src->num_layers; l++) {
        size_t in = src->sizes[l], out = src->sizes[l + 1];
        memcpy(dst->W[l], src->W[l], out * in * sizeof(float));
        memcpy(dst->b[l], src->b[l], out * sizeof(float));
    }
}

float network_loss(const Network *net, int label) {
    float p = net->a[net->num_layers][label];
    if (p < 1e-30f) p = 1e-30f;
    return -logf(p);
}

int network_predict(const Network *net) {
    int n = net->sizes[net->num_layers];
    const float *p = net->a[net->num_layers];
    int best = 0;
    for (int k = 1; k < n; k++) {
        if (p[k] > p[best]) best = k;
    }
    return best;
}
