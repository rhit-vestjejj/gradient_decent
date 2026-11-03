#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "ffnn.h"
#include "mnist.h"

/******************************************************************************
 * Timing for the baseline serial FFNN.
 *
 * Network: 784 -> [width] x hidden_layers -> 10, trained with SGD on MNIST
 * samples (one sample per step). For each configuration we run several
 * trials of K steps and keep the best trial (best-of-N, as in linsolve).
 * Each step's forward / backward / update phases are timed separately with
 * omp_get_wtime().
 *
 * Usage: timing_baseline <output.csv> [width1] [width2] ...
 *
 * CSV columns (one row per configuration):
 *   impl,hidden_layers,width,threads,params,steps,
 *   time_per_step,forward,backward,update      (all seconds per step)
 *****************************************************************************/

static const NNImpl IMPLS[] = {
    {"serial", forward_serial, backward_serial, update_serial},
};
static const int NUM_IMPLS = sizeof(IMPLS) / sizeof(IMPLS[0]);

typedef struct {
    double total, forward, backward, update;
} StepTimes;

/* Pick K so each trial does roughly the same amount of work (~1.5e9 MACs,
 * a few seconds serial) no matter how big the network is. */
static int steps_for(long params) {
    long k = 500000000L / params;
    if (k < 5) k = 5;
    if (k > 2000) k = 2000;
    return (int)k;
}

static StepTimes time_config(const NNImpl *impl, int hidden_layers, int width,
                             int threads, const MnistSet *data,
                             int num_trials, int *steps_out, long *params_out) {
    int num_layers = hidden_layers + 1;
    int *sizes = malloc((num_layers + 1) * sizeof(int));
    sizes[0] = MNIST_PIXELS;
    for (int l = 1; l <= hidden_layers; l++) sizes[l] = width;
    sizes[num_layers] = MNIST_CLASSES;

    if (threads > 0) omp_set_num_threads(threads);

    Network *net = network_create(num_layers, sizes, 12345);
    long params = network_num_params(net);
    int steps = steps_for(params);
    const float lr = 0.001f;

    // Warm-up: touch all the weights once so page faults aren't timed
    for (int s = 0; s < 3; s++) {
        train_step(impl, net, data->images + (size_t)s * MNIST_PIXELS, data->labels[s], lr);
    }

    StepTimes best = {INFINITY, 0, 0, 0};
    int sample = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        StepTimes t = {0, 0, 0, 0};
        for (int s = 0; s < steps; s++) {
            const float *x = data->images + (size_t)sample * MNIST_PIXELS;
            int label = data->labels[sample];
            sample = (sample + 1) % data->count;

            double t0 = omp_get_wtime();
            impl->forward(net, x);
            double t1 = omp_get_wtime();
            impl->backward(net, label);
            double t2 = omp_get_wtime();
            impl->update(net, lr);
            double t3 = omp_get_wtime();

            t.forward += t1 - t0;
            t.backward += t2 - t1;
            t.update += t3 - t2;
        }
        t.total = t.forward + t.backward + t.update;
        if (t.total < best.total) best = t;
    }

    best.total /= steps;
    best.forward /= steps;
    best.backward /= steps;
    best.update /= steps;

    *steps_out = steps;
    *params_out = params;
    network_free(net);
    free(sizes);
    return best;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <output.csv> [width1] [width2] ...\n", argv[0]);
        printf("If no widths provided, uses 64 256 1024 4096\n");
        return 1;
    }
    char *output_file = argv[1];

    int default_widths[] = {64, 256, 1024, 4096};
    int num_widths = 4;
    int *widths = default_widths;
    if (argc > 2) {
        num_widths = argc - 2;
        widths = malloc(num_widths * sizeof(int));
        for (int i = 0; i < num_widths; i++) widths[i] = atoi(argv[i + 2]);
    }

    int hidden_layer_counts[] = {2, 4};
    int num_hidden_counts = 2;
    int num_trials = 5;  // best-of-5

    MnistSet *data = mnist_load(MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, 0);
    if (!data) return 1;

    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_file);
        return 1;
    }
    fprintf(fp, "impl,hidden_layers,width,threads,params,steps,"
                "time_per_step,forward,backward,update\n");

    printf("Running baseline FFNN timing...\n");
    printf("Output: %s\n", output_file);
    printf("MNIST training samples: %d\n", data->count);
    printf("Trial method: best of %d trials per configuration\n\n", num_trials);

    for (int h = 0; h < num_hidden_counts; h++) {
        int hl = hidden_layer_counts[h];
        for (int w = 0; w < num_widths; w++) {
            int width = widths[w];
            printf("Network 784-[%d x %d]-10:\n", width, hl);
            for (int m = 0; m < NUM_IMPLS; m++) {
                int threads = 1;  // serial baseline
                int steps;
                long params;
                printf("  %-10s (%d thread)... ", IMPLS[m].name, threads);
                fflush(stdout);
                StepTimes t = time_config(&IMPLS[m], hl, width, threads, data,
                                          num_trials, &steps, &params);
                printf("%9.3f ms/step  (fwd %.3f, bwd %.3f, upd %.3f)  [%ld params, %d steps]\n",
                       t.total * 1e3, t.forward * 1e3, t.backward * 1e3, t.update * 1e3,
                       params, steps);
                fprintf(fp, "%s,%d,%d,%d,%ld,%d,%.9f,%.9f,%.9f,%.9f\n",
                        IMPLS[m].name, hl, width, threads, params, steps,
                        t.total, t.forward, t.backward, t.update);
                fflush(fp);
            }
            printf("\n");
        }
    }

    fclose(fp);
    mnist_free(data);
    if (widths != default_widths) free(widths);
    printf("Timing complete! Results written to %s\n", output_file);
    return 0;
}
