#ifndef FFNN_H
#define FFNN_H

/******************************************************************************
 * Feedforward neural network (FFNN) shared definitions.
 *
 * Network shape: sizes[0] -> sizes[1] -> ... -> sizes[num_layers]
 *   - sizes[0] is the input size (784 for MNIST)
 *   - sizes[num_layers] is the output size (10 for MNIST)
 *   - hidden layers use ReLU, the output layer uses softmax + cross-entropy
 *
 * Weight layout ("stored by neuron"):
 *   Layer l maps sizes[l] inputs to sizes[l+1] outputs.
 *   W[l] is a sizes[l+1] x sizes[l] matrix in ROW-MAJOR order, so
 *   row j holds every incoming weight of output neuron j:
 *
 *       W[l][j * in + i]  = weight from input i to neuron j
 *
 *   This is the layout the main parallel track splits across threads:
 *   each thread owns a contiguous block of neurons (rows).
 *
 * Buffers (all owned by the Network):
 *   a[0..num_layers]   activations; a[0] is the input, a[num_layers] is the
 *                      softmax output (class probabilities)
 *   z[1..num_layers]   pre-activations (W a + b) for each layer
 *   delta[1..num_layers] dLoss/dz for each layer (filled by backward)
 *****************************************************************************/

typedef struct {
    int num_layers;      // number of weight layers (hidden layers + 1)
    int *sizes;          // num_layers + 1 entries

    float **W;           // W[l], l = 0..num_layers-1, sizes[l+1] x sizes[l]
    float **b;           // b[l], length sizes[l+1]

    float **a;           // a[l], l = 0..num_layers, length sizes[l]
    float **z;           // z[l], l = 1..num_layers (z[0] unused, NULL)
    float **delta;       // delta[l], l = 1..num_layers (delta[0] unused, NULL)
} Network;

/* ---------------------------------------------------------------------------
 * Network management (test_framework/network.c, shared by all versions)
 * ------------------------------------------------------------------------- */
Network *network_create(int num_layers, const int *sizes, unsigned int seed);
void network_free(Network *net);
long network_num_params(const Network *net);
void network_copy_params(const Network *src, Network *dst);  // same shape
float network_loss(const Network *net, int label);   // CE loss of last forward
int network_predict(const Network *net);              // argmax of last forward

/* ---------------------------------------------------------------------------
 * Implementations. A training step on one sample (x, label) is:
 *
 *     forward(net, x);          // fills a[], z[]
 *     backward(net, label);     // fills delta[] using the CURRENT weights
 *     update(net, lr);          // W -= lr * delta (x) a_prev,  b -= lr * delta
 *
 * backward() computes every delta before update() touches any weight, so the
 * result is exactly plain SGD. The three phases are separate functions so the
 * timing harness can measure each one.
 *
 * Different files provide different versions:
 *   - baseline_nn.c : simple serial reference (no optimizations)
 *   - (later) serial optimized, parallel neuron-split, ...
 * ------------------------------------------------------------------------- */

// Baseline serial implementation (baseline_nn.c)
void forward_serial(Network *net, const float *x);
void backward_serial(Network *net, int label);
void update_serial(Network *net, float lr);

// Convenience: one full SGD step using a given implementation
typedef struct {
    const char *name;
    void (*forward)(Network *, const float *);
    void (*backward)(Network *, int);
    void (*update)(Network *, float);
} NNImpl;

static inline void train_step(const NNImpl *impl, Network *net,
                              const float *x, int label, float lr) {
    impl->forward(net, x);
    impl->backward(net, label);
    impl->update(net, lr);
}

#endif // FFNN_H
