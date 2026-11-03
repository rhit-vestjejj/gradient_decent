Proposal: Parallelizing Stochastic Gradient Descent for a Feedforward Neural Network

Topic
Stochastic gradient descent is a hard case for parallel computing. Data parallelism has nothing to split because each step uses one sample, and each step needs the weights produced by the previous one. All usable parallelism sits inside a single step. Layers run in order, but the neurons within a layer do not depend on each other. In this project we parallelize that inner structure for a feedforward neural network (FFNN) written from scratch.

Theory suggests speedup is possible. In the forward pass, each neuron’s output is its own dot product, so we can split the neurons in a layer across threads. In the backward pass, each neuron’s weight update is also independent. We use ReLU activations, so the derivative is 1 for positive inputs and 0 otherwise, which keeps the backward pass simple. Our simple model for step time has two parts: the time to do the math and read the weights, which shrinks as we add threads, and the time threads spend waiting for each other after each layer, which grows with thread count. Wide layers give each thread a lot of work, so we expect them to speed up. Narrow layers give each thread little work, so we expect waiting to dominate and speedup to be small or negative. The reference code is a serial version of the same network (forward pass, backpropagation, weight update), which we will write ourselves.

Two tracks
The main track splits the neurons of each layer across threads. The hardest part is the backward pass, where passing the error to the previous layer needs a sum across all threads. We will try two ways of handling it and compare them: each thread keeps its own partial result and we add the results at the end, or we store the weight matrix in a second, transposed copy so the work splits cleanly.
The backup track is making the serial matrix-vector multiply faster by reading memory in a cache-friendly order. This also gives us a stronger serial baseline for the speedup numbers. If the parallel work runs into trouble, this track is a complete project on its own.

Learning objectives
Apply theory: we use the simple compute-versus-waiting model to predict which layer widths should speed up and which should not.
Prototype without a scaffold: we write the network, backpropagation, and threading from scratch in C with OpenMP, using no ML framework.
Run on a cluster: we run all parallel experiments on the school server.
Evaluate empirically: we measure time per training step for different thread counts and report speedup and efficiency, then compare against our prediction.
Communicate with peers: we present our approach, results, and the gap between prediction and measurement in a class presentation.

Scope
The project covers fully connected layers, ReLU activations, SGD, and shared-memory threads, using MNIST or synthetic data. Running across multiple machines with MPI is out of scope. Work proceeds in three stages: serial baseline, faster serial multiply, then the parallel version and scaling study. We have a usable result after stage two even if stage three slips.

Feasibility and resources
We write the implementation in C with OpenMP and run it on the school server. Network size is adjustable, so the experiments fit whatever hardware is available. If time or resources are limited, we reduce the number of training steps per run and keep the size and thread sweeps.

Experiments
We will compare hidden layer widths of 64, 256, 1024, and 4096, with 2 or 4 hidden layers, using 1, 2, 4, 8, and up to max cores threads. The experiment is strong scaling: the network is fixed for each run and only the thread count changes. We compare three versions: serial, optimized serial, and parallel. We measure communication as the time threads spend waiting and summing partial results, timed with a clock around each phase.

