# ============================================================================
# Parallel SGD FFNN Project - Makefile
# ============================================================================

# Compiler settings (same as the linsolve project; this is for the cluster)
# If you want to play with C flags, add them here.
CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp -std=c99 -Itest_framework
LDFLAGS = -lm -fopenmp

# On a Mac, `gcc` is really clang, which needs Homebrew's libomp for OpenMP
# (brew install libomp). This block is ignored on the cluster.
ifeq ($(shell uname),Darwin)
LIBOMP = $(shell brew --prefix libomp 2>/dev/null)
CC = clang
CFLAGS = -Wall -Wextra -O2 -Xpreprocessor -fopenmp -I$(LIBOMP)/include -std=c99 -Itest_framework
LDFLAGS = -lm -L$(LIBOMP)/lib -lomp
endif

# Hidden-layer widths for timing (hidden layer counts 2 and 4 are set in
# the timing program).
TIMING_WIDTHS = 64 256 1024 4096

# MNIST download mirror (the original yann.lecun.com links are unreliable)
MNIST_URL = https://ossci-datasets.s3.amazonaws.com/mnist
MNIST_FILES = train-images-idx3-ubyte train-labels-idx1-ubyte \
              t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

# Shared framework sources
FRAMEWORK = test_framework/network.c test_framework/mnist.c
HEADERS = test_framework/ffnn.h test_framework/mnist.h

# ================
# Student Targets
# ================

# Get help on available commands
help:
	@echo "============================================================================"
	@echo "Parallel SGD FFNN Project - Available make Commands"
	@echo "============================================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make data           - Download MNIST into data/ (run on the login node)"
	@echo ""
	@echo "Testing your implementations:"
	@echo "  make baseline_test  - Test baseline (serial) implementation"
	@echo ""
	@echo "Performance timing:"
	@echo "  make baseline_time  - Time baseline serial SGD step, write CSV + plot"
	@echo ""
	@echo "Utility commands:"
	@echo "  make clean          - Remove all compiled files and results"
	@echo "  make help           - Show this help message"
	@echo ""

# Remove all generated files (keeps data/)
clean:
	@echo "Cleaning up all generated files..."
	rm -rf bin/
	rm -f *.csv *.png
	@echo "Clean complete."

# Download and unpack MNIST
data: $(addprefix data/,$(MNIST_FILES))

data/%:
	@mkdir -p data
	curl -fL -o $@.gz $(MNIST_URL)/$*.gz
	gunzip -f $@.gz

# Test baseline (serial) implementation
baseline_test: bin/test_baseline
	@echo "Testing baseline (serial) FFNN..."
	./bin/test_baseline

# Time baseline serial implementation
baseline_time: bin/timing_baseline
	@echo "Running baseline FFNN timing analysis..."
	./bin/timing_baseline baseline_timing_results.csv $(TIMING_WIDTHS)
	@echo ""
	@if command -v python3 >/dev/null 2>&1; then \
		python3 test_framework/analyze_baseline.py baseline_timing_results.csv; \
	else \
		echo "Python3 not found - skipping plot generation"; \
		echo "CSV results saved to baseline_timing_results.csv"; \
	fi

# ============================================================================
# Build Rules
# ============================================================================

# Create bin directory if it doesn't exist
bin:
	mkdir -p bin

# Test executables
bin/test_baseline: test_framework/test_baseline_nn.c baseline_nn.c $(FRAMEWORK) $(HEADERS) | bin
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

# Timing executables
bin/timing_baseline: test_framework/timing_baseline.c baseline_nn.c $(FRAMEWORK) $(HEADERS) | bin
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

# Mark targets as phony
.PHONY: help clean data baseline_test baseline_time
