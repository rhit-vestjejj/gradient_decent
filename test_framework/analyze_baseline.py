#!/usr/bin/env python3
"""
Baseline FFNN Timing Analysis
Reads the CSV from timing_baseline and produces a tabular summary plus plots:
  - time per SGD step vs hidden-layer width (2 and 4 hidden layers)
  - breakdown of each step into forward / backward / update
  - throughput (GFLOP/s) vs width
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

EXPECTED_COLS = ['impl', 'hidden_layers', 'width', 'threads', 'params', 'steps',
                 'time_per_step', 'forward', 'backward', 'update']


def add_flops(df):
    # Each phase is ~1 multiply-add per weight -> 2 flops per weight per phase,
    # so one SGD step (forward + backward + update) is ~6 flops per weight.
    df = df.copy()
    df['gflops'] = 6.0 * df['params'] / df['time_per_step'] / 1e9
    return df


def create_plots(df, output_prefix='baseline'):
    serial = df[df['impl'] == 'serial']
    layer_counts = sorted(serial['hidden_layers'].unique())
    markers = {2: 'o', 4: 's'}
    colors = {2: 'tab:blue', 4: 'tab:orange'}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline Serial FFNN (MNIST, SGD, float32) Performance',
                 fontsize=16, fontweight='bold')

    # Plot 1: time per step vs width
    for hl in layer_counts:
        d = serial[serial['hidden_layers'] == hl].sort_values('width')
        ax1.loglog(d['width'], d['time_per_step'] * 1e3, marker=markers.get(hl, 'o'),
                   color=colors.get(hl), linewidth=2, base=2,
                   label=f'{hl} hidden layers')
    ax1.set_xlabel('Hidden layer width')
    ax1.set_ylabel('Time per SGD step (ms)')
    ax1.set_title('Time per Training Step vs Width')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    # Plots 2 & 3: phase breakdown (fraction of step time) per layer count
    for ax, hl in zip((ax2, ax3), layer_counts[:2]):
        d = serial[serial['hidden_layers'] == hl].sort_values('width')
        labels = [str(w) for w in d['width']]
        fwd = d['forward'] / d['time_per_step']
        bwd = d['backward'] / d['time_per_step']
        upd = d['update'] / d['time_per_step']
        ax.bar(labels, fwd, label='forward', color='tab:green')
        ax.bar(labels, bwd, bottom=fwd, label='backward', color='tab:red')
        ax.bar(labels, upd, bottom=fwd + bwd, label='update', color='tab:purple')
        ax.set_xlabel('Hidden layer width')
        ax.set_ylabel('Fraction of step time')
        ax.set_title(f'Phase Breakdown ({hl} hidden layers)')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)

    # Plot 4: throughput
    for hl in layer_counts:
        d = serial[serial['hidden_layers'] == hl].sort_values('width')
        ax4.semilogx(d['width'], d['gflops'], marker=markers.get(hl, 'o'),
                     color=colors.get(hl), linewidth=2, base=2,
                     label=f'{hl} hidden layers')
    ax4.set_xlabel('Hidden layer width')
    ax4.set_ylabel('GFLOP/s (approx. 6 flops / weight / step)')
    ax4.set_title('Serial Throughput vs Width')
    ax4.set_ylim(bottom=0)
    ax4.legend()
    ax4.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plot_filename = f'{output_prefix}_analysis.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.close()


def print_summary_table(df):
    print("\n" + "=" * 104)
    print("BASELINE SERIAL FFNN TIMING SUMMARY")
    print("=" * 104)
    print(f"{'Hidden':<8} {'Width':<8} {'Params':<12} {'ms/step':<12} "
          f"{'Forward':<14} {'Backward':<14} {'Update':<14} {'GFLOP/s':<10}")
    print(f"{'layers':<8} {'':<8} {'':<12} {'':<12} "
          f"{'ms (%)':<14} {'ms (%)':<14} {'ms (%)':<14} {'':<10}")
    print("-" * 104)
    for _, r in df.sort_values(['impl', 'hidden_layers', 'width']).iterrows():
        t = r['time_per_step']

        def cell(v):
            return f"{v * 1e3:.3f} ({100 * v / t:.0f}%)"

        print(f"{int(r['hidden_layers']):<8} {int(r['width']):<8} {int(r['params']):<12} "
              f"{t * 1e3:<12.3f} {cell(r['forward']):<14} {cell(r['backward']):<14} "
              f"{cell(r['update']):<14} {r['gflops']:<10.2f}")
    print("=" * 104)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_baseline.py <input.csv>")
        sys.exit(1)
    csv_file = sys.argv[1]

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file}")
        sys.exit(1)

    if not all(col in df.columns for col in EXPECTED_COLS):
        print(f"Error: CSV must contain columns: {EXPECTED_COLS}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    df = add_flops(df)
    print_summary_table(df)
    create_plots(df)


if __name__ == "__main__":
    main()
