#!/usr/bin/env python3
"""Strong-scaling analysis for the parallel FFNN implementations."""

import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPECTED_COLS = ['impl', 'hidden_layers', 'width', 'threads', 'params', 'steps',
                 'time_per_step', 'forward', 'backward', 'update']
NAMES = {'partial': 'Partial sums', 'transposed': 'Transposed copy'}


def serial_time(df, hl, width):
    r = df[(df['impl'] == 'serial') & (df['hidden_layers'] == hl) & (df['width'] == width)]
    return r['time_per_step'].iloc[0] if not r.empty else float('nan')


def create_plots(df, output_prefix='parallel'):
    impls = [m for m in ['partial', 'transposed'] if m in set(df['impl'])]
    layer_counts = sorted(df['hidden_layers'].unique())
    widths = sorted(df['width'].unique())
    threads = sorted(df[df['impl'] != 'serial']['threads'].unique())
    cmap = plt.get_cmap('viridis')
    colors = {w: cmap(k / max(1, len(widths) - 1)) for k, w in enumerate(widths)}

    fig, axes = plt.subplots(len(layer_counts), 3, figsize=(20, 6 * len(layer_counts)), squeeze=False)
    fig.suptitle('Parallel FFNN Strong Scaling (MNIST, SGD, float32)', fontsize=16, fontweight='bold')

    for r, hl in enumerate(layer_counts):
        for c, m in enumerate(impls):
            ax = axes[r][c]
            for w in widths:
                d = df[(df['impl'] == m) & (df['hidden_layers'] == hl) & (df['width'] == w)].sort_values('threads')
                if d.empty:
                    continue
                base = d[d['threads'] == d['threads'].min()]['time_per_step'].iloc[0]
                ax.plot(d['threads'], base / d['time_per_step'], marker='o', linewidth=2,
                        color=colors[w], label=f'width {w}')
            ax.plot(threads, threads, '--', color='black', alpha=0.5, label='ideal')
            ax.set_xlabel('Threads')
            ax.set_ylabel('Speedup vs same impl at 1 thread')
            ax.set_title(f'{NAMES[m]} ({hl} hidden layers)')
            ax.set_xticks(threads)
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax = axes[r][2]
        for w in widths:
            st = serial_time(df, hl, w)
            for m, style in zip(impls, ['-', '--']):
                d = df[(df['impl'] == m) & (df['hidden_layers'] == hl) & (df['width'] == w)].sort_values('threads')
                if d.empty:
                    continue
                ax.plot(d['threads'], st / d['time_per_step'], marker='o', linestyle=style,
                        linewidth=2, color=colors[w], label=f'{m}, width {w}')
        ax.axhline(1.0, color='black', linestyle=':', alpha=0.5)
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup vs serial baseline')
        ax.set_yscale('log', base=2)
        ax.set_title(f'Speedup vs Serial Baseline ({hl} hidden layers)')
        ax.set_xticks(threads)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plot_filename = f'{output_prefix}_analysis.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    plt.close()


def print_summary_table(df):
    threads = sorted(df[df['impl'] != 'serial']['threads'].unique())
    for m in [x for x in ['partial', 'transposed'] if x in set(df['impl'])]:
        print("\n" + "=" * (32 + 18 * len(threads)))
        print(f"{NAMES[m].upper()}: ms/step and speedup vs serial baseline")
        print("=" * (32 + 18 * len(threads)))
        print(f"{'Hidden':<8}{'Width':<8}{'Serial ms':<16}" + "".join(f"{str(t) + 'T':<18}" for t in threads))
        print("-" * (32 + 18 * len(threads)))
        for hl in sorted(df['hidden_layers'].unique()):
            for w in sorted(df['width'].unique()):
                st = serial_time(df, hl, w)
                row = f"{hl:<8}{w:<8}{st * 1e3:<16.3f}"
                for t in threads:
                    r = df[(df['impl'] == m) & (df['hidden_layers'] == hl) & (df['width'] == w) & (df['threads'] == t)]
                    if r.empty:
                        row += f"{'N/A':<18}"
                    else:
                        v = r['time_per_step'].iloc[0]
                        row += f"{f'{v * 1e3:.3f}/{st / v:.2f}x':<18}"
                print(row)

    print("\nBEST CONFIGURATION PER NETWORK:")
    print("-" * 70)
    par = df[df['impl'] != 'serial']
    for (hl, w), g in par.groupby(['hidden_layers', 'width']):
        best = g.loc[g['time_per_step'].idxmin()]
        st = serial_time(df, hl, w)
        print(f"  {hl} x {w:<6} {best['impl']} @ {int(best['threads'])}T: "
              f"{best['time_per_step'] * 1e3:.3f} ms/step, {st / best['time_per_step']:.2f}x vs serial")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_parallel.py <input.csv>")
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1])
    except FileNotFoundError:
        print(f"Error: Could not find file {sys.argv[1]}")
        sys.exit(1)
    if not all(col in df.columns for col in EXPECTED_COLS):
        print(f"Error: CSV must contain columns: {EXPECTED_COLS}")
        sys.exit(1)
    print_summary_table(df)
    create_plots(df)


if __name__ == "__main__":
    main()
