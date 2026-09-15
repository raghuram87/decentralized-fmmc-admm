"""
Sweeps needed per graph-projection target along the outer ADMM trajectory
for the three initializations of experiment_warm_start.py, on one graph.

Usage: python make_warm_start_figure.py warm_start.json [graph] [output.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = (  # key, label, color, marker, linestyle
    ("cold", "cold", "#2a78d6", "o", "-"),
    ("primal", "primal warm", "#eb6834", "^", "--"),
    ("full", "primal–dual warm", "#1baf7a", "s", "-"),
)


def main(data_path, graph="random_geometric_n50", out_path="warm_start_sweeps.png"):
    rows = json.load(open(data_path))[graph]["rows"]
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    for key, label, color, marker, ls in SERIES:
        ax.plot(t, [r[key] for r in rows], ls, color=color, linewidth=1.4, label=label,
                marker=marker, markersize=3.2, markevery=10)
    ax.set_xlabel("outer ADMM iteration", fontsize=8)
    ax.set_ylabel("sweeps to $10^{-8}$", fontsize=8)
    ax.tick_params(labelsize=7, color="#9a9a96")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a96")
    ax.grid(True, color="#e4e4e0", linewidth=0.6)
    ax.set_xlim(0, len(rows) - 1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    print("saved", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
