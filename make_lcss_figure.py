"""
Convergence-curve figure for the graph-feasibility projection letter:
relative error to the centralized projection P(M) versus sweeps K for the
project-then-average heuristic and the consensus-ADMM projection
(rho_c = 2), on the four graphs of experiment_consensus_projection.py.

Usage: python make_lcss_figure.py [output.png]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_consensus_projection import build_graphs, find_adversarial_M, head_to_head

SWEEPS = (10, 25, 50, 100, 200, 400, 800)
TITLES = {
    "random_geometric_n50": "RGG, n=50",
    "cycle_n50": "Cycle, n=50",
    "hypercube_n64": "Hypercube, n=64",
    "petersen_n10": "Petersen, n=10",
}


def main(out_path):
    data = {}
    for gname, adj in build_graphs().items():
        seed, _ = find_adversarial_M(adj)
        data[gname] = head_to_head(gname, adj, seed, sweeps_list=SWEEPS)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(data, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.35), sharex=True)
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, TITLES.items()):
        rows = data[key]
        K = [r["K"] for r in rows]
        ax.semilogy(K, [r["err_old"] for r in rows], "o-", color="#b03030",
                    label="heuristic", markersize=5.5, linewidth=1.8)
        ax.semilogy(K, [r["err_new"] for r in rows], "s-", color="#2060a0",
                    label="consensus-ADMM", markersize=5.5, linewidth=1.8)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("sweeps $K$", fontsize=12)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("relative error to $P(M)$", fontsize=12)
    axes[0].legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "convergence_curves.png")
