"""
Sensitivity of the consensus-ADMM graph projection to the penalty rho_c.

For each standard graph, uses the same adversarial target M as
experiment_consensus_projection.py (worst of 8 seeds for the heuristic at
800 sweeps) and records, for each rho_c, the number of sweeps needed to
reach a relative error <= 1e-6 and <= 1e-8 against the centralized
solution, plus the error at the end of the run (at most 800 sweeps).
"""
import json
import sys

import numpy as np

from fmmc_exact_admm import EdgeGraphProjection
from decentralized_graph_projection import (
    decentralized_graph_projection_consensus, build_topology,
)
from decentralized_chebyshev import CommLedger
from experiment_backend_comparison import build_graphs
from experiment_consensus_projection import find_adversarial_M, random_M

RHOS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MAX_SWEEPS = 800
TOLS = (1e-6, 1e-8)
STOP_BELOW = 1e-11


def run_one(adj, M, rho):
    n = adj.shape[0]
    proj = EdgeGraphProjection(adj)
    neighbors, edge_index_by_node = build_topology(proj.edges, n)
    M_ii = np.diag(M).copy()
    M_ij = np.array([M[i, j] for (i, j) in proj.edges])
    w_ref, _, _, _ = proj.project(M)
    ref_norm = max(np.linalg.norm(w_ref), 1e-12)

    state = {"k": 0, "hit": {t: None for t in TOLS}, "err": np.nan}

    def cb(z):
        state["k"] += 1
        err = np.linalg.norm(z - w_ref) / ref_norm
        state["err"] = err
        for t in TOLS:
            if state["hit"][t] is None and err <= t:
                state["hit"][t] = state["k"]
        return err <= STOP_BELOW

    decentralized_graph_projection_consensus(
        proj.edges, n, M_ii, M_ij, np.zeros(proj.m), neighbors,
        edge_index_by_node, MAX_SWEEPS, CommLedger(), rho_cons=rho, callback=cb)
    return {"sweeps_to_1e-6": state["hit"][1e-6], "sweeps_to_1e-8": state["hit"][1e-8],
            "final_err": float(state["err"]), "sweeps_run": state["k"]}


def main(out_path):
    results = {}
    for gname, adj in build_graphs().items():
        seed, _ = find_adversarial_M(adj)
        M = random_M(adj.shape[0], seed)
        results[gname] = {"seed": int(seed), "rho": {}}
        for rho in RHOS:
            r = run_one(adj, M, rho)
            results[gname]["rho"][str(rho)] = r
            print(f"{gname:22s} rho={rho:<5} 1e-6@{r['sweeps_to_1e-6']} "
                  f"1e-8@{r['sweeps_to_1e-8']} final={r['final_err']:.2e} "
                  f"(ran {r['sweeps_run']})", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rho_sensitivity.json")
