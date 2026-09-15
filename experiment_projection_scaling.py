"""
Dependence of the consensus-ADMM graph projection on graph size.

For cycles, hypercubes, and random geometric graphs (radius sqrt(8/n), so
the expected degree stays roughly constant) of increasing size, records the
number of sweeps needed to reach relative error <= 1e-8 against the
centralized projection (rho_c = 2, cold start w = 0), and the relative
error of the project-then-average heuristic after the same budget.
Targets are random symmetric matrices (random_M, seed 0).

Usage: python experiment_projection_scaling.py [output.json]
"""
import json
import sys
import time

import networkx as nx
import numpy as np

from decentralized_chebyshev import CommLedger
from decentralized_graph_projection import (
    adaptive_omega, build_topology, decentralized_graph_projection,
    decentralized_graph_projection_consensus,
)
from experiment_consensus_projection import random_M
from fmmc_exact_admm import EdgeGraphProjection

TOL = 1e-8
MAX_SWEEPS = 2000
RHO_C = 2.0


def rgg(n, seed=42):
    s = seed
    while True:
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=s)
        if nx.is_connected(G):
            return G
        s += 1


def graphs():
    for n in (10, 50, 200, 800):
        yield f"cycle_n{n}", nx.cycle_graph(n)
    for d in (4, 6, 8):
        yield f"hypercube_n{2**d}", nx.hypercube_graph(d)
    for n in (50, 200, 800):
        yield f"rgg_n{n}", rgg(n)


def run(G):
    G = nx.convert_node_labels_to_integers(G)
    adj = nx.to_numpy_array(G, dtype=float)
    n = adj.shape[0]
    proj = EdgeGraphProjection(adj)
    neighbors, edge_index_by_node = build_topology(proj.edges, n)
    M = random_M(n, 0)
    M_ii = np.diag(M).copy()
    M_ij = np.array([M[i, j] for (i, j) in proj.edges])
    w_ref = proj.project(M)[0]
    ref = max(np.linalg.norm(w_ref), 1e-12)

    state = {"k": 0, "hit": None}

    def cb(z):
        state["k"] += 1
        if np.linalg.norm(z - w_ref) / ref <= TOL:
            state["hit"] = state["k"]
            return True
        return False

    t0 = time.time()
    decentralized_graph_projection_consensus(
        proj.edges, n, M_ii, M_ij, np.zeros(proj.m), neighbors, edge_index_by_node,
        MAX_SWEEPS, CommLedger(), rho_cons=RHO_C, callback=cb)
    t_new = time.time() - t0
    K = state["hit"] or MAX_SWEEPS
    w_old, _ = decentralized_graph_projection(
        proj.edges, n, M_ii, M_ij, np.zeros(proj.m), neighbors, edge_index_by_node,
        K, CommLedger(), omega=adaptive_omega(adj))
    degs = adj.sum(axis=1)
    return {"n": n, "m": int(proj.m), "max_degree": int(degs.max()),
            "mean_degree": float(degs.mean()), "sweeps_to_tol": state["hit"],
            "heuristic_err_same_budget": float(np.linalg.norm(w_old - w_ref) / ref),
            "consensus_seconds": t_new}


def main(out_path):
    results = {}
    for name, G in graphs():
        r = run(G)
        results[name] = r
        print(f"{name:16s} n={r['n']:4d} m={r['m']:5d} dmax={r['max_degree']:2d} "
              f"sweeps_to_1e-8={r['sweeps_to_tol']} heuristic_err={r['heuristic_err_same_budget']:.2e} "
              f"({r['consensus_seconds']:.1f}s)", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "projection_scaling.json")
