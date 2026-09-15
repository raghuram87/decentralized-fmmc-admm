"""
Topological scope of the SLEM = 1 fixed point observed on the 50-node cycle.

Runs the ADMM (rho = 0.1, 800 iterations, residual stopping criterion) with
the Chebyshev and the full-EVD spectral backends on bipartite graphs (even
cycles, hypercubes, a path, a grid, a binary tree) and non-bipartite graphs
(odd cycles, the Petersen graph, a random geometric graph), and compares the
final SLEM with the CLARABEL solution of the FMMC SDP (closed form for
cycles with n > 64).

Usage: python experiment_bipartite_scope.py [output.json]
"""
import json
import sys
import time

import networkx as nx
import numpy as np

from centralized_reference import solve_fmmc_cvxpy
from des_fmmc_admm import des_fmmc_admm


def rgg(n, seed=42):
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=seed)
    assert nx.is_connected(G)
    return G


def graphs():
    yield "cycle_n20", nx.cycle_graph(20)
    yield "cycle_n50", nx.cycle_graph(50)
    yield "cycle_n100", nx.cycle_graph(100)
    yield "hypercube_n16", nx.hypercube_graph(4)
    yield "hypercube_n64", nx.hypercube_graph(6)
    yield "path_n20", nx.path_graph(20)
    yield "grid_5x10", nx.grid_2d_graph(5, 10)
    yield "binary_tree_n31", nx.balanced_tree(2, 4)
    yield "cycle_n21", nx.cycle_graph(21)
    yield "cycle_n51", nx.cycle_graph(51)
    yield "petersen_n10", nx.petersen_graph()
    yield "rgg_n50", rgg(50)


def reference_slem(name, adj):
    n = adj.shape[0]
    if name.startswith("cycle") and n > 64:
        if n % 2 == 0:
            c = np.cos(2 * np.pi / n)
            return (1 + c) / (3 - c), "closed form"
        return None, "none"
    _, s, status, solver = solve_fmmc_cvxpy(adj)
    return float(s), solver


def main(out_path):
    results = {}
    for name, G in graphs():
        G = nx.convert_node_labels_to_integers(G)
        adj = nx.to_numpy_array(G, dtype=float)
        ref, ref_src = reference_slem(name, adj)
        row = {"n": adj.shape[0], "m": int(adj.sum() // 2), "bipartite": nx.is_bipartite(G),
               "reference": ref, "reference_source": ref_src}
        for backend in ("chebyshev", "full_evd"):
            t0 = time.time()
            r = des_fmmc_admm(adj, backend=backend, rho=0.1, num_iter=800, verbose=False)
            row[backend] = {"iterations": len(r.history), "final_slem": float(r.history[-1].slem),
                            "best_slem": float(r.best_slem), "best_iteration": int(r.best_iteration),
                            "min_self_loop": float(np.diag(r.B).min()),
                            "seconds": time.time() - t0}
        results[name] = row
        print(name, json.dumps(row), flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bipartite_scope.json")
