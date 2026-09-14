"""
Minimal instance of the heuristic's bias: the five-node cycle.

Selects the least favorable of 40 random targets for the project-then-average
heuristic at 800 sweeps, then prints the per-edge values of the centralized
projection, the heuristic, and the consensus-ADMM projection (rho_c = 2),
together with the relative errors at several sweep budgets.
"""
import networkx as nx
import numpy as np

from decentralized_chebyshev import CommLedger
from decentralized_graph_projection import (
    adaptive_omega, build_topology, decentralized_graph_projection,
    decentralized_graph_projection_consensus,
)
from fmmc_exact_admm import EdgeGraphProjection


def random_M(n, seed, scale=0.5):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) * scale
    return 0.5 * (M + M.T) + np.eye(n) * 0.5


def run(projector, adj, M, K):
    n = adj.shape[0]
    neighbors, edge_index_by_node = build_topology(projector.edges, n)
    M_ii = np.diag(M).copy()
    M_ij = np.array([M[i, j] for (i, j) in projector.edges])
    w0 = np.zeros(projector.m)
    w_old, _ = decentralized_graph_projection(
        projector.edges, n, M_ii, M_ij, w0, neighbors, edge_index_by_node, K,
        CommLedger(), omega=adaptive_omega(adj))
    w_new, _ = decentralized_graph_projection_consensus(
        projector.edges, n, M_ii, M_ij, w0, neighbors, edge_index_by_node, K,
        CommLedger(), rho_cons=2.0)
    return w_old, w_new


def rel_err(w, w_ref):
    return np.linalg.norm(w - w_ref) / max(np.linalg.norm(w_ref), 1e-12)


def main():
    adj = nx.to_numpy_array(nx.cycle_graph(5), dtype=float)
    projector = EdgeGraphProjection(adj)

    def heuristic_err(seed):
        M = random_M(5, seed)
        return rel_err(run(projector, adj, M, 800)[0], projector.project(M)[0])

    seed = max(range(40), key=heuristic_err)
    M = random_M(5, seed)
    w_ref = projector.project(M)[0]
    np.set_printoptions(precision=4, suppress=True)
    print(f"seed={seed}  edges={projector.edges}")
    print("M_ii      :", np.diag(M))
    print("M_e       :", np.array([M[i, j] for (i, j) in projector.edges]))
    print("w* (QP)   :", w_ref)
    for K in (10, 50, 200, 800):
        w_old, w_new = run(projector, adj, M, K)
        print(f"K={K:4d}  heuristic rel_err={rel_err(w_old, w_ref):.4e}  "
              f"consensus rel_err={rel_err(w_new, w_ref):.4e}")
    print("heuristic :", w_old)
    print("consensus :", w_new)


if __name__ == "__main__":
    main()
