"""
Validates the consensus-ADMM graph projection (`decentralized_graph_projection.py::
decentralized_graph_projection_consensus`) against the OLD Jacobi+symmetrization
heuristic it replaces (the old scheme was found, Section 15.7, to
converge to a measured, non-vanishing bias for at least some M encountered
on real trajectories, meaning Theorem 1's eta_t -> 0 requirement is not
satisfiable by it, no matter how many sweeps are run).

For each of this paper's four standard graphs, finds an adversarial M (one
that triggers the OLD scheme's plateau -- not every M does, Section 15.7
already found this is M-dependent) via a seed sweep, then reports both
schemes' convergence up to 3000 sweeps on that SAME M, head-to-head.
"""
import numpy as np
import networkx as nx

from fmmc_exact_admm import EdgeGraphProjection
from decentralized_graph_projection import (
    decentralized_graph_projection, decentralized_graph_projection_consensus,
    build_topology, adaptive_omega,
)
from decentralized_chebyshev import CommLedger
from experiment_backend_comparison import build_graphs


def random_M(n, seed, scale=0.5):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) * scale
    M = 0.5 * (M + M.T) + np.eye(n) * 0.5
    return M


def find_adversarial_M(adj, n_seeds=8, sweep_check=800):
    n = adj.shape[0]
    projector = EdgeGraphProjection(adj)
    neighbors, edge_index_by_node = build_topology(projector.edges, n)
    omega = adaptive_omega(adj)
    worst_seed, worst_err = None, 0.0
    for seed in range(n_seeds):
        M = random_M(n, seed)
        M_ii = np.diag(M).copy()
        M_ij = np.array([M[i, j] for (i, j) in projector.edges])
        w_ref, _, _, _ = projector.project(M)
        ledger = CommLedger()
        w_old, _ = decentralized_graph_projection(
            projector.edges, n, M_ii, M_ij, np.zeros(projector.m),
            neighbors, edge_index_by_node, sweep_check, ledger, omega=omega)
        err = np.linalg.norm(w_old - w_ref) / max(np.linalg.norm(w_ref), 1e-12)
        if err > worst_err:
            worst_err, worst_seed = err, seed
    return worst_seed, worst_err


def head_to_head(gname, adj, seed, sweeps_list=(10, 50, 200, 800), rho_cons=2.0):
    n = adj.shape[0]
    projector = EdgeGraphProjection(adj)
    neighbors, edge_index_by_node = build_topology(projector.edges, n)
    omega = adaptive_omega(adj)
    M = random_M(n, seed)
    M_ii = np.diag(M).copy()
    M_ij = np.array([M[i, j] for (i, j) in projector.edges])
    w_ref, _, _, _ = projector.project(M)

    print(f"\n{gname} (n={n}, m={projector.m}, adversarial seed={seed}):")
    rows = []
    for K in sweeps_list:
        ledger_old = CommLedger()
        w_old, S_old = decentralized_graph_projection(
            projector.edges, n, M_ii, M_ij, np.zeros(projector.m),
            neighbors, edge_index_by_node, K, ledger_old, omega=omega)
        err_old = np.linalg.norm(w_old - w_ref) / max(np.linalg.norm(w_ref), 1e-12)

        ledger_new = CommLedger()
        w_new, S_new = decentralized_graph_projection_consensus(
            projector.edges, n, M_ii, M_ij, np.zeros(projector.m),
            neighbors, edge_index_by_node, K, ledger_new, rho_cons=rho_cons)
        err_new = np.linalg.norm(w_new - w_ref) / max(np.linalg.norm(w_ref), 1e-12)

        print(f"  sweeps={K:5d} | OLD err={err_old:.4e} (rounds={ledger_old.rounds:6d}) | "
              f"NEW err={err_new:.4e} (rounds={ledger_new.rounds:6d})")
        rows.append(dict(K=K, err_old=err_old, rounds_old=ledger_old.rounds,
                          err_new=err_new, rounds_new=ledger_new.rounds))
    return rows


def run():
    graphs = build_graphs()
    results = {}
    for gname, adj in graphs.items():
        seed, err = find_adversarial_M(adj)
        print(f"{gname}: least favorable of 8 seeds, heuristic error at 800 sweeps = {err:.4e} (seed={seed})")
        results[gname] = head_to_head(gname, adj, seed)
    return results


if __name__ == "__main__":
    run()
