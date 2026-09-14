"""
Persistent bias of the project-then-average graph projection heuristic.

Builds a symmetric target M from 25 iterations of the matrix-free ADMM loop
(Chebyshev oracle) on the n=50 random geometric graph, then runs the
heuristic `decentralized_graph_projection` for K = 100 ... 3000 sweeps and
reports the relative error to the centralized projection. Also reports the
same sweeps without the final rescaling passes (with the resulting maximum
row-sum violation) and with the Metropolis-Hastings matrix as the target.
"""
import networkx as nx
import numpy as np

from des_fmmc_admm import _make_V_matvec
from decentralized_chebyshev import CommLedger
from decentralized_graph_projection import (
    adaptive_omega, build_topology, decentralized_graph_projection,
)
from fmmc_exact_admm import B_from_w, EdgeGraphProjection, metropolis_hastings_weights
from spectral_oracle import oracle_chebyshev

SWEEPS = (100, 300, 1000, 3000)


def admm_target(adj, projector, rho=0.1, iters=25):
    """Target M = J + Z - U after `iters` matrix-free ADMM iterations, and the
    last projected w (used as warm start)."""
    n = adj.shape[0]
    J = np.ones((n, n)) / n
    w_mh = metropolis_hastings_weights(adj)
    w = np.array([w_mh[i, j] for (i, j) in projector.edges])
    B = B_from_w(projector.edges, w, n)
    Q_ext, delta_evals = np.zeros((n, 0)), np.zeros(0)
    U = np.zeros((n, n))
    Z = B - J
    s = 0.0
    for _ in range(iters):
        w, B, _, _ = projector.project(J + Z - U, w0=w)
        V = B - J + U
        V = 0.5 * (V + V.T)
        res = oracle_chebyshev(_make_V_matvec(B, Q_ext, delta_evals), n, rho,
                               np.abs(V).sum(axis=1), s_estimate=s if s > 0 else 0.1,
                               delta_t=1e-6, block_size=4)
        s = res.s_hat
        Q_ext = res.Q_active
        delta_evals = np.clip(res.Lambda_active, -s, s) - res.Lambda_active
        low_rank = Q_ext @ np.diag(delta_evals) @ Q_ext.T if Q_ext.shape[1] else 0.0
        Z_next = 0.5 * ((V + low_rank) + (V + low_rank).T)
        U = U + (B - J - Z_next)
        Z = Z_next
    return J + Z - U, w


def heuristic_error(adj, projector, M, w0, K, rescale_passes=8):
    n = adj.shape[0]
    neighbors, edge_index_by_node = build_topology(projector.edges, n)
    M_ij = np.array([M[i, j] for (i, j) in projector.edges])
    w_ref, _, _, _ = projector.project(M, w0=w0)
    w_dec, S = decentralized_graph_projection(
        projector.edges, n, np.diag(M).copy(), M_ij, w0.copy(), neighbors,
        edge_index_by_node, K, CommLedger(), omega=adaptive_omega(adj),
        rescale_passes=rescale_passes)
    err = np.linalg.norm(w_dec - w_ref) / max(np.linalg.norm(w_ref), 1e-12)
    return err, max(S.max() - 1.0, 0.0)


def main():
    n = 50
    adj = nx.to_numpy_array(nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=42), dtype=float)
    projector = EdgeGraphProjection(adj)
    M, w0 = admm_target(adj, projector)

    print("ADMM target, with final rescaling:")
    for K in SWEEPS:
        err, _ = heuristic_error(adj, projector, M, w0, K)
        print(f"  K={K:5d}  rel_err={err:.6f}")
    print("ADMM target, without final rescaling:")
    for K in SWEEPS:
        err, viol = heuristic_error(adj, projector, M, w0, K, rescale_passes=0)
        print(f"  K={K:5d}  rel_err={err:.4f}  max row-sum violation={viol:.3f}")

    B_mh = B_from_w(projector.edges, np.array(
        [metropolis_hastings_weights(adj)[i, j] for (i, j) in projector.edges]), n)
    err, _ = heuristic_error(adj, projector, B_mh, np.zeros(projector.m), 800)
    print(f"Metropolis-Hastings target, K=800: rel_err={err:.2e}")


if __name__ == "__main__":
    main()
