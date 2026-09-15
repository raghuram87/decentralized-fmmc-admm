"""
Warm starting the consensus-ADMM graph projection inside an outer ADMM.

Generates a sequence of targets M_t = J + Z^t - U^t from T iterations of the
matrix-free FMMC ADMM (Chebyshev oracle, rho = 0.1, exact centralized
projection, so the target sequence does not depend on the inner solver).
For every target, the consensus-ADMM projection (rho_c = 2) is run until
its relative error to the centralized projection is <= 1e-8, under three
initializations:
  cold    z = 0,           u = 0
  primal  z = z_{t-1},     u = 0
  full    z = z_{t-1},     u = u_{t-1}   (final iterates for the previous target)
The project-then-average heuristic, warm-started from its own previous
output, is run for the number of sweeps used by the cold start, and its
relative error is recorded.

Usage: python experiment_warm_start.py [output.json]
"""
import json
import sys

import numpy as np

from decentralized_chebyshev import CommLedger
from decentralized_graph_projection import (
    adaptive_omega, build_topology, decentralized_graph_projection,
    decentralized_graph_projection_consensus,
)
from des_fmmc_admm import _make_V_matvec
from experiment_backend_comparison import build_graphs
from fmmc_exact_admm import B_from_w, EdgeGraphProjection, metropolis_hastings_weights
from spectral_oracle import oracle_chebyshev

T = 100
TOL = 1e-8
MAX_SWEEPS = 2000
RHO_C = 2.0


def admm_targets(adj, projector, iters, rho=0.1):
    """Yield the graph-projection target M_t of each outer ADMM iteration."""
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
        M = J + Z - U
        yield M
        w, B, _, _ = projector.project(M, w0=w)
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


def run_graph(adj):
    n = adj.shape[0]
    proj = EdgeGraphProjection(adj)
    neighbors, edge_index_by_node = build_topology(proj.edges, n)
    omega = adaptive_omega(adj)
    zeros = np.zeros(proj.m)
    prev = {"primal": (zeros, None), "full": (zeros, None)}
    w_heur = zeros
    rows = []
    for t, M in enumerate(admm_targets(adj, proj, T)):
        M_ii = np.diag(M).copy()
        M_ij = np.array([M[i, j] for (i, j) in proj.edges])
        w_ref = proj.project(M)[0]
        ref = max(np.linalg.norm(w_ref), 1e-12)

        def solve(z0, lam0):
            k = [0]

            def cb(z):
                k[0] += 1
                return np.linalg.norm(z - w_ref) / ref <= TOL

            z, _, lam = decentralized_graph_projection_consensus(
                proj.edges, n, M_ii, M_ij, z0, neighbors, edge_index_by_node,
                MAX_SWEEPS, CommLedger(), rho_cons=RHO_C, callback=cb,
                lam_init=lam0, return_duals=True)
            err = np.linalg.norm(z - w_ref) / ref
            return (k[0] if err <= TOL else None), z, lam

        row = {"t": t}
        k_cold, _, _ = solve(zeros, None)
        row["cold"] = k_cold
        k_p, z_p, _ = solve(prev["primal"][0], None)
        row["primal"] = k_p
        prev["primal"] = (z_p, None)
        k_f, z_f, lam_f = solve(*prev["full"])
        row["full"] = k_f
        prev["full"] = (z_f, lam_f)

        w_heur, _ = decentralized_graph_projection(
            proj.edges, n, M_ii, M_ij, w_heur, neighbors, edge_index_by_node,
            k_cold or MAX_SWEEPS, CommLedger(), omega=omega)
        row["heuristic_err"] = float(np.linalg.norm(w_heur - w_ref) / ref)
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for key in ("cold", "primal", "full"):
        ks = [r[key] for r in rows]
        ok = [k for k in ks if k is not None]
        out[key] = {"total": int(sum(ok)), "failures": len(ks) - len(ok),
                    "median": float(np.median(ok)), "max": int(max(ok)),
                    "median_last_half": float(np.median(ok[len(ok) // 2:]))}
    he = np.array([r["heuristic_err"] for r in rows])
    out["heuristic_err"] = {"min": float(he.min()), "median": float(np.median(he)),
                            "max": float(he.max())}
    return out


def main(out_path):
    results = {}
    for gname, adj in build_graphs().items():
        rows = run_graph(adj)
        results[gname] = {"summary": summarize(rows), "rows": rows}
        print(gname, json.dumps(results[gname]["summary"]), flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "warm_start.json")
