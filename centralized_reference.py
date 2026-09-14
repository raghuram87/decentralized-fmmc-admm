"""
Independent, trusted reference solver for the exact FMMC SDP (technical
report Eq. 1), via CVXPY + CLARABEL (falling back to SCS). Validates,
independently of this project's own ADMM code, that fmmc_exact_admm.py
actually reaches the TRUE FMMC optimum -- the three
oracle backends only agreed with EACH OTHER, which rules out disagreement
between them but not a shared bug in the ADMM formulation itself.

Solves the SDP directly in dense matrix form (not the edge-weight w
parameterization fmmc_exact_admm.py uses) -- a different code
path, which is the point of having a second, independently-implemented
reference. CLARABEL
and SCS are both open-source, SDP-capable, and already bundled with
cvxpy (confirmed installed), which is the substitution the working
paper's own baseline list (Section 14.2, item 1) needs to make explicit.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp
import networkx as nx


def solve_fmmc_cvxpy(adj_matrix: np.ndarray, solver_preference=("CLARABEL", "SCS"),
                      verbose: bool = False):
    """
    min_{B,s}  s
    s.t.  -sI <= B-J <= sI,  B=B^T,  B@1=1,  B_ij=0 off-graph,  B>=0

    Returns (B, s, status, solver_used).
    """
    n = adj_matrix.shape[0]
    adj = np.asarray(adj_matrix, dtype=float)
    J = np.ones((n, n)) / n
    mask_allowed = (adj > 0) | np.eye(n, dtype=bool)

    B = cp.Variable((n, n), symmetric=True)
    s = cp.Variable(nonneg=True)

    constraints = [
        B @ np.ones(n) == np.ones(n),
        B >= 0,
        s * np.eye(n) - (B - J) >> 0,
        s * np.eye(n) + (B - J) >> 0,
    ]
    disallowed_i, disallowed_j = np.where(~mask_allowed)
    if disallowed_i.size:
        constraints.append(B[disallowed_i, disallowed_j] == 0)

    prob = cp.Problem(cp.Minimize(s), constraints)

    last_status = None
    for solver in solver_preference:
        try:
            prob.solve(solver=solver, verbose=verbose)
            last_status = prob.status
            if prob.status in ("optimal", "optimal_inaccurate"):
                return B.value, float(s.value), prob.status, solver
        except (cp.error.SolverError, Exception) as e:
            last_status = f"{solver} raised: {e}"
            continue
    raise RuntimeError(f"All solvers failed to reach optimality. Last status: {last_status}")


def compute_slem(B: np.ndarray) -> float:
    n = B.shape[0]
    J = np.ones((n, n)) / n
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * ((B - J) + (B - J).T)))))


if __name__ == "__main__":
    from fmmc_exact_admm import fmmc_admm_exact

    np.random.seed(7)
    graphs = {}

    n = 20
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=1)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
    graphs["random_geometric_n20"] = nx.to_numpy_array(G, dtype=float)

    graphs["cycle_n12"] = nx.to_numpy_array(nx.cycle_graph(12), dtype=float)
    graphs["petersen_n10"] = nx.to_numpy_array(nx.petersen_graph(), dtype=float)

    print(f"{'graph':>22} | {'SLEM (CVXPY ref)':>17} | {'SLEM (our ADMM)':>16} | "
          f"{'|delta|':>10} | {'solver':>8} | {'ADMM iters':>10}")

    for gname, adj in graphs.items():
        B_ref, s_ref, status, solver = solve_fmmc_cvxpy(adj)
        slem_ref = compute_slem(B_ref)

        result = fmmc_admm_exact(adj, rho=0.1, num_iter=1500, verbose=False)
        slem_ours = result.history[-1].slem

        delta = abs(slem_ref - slem_ours)
        print(f"{gname:>22} | {slem_ref:>17.8f} | {slem_ours:>16.8f} | "
              f"{delta:>10.2e} | {solver:>8} | {result.history[-1].iteration+1:>10}")
