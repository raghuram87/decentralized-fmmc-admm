"""
Faithful implementation of Rokade & Kalaimani, "Distributed computation of
fast consensus weights using ADMM" (Automatica 142, 2022; arXiv:2002.08106v3
-- fetched and read directly, page/equation numbers below refer to that PDF).

IMPORTANT -- this solves a DIFFERENT problem than the rest of this project:
  - FMMC (this paper's own subject): minimize ||B-J||_2 over SYMMETRIC,
    NONNEGATIVE, doubly-stochastic, graph-supported B.
  - Rokade-Kalaimani (RK): minimize ||W-J||_2 (induced 2-norm/spectral norm,
    NOT Frobenius -- their own notation section) over graph-supported W with
    row+column sums = 1 (Eq. 4/Prop. 1) -- NO symmetry, NO nonnegativity
    required (their Remark 1, page 3: "the weights Wij are allowed to be
    negative in general"). This is the broader FDLA problem (Xiao & Boyd
    2004), not FMMC. Kept as a contrasting baseline, not a competitor on
    the same problem -- see technical report Section 14 and its
    Limitations section for the full discussion.

Algorithm 1 (page 4) architecture, faithfully reproduced:
  - Each of n agents keeps a DENSE n x n local copy W_i (their own estimate
    of the global optimal W*), not just an edge-weight vector -- a real,
    substantial architectural difference from this project's edge-local
    parameterization, and exactly why RK does not scale the way the
    Chebyshev/Lanczos-ADMM methods here do.
  - The topological constraint (W_i)_{i,j}=0 for (i,j) not in E is imposed
    ONLY ON ROW i of W_i (page 3, right column: "impose the topological
    constraint ... only on the i-th row of the matrix Wi ... each agent can
    impose this constraint on its estimate Wi knowing only its neighbour
    set Ni") -- the other n-1 rows of W_i are NOT locally constrained; they
    are pulled toward feasibility only via the ADMM consensus penalty
    against neighbors' copies. This is the key architectural fact that
    makes RK's per-node subproblem an n x n optimization, not an O(deg_i)
    one.
  - Primal update (Eq. 13) is a convex program: minimize a spectral-norm
    term plus dual-variable linear terms plus a quadratic ADMM penalty,
    subject to that one-row sparsity constraint. Solved here via CVXPY
    (correctness over hand-derived speed -- getting a proximal operator
    for the spectral-norm ball wrong would misrepresent their method).
  - Dual update (page 4, right column) uses a(k), b(k) (row/column-sum
    multipliers) and M_i(k) = sum_{j in N_i} C_ij(k) DIRECTLY (Algorithm 1
    itself, not the C_ij/D_ij-tracking Algorithm 2 it's derived from, so
    M_i can be updated in closed form without ever materializing C_ij).
  - Communication per iteration: each agent exchanges its FULL n x n
    matrix W_i(k+1) with every neighbor -- one exact round, O(n^2) floats
    per message (not modeled/estimated: this is the literal per-iteration
    cost their algorithm has, and it is the direct point of contrast with
    this project's O(deg_i)-per-round edge-local scheme).
  - Stopping criterion: the residual R_i(k) (Eq. 25-27), max of primal
    feasibility residuals for agent i.

We did not attempt to reconstruct the paper's own n=6 Fig. 1 example graph
from the (low-resolution) published figure -- guessing its exact edge set
and claiming to "reproduce Table I's 0.4519" would be false precision if
the edges are wrong. Instead: `verify_against_centralized()` below checks
that Algorithm 1 converges to the SAME optimum as the independently-solved
centralized Problem (P2) (page 3, spectral-norm minimization via CVXPY) on
several graphs -- this validates the same claim their Theorem 1 makes
(convergence factor -> optimal value), just on graphs we generated and can
verify end-to-end ourselves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import cvxpy as cp


def _neighbors_no_self(adj_matrix: np.ndarray, i: int) -> list[int]:
    return [j for j in range(adj_matrix.shape[0]) if j != i and adj_matrix[i, j] > 0]


def solve_centralized_p2(adj_matrix: np.ndarray, solver_preference=("CLARABEL", "SCS")) -> tuple[np.ndarray, float, str]:
    """Problem (P2), page 3: minimize ||W - J||_2 (induced/spectral norm)
    subject to W1=1, W^T1=1, and graph-support zero pattern -- the
    CENTRALIZED problem Algorithm 1 is a distributed solver for."""
    n = adj_matrix.shape[0]
    J = np.ones((n, n)) / n
    W = cp.Variable((n, n))
    ones = np.ones(n)
    constraints = [W @ ones == ones, W.T @ ones == ones]
    for i in range(n):
        off = [j for j in range(n) if j != i and adj_matrix[i, j] == 0]
        for j in off:
            constraints.append(W[i, j] == 0)
    prob = cp.Problem(cp.Minimize(cp.norm(W - J, 2)), constraints)
    last_err = None
    for solver_name in solver_preference:
        try:
            prob.solve(solver=getattr(cp, solver_name))
            if W.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                return W.value, float(np.linalg.norm(W.value - J, 2)), solver_name
        except Exception as e:  # solver not installed / failed -- try the next
            last_err = e
            continue
    raise RuntimeError(f"No solver succeeded for centralized P2. Last error: {last_err}")


@dataclass
class RKResult:
    W_admm: np.ndarray            # (n,n) -- row i taken from agent i's final W_i
    convergence_factor: float     # ||W_admm - J||_2
    iterations: int
    converged: bool
    residual_history: list = field(default_factory=list)
    comm_floats_per_iteration: int = 0   # sum_i deg(i) * n^2 (full-matrix exchange, Remark 4/Algorithm 1)
    wall_time_s: float = 0.0


def rk_admm(adj_matrix: np.ndarray, rho: float = 1.0 / 16, eps: float = 1e-3,
            max_iter: int = 300, solver_preference=("CLARABEL", "SCS"),
            verbose: bool = False) -> RKResult:
    n = adj_matrix.shape[0]
    ones = np.ones(n)
    J = np.ones((n, n)) / n
    N = [_neighbors_no_self(adj_matrix, i) for i in range(n)]
    support = [sorted(set(N[i]) | {i}) for i in range(n)]  # (i,i) in E by the paper's own convention

    W = [np.zeros((n, n)) for _ in range(n)]
    a = [np.zeros(n) for _ in range(n)]
    b = [np.zeros(n) for _ in range(n)]
    M = [np.zeros((n, n)) for _ in range(n)]

    def solve_primal(i, W_prev):
        Wi = cp.Variable((n, n))
        off = [j for j in range(n) if j not in support[i]]
        constraints = [Wi[i, j] == 0 for j in off]

        obj = (1.0 / n) * cp.norm(Wi - J, 2)
        obj += a[i] @ (Wi @ ones - ones)
        obj += b[i] @ (Wi.T @ ones - ones)
        obj += cp.trace(Wi.T @ M[i])
        obj += (rho / 2.0) * (cp.sum_squares(Wi @ ones - ones) + cp.sum_squares(Wi.T @ ones - ones))
        for j in N[i]:
            avg = 0.5 * (W_prev[i] + W_prev[j])
            obj += (rho / 2.0) * cp.sum_squares(Wi - avg)

        prob = cp.Problem(cp.Minimize(obj), constraints)
        last_err = None
        for solver_name in solver_preference:
            try:
                prob.solve(solver=getattr(cp, solver_name))
                if Wi.value is not None:
                    return Wi.value
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Primal update failed for agent {i}. Last error: {last_err}")

    comm_floats_per_iteration = int(sum(len(N[i]) for i in range(n)) * n * n)

    t0 = time.time()
    residual_history = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        W_prev = [w.copy() for w in W]
        W_new = [solve_primal(i, W_prev) for i in range(n)]
        W = W_new

        for i in range(n):
            a[i] = a[i] + rho * (W[i] @ ones - ones)
            b[i] = b[i] + rho * (W[i].T @ ones - ones)
            M[i] = M[i] + (rho / 2.0) * sum(W[i] - W[j] for j in N[i])

        # Stopping residual (Eq. 25-27): R_i(k) = max over the 4 residual types.
        R = np.zeros(n)
        for i in range(n):
            r1 = np.linalg.norm(W[i] @ ones - ones) / np.sqrt(n)
            r2 = np.linalg.norm(W[i].T @ ones - ones) / np.sqrt(n)
            r3 = max((np.linalg.norm(W[i] - W[j], 'fro') / n for j in N[i]), default=0.0)
            off = [j for j in range(n) if j not in support[i]]
            r4 = max((abs(W[i][i, j]) for j in off), default=0.0)
            R[i] = max(r1, r2, r3, r4)
        residual_history.append(float(R.max()))
        if verbose:
            print(f"  iter {it:4d}  max R_i = {R.max():.6f}")
        if R.max() <= eps:
            converged = True
            break

    W_admm = np.vstack([W[i][i, :] for i in range(n)])
    conv_factor = float(np.linalg.norm(W_admm - J, 2))
    return RKResult(W_admm=W_admm, convergence_factor=conv_factor, iterations=it,
                     converged=converged, residual_history=residual_history,
                     comm_floats_per_iteration=comm_floats_per_iteration,
                     wall_time_s=time.time() - t0)


def metropolis_convergence_factor(adj_matrix: np.ndarray) -> float:
    """Eq. 8, page 3: (WM)_ij = min{1/(1+|Ni|), 1/(1+|Nj|)} for (i,j) in E,
    i!=j (note the "+1" -- this is THEIR Metropolis convention, distinct
    from fmmc_exact_admm.py's 1/max(deg_i,deg_j) used for the FMMC
    problem elsewhere in this project); diagonal = 1 - sum_{j in Ni} (WM)_ij."""
    n = adj_matrix.shape[0]
    deg = adj_matrix.sum(axis=1)
    denom = np.maximum(1.0 + deg[:, None], 1.0 + deg[None, :])
    WM = np.where(adj_matrix > 0, 1.0 / denom, 0.0)
    np.fill_diagonal(WM, 1.0 - WM.sum(axis=1))
    J = np.ones((n, n)) / n
    return float(np.linalg.norm(WM - J, 2))


def verify_against_centralized(graphs: dict, rho=1.0 / 16, eps=1e-3, max_iter=300):
    print(f"{'graph':>18} | {'centralized':>11} | {'RK-ADMM':>11} | {'Metropolis':>11} | "
          f"{'iters':>6} | {'converged':>9} | {'time(s)':>8}")
    for name, adj in graphs.items():
        W_star, cf_star, solver = solve_centralized_p2(adj)
        result = rk_admm(adj, rho=rho, eps=eps, max_iter=max_iter)
        cf_mh = metropolis_convergence_factor(adj)
        print(f"{name:>18} | {cf_star:>11.4f} | {result.convergence_factor:>11.4f} | "
              f"{cf_mh:>11.4f} | {result.iterations:>6d} | {str(result.converged):>9} | "
              f"{result.wall_time_s:>8.1f}")


if __name__ == "__main__":
    import networkx as nx

    graphs = {
        "petersen_n10": nx.to_numpy_array(nx.petersen_graph(), dtype=float),
        "cycle_n6": nx.to_numpy_array(nx.cycle_graph(6), dtype=float),
        "small_rgg_n8": nx.to_numpy_array(
            nx.random_geometric_graph(8, radius=0.6, seed=1), dtype=float),
    }
    verify_against_centralized(graphs)
