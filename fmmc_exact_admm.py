"""
Exact centralized reference solver for the edge-weight FMMC formulation used
in the technical report (Sections 3-4).

Unlike formulations that parameterize B directly via unique symmetric
entries under an EQUALITY row-sum constraint, this formulation uses edge weights w_e >= 0 as the ONLY decision variable, with
B(w) = I - L(w) and an INEQUALITY constraint sum_j w_ij <= 1 (the diagonal
B_ii = 1 - sum_j w_ij is a DERIVED quantity, not a free variable). Projecting
a target matrix M onto this feasible set is therefore a coupled QP
(the diagonal-fit term couples every edge touching a node), not the simple
decoupled dual problem of the equality-constrained case -- solved here via
OSQP rather than
a hand-derived semismooth Newton step, since OSQP is a well-tested QP solver
appropriate for a "reference" implementation.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import networkx as nx
import osqp


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------
@dataclass
class ADMMDiagnostics:
    iteration: int
    s: float
    slem: float
    primal_residual: float
    dual_residual: float
    primal_residual_scaled_eps: float
    dual_residual_scaled_eps: float
    row_sum_error: float
    symmetry_error: float
    min_entry: float
    k_active: int
    spectral_gap: float
    total_time: float


@dataclass
class FMMCResult:
    w: np.ndarray
    B: np.ndarray
    Z: np.ndarray
    U: np.ndarray
    s: float
    history: list[ADMMDiagnostics]
    elapsed: float


# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------
def validate_graph(adj_matrix: np.ndarray) -> None:
    adj = np.asarray(adj_matrix, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj_matrix must be square.")
    if not np.allclose(adj, adj.T):
        raise ValueError("adj_matrix must be symmetric.")
    if np.any(np.diag(adj) != 0):
        raise ValueError("adj_matrix must not contain self-loops.")
    if np.any(adj < 0):
        raise ValueError("adj_matrix cannot contain negative entries.")
    G = nx.from_numpy_array((adj > 0).astype(int))
    if not nx.is_connected(G):
        raise ValueError("The graph must be connected.")


def metropolis_hastings_weights(adj_matrix: np.ndarray) -> np.ndarray:
    """w_ij = 1/max(deg_i, deg_j) for every edge -- a feasible warm start
    (automatically satisfies w>=0 and sum_j w_ij <= 1, since deg_i terms of
    1/max(deg_i,deg_j) sum to at most 1)."""
    adj = np.asarray(adj_matrix, dtype=float)
    deg = adj.sum(axis=1)
    max_deg = np.maximum(deg[:, None], deg[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_full = np.where(adj > 0, 1.0 / max_deg, 0.0)
    return w_full


def B_from_w(edges: list[tuple[int, int]], w: np.ndarray, n: int) -> np.ndarray:
    B = np.eye(n)
    row_sum = np.zeros(n)
    for val, (i, j) in zip(w, edges):
        # B = I - L(w), and L_ij = -w_ij off-diagonal, so B_ij = w_ij.
        B[i, j] += val
        B[j, i] += val
        row_sum[i] += val
        row_sum[j] += val
    B[np.arange(n), np.arange(n)] = 1.0 - row_sum
    return B


# ---------------------------------------------------------------------
# Exact B-update: projection onto {w>=0, Aw<=1} minimizing ||B(w)-M||_F^2
# ---------------------------------------------------------------------
class EdgeGraphProjection:
    """
    Solves, for a symmetric target M:

        min_w  ||c - A w||_2^2 + 2||w - d||_2^2      (== ||B(w)-M||_F^2 + const)
        s.t.   w >= 0,  A w <= 1

    where A is the unsigned (n x m) vertex-edge incidence matrix,
    c_i = 1 - M_ii, d_e = M_ij for edge e=(i,j). Derivation: B(w) = I-L(w)
    has B_ii = 1-sum_{j~i} w_ij = 1-(Aw)_i and B_ij=w_ij for edges, so
    ||B(w)-M||_F^2 = sum_i(1-(Aw)_i-M_ii)^2 + 2 sum_e(w_e-M_ij)^2 + const
    (the factor 2 because each undirected edge appears twice in the
    Frobenius norm, at (i,j) and (j,i)).

    Solved with OSQP: P and the constraint matrix are FIXED across ADMM
    iterations (only the linear term q, built from the current M, changes),
    so the QP is set up once and updated via `prob.update(q=...)` per call
    -- no re-factorization needed.
    """

    def __init__(self, adj_matrix: np.ndarray, graph_reg: float = 0.0):
        """
        graph_reg >= 0 (opt-in, default 0, changes NOTHING previously
        validated when left at 0): adds `graph_reg * ||w||_2^2` to the
        graph-projection subproblem's own objective, i.e. solves
            min_w  ||c - A w||_2^2 + 2||w - d||_2^2 + graph_reg * ||w||_2^2
        instead of the pure projection. This is the ADMM-subproblem form of
        regularizing the ORIGINAL FMMC objective by `eps * ||w||_2^2`
        (Section 17's cycle_n50 fixed-point pathology): since this subproblem is the augmented-Lagrangian
        B-update at fixed penalty `rho` -- argmin_w (rho/2)||B(w)-M||_F^2 +
        eps||w||_2^2 -- matching this method's UNSCALED
        ||B(w)-M||_F^2 + graph_reg||w||_2^2 form gives
            graph_reg = 2*eps/rho,  i.e.  eps = graph_reg * rho / 2.
        Reported explicitly wherever `graph_reg` is used so a reader can
        recover the eps this corresponds to for a given rho.
        """
        validate_graph(adj_matrix)
        self.adj = np.asarray(adj_matrix, dtype=float)
        self.n = self.adj.shape[0]
        self.graph_reg = float(graph_reg)

        self.edges: list[tuple[int, int]] = []
        for i in range(self.n):
            for j in np.flatnonzero(self.adj[i] > 0):
                j = int(j)
                if j > i:
                    self.edges.append((i, j))
        self.m = len(self.edges)

        rows, cols, data = [], [], []
        for e, (i, j) in enumerate(self.edges):
            rows.extend([i, j])
            cols.extend([e, e])
            data.extend([1.0, 1.0])
        self.A_inc = sp.csr_matrix((data, (rows, cols)), shape=(self.n, self.m))

        # P = 2*(A_inc^T A_inc + 2I) + 2*graph_reg*I  (OSQP convention: (1/2) w^T P w + q^T w)
        AtA = (self.A_inc.T @ self.A_inc).tocsc()
        self.P = (2.0 * AtA + (4.0 + 2.0 * self.graph_reg) * sp.eye(self.m, format="csc")).tocsc()

        # Constraints: [I; A_inc] w in ([0,inf) x (-inf,1])
        A_stack = sp.vstack([sp.eye(self.m, format="csc"), self.A_inc], format="csc")
        l = np.concatenate([np.zeros(self.m), -np.inf * np.ones(self.n)])
        u = np.concatenate([np.inf * np.ones(self.m), np.ones(self.n)])

        self.prob = osqp.OSQP()
        q0 = np.zeros(self.m)
        # polish=False: OSQP's polish step prints "Polishing not needed..."
        # regardless of verbose=False (a known OSQP quirk); our own
        # eps_abs/eps_rel are already tight enough without it.
        self.prob.setup(P=self.P, q=q0, A=A_stack, l=l, u=u, verbose=False,
                         eps_abs=1e-10, eps_rel=1e-10, max_iter=20000, polish=False)

    def target_vectors(self, M: np.ndarray):
        M_sym = 0.5 * (M + M.T)
        c = 1.0 - np.diag(M_sym)
        d = np.array([M_sym[i, j] for (i, j) in self.edges])
        return c, d

    def project(self, M: np.ndarray, w0: Optional[np.ndarray] = None):
        c, d = self.target_vectors(M)
        q = -2.0 * (self.A_inc.T @ c + 2.0 * d)
        self.prob.update(q=q)
        if w0 is not None:
            self.prob.warm_start(x=w0)
        res = self.prob.solve()
        w = np.asarray(res.x, dtype=float)
        w = np.maximum(w, 0.0)  # guard tiny negative numerical noise
        B = B_from_w(self.edges, w, self.n)
        feas = float(max(0.0, -w.min() if w.size else 0.0,
                          np.max(self.A_inc @ w - 1.0)))
        success = res.info.status in ("solved", "solved inaccurate")
        return w, B, feas, success


# ---------------------------------------------------------------------
# Exact spectral (Z,s) update (closed form)
# ---------------------------------------------------------------------
def solve_optimal_s(eigenvalues: np.ndarray, rho: float, tol: float = 1e-12,
                     maxiter: int = 200) -> float:
    a = np.abs(np.asarray(eigenvalues, dtype=float))

    def f(s: float) -> float:
        return rho * np.maximum(a - s, 0.0).sum() - 1.0

    if f(0.0) <= 0.0:
        return 0.0
    lo, hi = 0.0, float(np.max(a))
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def spectral_box_projection(V: np.ndarray, rho: float, tol: float = 1e-12):
    V = 0.5 * (V + V.T)
    eigenvalues, Q = la.eigh(V)
    s = solve_optimal_s(eigenvalues, rho, tol=tol)
    clipped = np.clip(eigenvalues, -s, s)
    Z = (Q * clipped) @ Q.T
    Z = 0.5 * (Z + Z.T)
    return Z, s, eigenvalues, clipped


def compute_slem(B: np.ndarray) -> float:
    n = B.shape[0]
    J = np.ones((n, n)) / n
    return float(np.max(np.abs(la.eigvalsh(0.5 * ((B - J) + (B - J).T)))))


# ---------------------------------------------------------------------
# Exact-EVD ADMM (reference / ground truth)
# ---------------------------------------------------------------------
def fmmc_admm_exact(adj_matrix: np.ndarray, rho: float = 1.0, num_iter: int = 500,
                     eps_abs: float = 1e-6, eps_rel: float = 1e-4,
                     init: str = "metropolis", rho_schedule: Optional[str] = None,
                     rho_mu: float = 10.0, rho_factor: float = 2.0,
                     verbose: bool = True, graph_reg: float = 0.0,
                     require_spectral_stability: bool = False,
                     spectral_stability_window: int = 10,
                     spectral_stability_tol: float = 1e-5) -> FMMCResult:
    """
    graph_reg: opt-in tiny edge-weight regularization, see
    EdgeGraphProjection's docstring -- 0.0 (default) reproduces every
    previously-validated number in this paper unchanged.

    require_spectral_stability (Section 17's cycle_n50 investigation): the standard primal/dual residual
    stopping test (Eq. 23-25) measures only Frobenius-norm consistency
    between B and Z -- it never looks at SLEM directly, and Section 17
    documents a real trajectory (cycle_n50) where SLEM drifts smoothly and
    monotonically from 0.9922 to 1.0 while BOTH residuals are already tiny
    and shrinking throughout. When this flag is on, convergence additionally
    requires the observed SLEM to have moved by less than
    `spectral_stability_tol` over the last `spectral_stability_window`
    iterations -- i.e. the solver must reach a fixed point of the SPECTRAL
    OBJECTIVE, not just of the Frobenius-consistency proxy, before
    declaring convergence. Default off, changes nothing when unused.
    """
    validate_graph(adj_matrix)
    adj = np.asarray(adj_matrix, dtype=float)
    n = adj.shape[0]
    J = np.ones((n, n)) / n

    projector = EdgeGraphProjection(adj, graph_reg=graph_reg)

    if init.lower() == "metropolis":
        w = np.array([metropolis_hastings_weights(adj)[i, j] for (i, j) in projector.edges])
    elif init.lower() == "zero":
        w = np.zeros(projector.m)
    else:
        raise ValueError("init must be 'metropolis' or 'zero'.")

    B = B_from_w(projector.edges, w, n)
    Z = B - J
    U = np.zeros((n, n))

    history: list[ADMMDiagnostics] = []
    start = time.perf_counter()
    s = 0.0

    for it in range(num_iter):
        M = J + Z - U
        w, B, feas, success = projector.project(M, w0=w)
        if not success:
            raise RuntimeError(f"B-projection (OSQP) failed to solve, status issue at iter {it}.")

        V = B - J + U
        V = 0.5 * (V + V.T)
        Z_next, s, eigenvalues, clipped = spectral_box_projection(V, rho=rho, tol=1e-12)

        abs_eig = np.abs(eigenvalues)
        order = np.argsort(abs_eig)[::-1]
        abs_sorted = abs_eig[order]
        k_active = int(np.sum(abs_sorted > s))
        spectral_gap = float(abs_sorted[k_active - 1] - abs_sorted[k_active]) if 0 < k_active < n else float("nan")

        primal_matrix = B - J - Z_next
        U_next = U + primal_matrix
        primal_residual = float(la.norm(primal_matrix, ord="fro"))
        dual_residual = float(rho * la.norm(Z_next - Z, ord="fro"))

        eps_pri = float(n) * eps_abs + eps_rel * max(la.norm(B, "fro"), la.norm(Z_next + J, "fro"))
        eps_dual = float(n) * eps_abs + eps_rel * la.norm(rho * U_next, "fro")

        row_sum_error = float(np.max(np.abs(B.sum(axis=1) - 1.0)))
        symmetry_error = float(np.max(np.abs(B - B.T)))
        min_entry = float(np.min(B))
        slem = compute_slem(B)
        elapsed = time.perf_counter() - start

        history.append(ADMMDiagnostics(
            iteration=it, s=float(s), slem=slem,
            primal_residual=primal_residual, dual_residual=dual_residual,
            primal_residual_scaled_eps=eps_pri, dual_residual_scaled_eps=eps_dual,
            row_sum_error=row_sum_error, symmetry_error=symmetry_error, min_entry=min_entry,
            k_active=k_active, spectral_gap=spectral_gap, total_time=elapsed,
        ))

        Z, U = Z_next, U_next

        if rho_schedule == "balance":
            if primal_residual > rho_mu * dual_residual:
                rho *= rho_factor
            elif dual_residual > rho_mu * primal_residual:
                rho /= rho_factor

        if verbose and (it == 0 or (it + 1) % 25 == 0):
            print(f"iter={it+1:4d} | rho={rho:.4g} | s={s:.8f} | SLEM={slem:.8f} | "
                  f"r_p={primal_residual:.3e} (eps={eps_pri:.3e}) | "
                  f"r_d={dual_residual:.3e} (eps={eps_dual:.3e}) | "
                  f"k={k_active} | gap={spectral_gap:.3e} | Bfeas={feas:.2e}")

        residual_converged = primal_residual <= eps_pri and dual_residual <= eps_dual
        if residual_converged:
            spectral_stable = True
            if require_spectral_stability:
                window = history[-spectral_stability_window:]
                slem_window = [d.slem for d in window]
                spectral_stable = (len(window) >= spectral_stability_window and
                                    (max(slem_window) - min(slem_window)) <= spectral_stability_tol)
            if spectral_stable:
                if verbose:
                    print(f"Converged at iteration {it + 1}.")
                break
            elif verbose and (it == 0 or (it + 1) % 25 == 0):
                print(f"  (residuals converged but SLEM not yet stable over last "
                      f"{spectral_stability_window} iters -- continuing)")

    return FMMCResult(w=w, B=B, Z=Z, U=U, s=float(s), history=history,
                       elapsed=time.perf_counter() - start)


def summarize_result(result: FMMCResult) -> None:
    last = result.history[-1]
    converged = last.primal_residual <= last.primal_residual_scaled_eps and \
        last.dual_residual <= last.dual_residual_scaled_eps
    print("\n=== FMMC exact ADMM summary ===")
    print(f"Iterations         : {last.iteration + 1}")
    print(f"Converged (scaled) : {converged}")
    print(f"s*                 : {result.s:.10f}")
    print(f"SLEM(B)            : {last.slem:.10f}")
    print(f"Primal residual    : {last.primal_residual:.3e} (eps={last.primal_residual_scaled_eps:.3e})")
    print(f"Dual residual      : {last.dual_residual:.3e} (eps={last.dual_residual_scaled_eps:.3e})")
    print(f"Row-sum error      : {last.row_sum_error:.3e}")
    print(f"Symmetry error     : {last.symmetry_error:.3e}")
    print(f"Minimum B entry    : {last.min_entry:.3e}")
    print(f"Active k*          : {last.k_active}  (gap={last.spectral_gap:.3e})")
    print(f"Wall-clock         : {result.elapsed:.3f}s")


if __name__ == "__main__":
    np.random.seed(42)
    n = 50
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=42)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
    adj = nx.to_numpy_array(G, dtype=float)

    result = fmmc_admm_exact(adj, rho=0.1, num_iter=1000, verbose=True)
    summarize_result(result)
