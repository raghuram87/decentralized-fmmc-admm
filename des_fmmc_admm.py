"""
Algorithm 1 (DES-FMMC, technical report Section 9): the general decentralized
ADMM loop with a SWAPPABLE spectral-oracle backend (full EVD / Lanczos /
Chebyshev), used for the backend-comparison and k-ablation experiments.

The dual U is kept in low-rank form across outer iterations
(Q_ext, delta_evals), as in a matrix-free implementation:

    V^t @ x = B^t @ x - mean(x) - Q_ext @ (delta_evals * (Q_ext.T @ x))

This module reuses EdgeGraphProjection from fmmc_exact_admm.py (same
directory) for the exact graph-feasibility B-update -- the oracle choice
only affects the (Z,s) spectral update, not graph feasibility, which the
technical report's Section 9 keeps separate (Algorithm 1, Steps 1 vs 3-4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import time
import numpy as np
import scipy.linalg as la
import networkx as nx

from fmmc_exact_admm import EdgeGraphProjection, B_from_w, compute_slem, validate_graph, metropolis_hastings_weights
from spectral_oracle import oracle_full_evd, oracle_lanczos, oracle_chebyshev, OracleResult
from communication_model import communication_cost_chebyshev, communication_cost_lanczos, CommunicationCost


@dataclass
class DESIterDiagnostics:
    iteration: int
    s: float
    slem: float
    primal_residual: float
    dual_residual: float
    eps_pri: float
    eps_dual: float
    k_active: int
    q_t: int
    backend: str
    certified: bool
    comm: Optional[CommunicationCost] = None
    total_time: float = 0.0


@dataclass
class DESResult:
    B: np.ndarray
    s: float
    history: list = field(default_factory=list)
    elapsed: float = 0.0
    B_history: Optional[list] = None
    best_B: Optional[np.ndarray] = None
    best_slem: float = float("inf")
    best_iteration: int = -1
    rho_history: Optional[list] = None


def _make_V_matvec(B_sparse_dense: np.ndarray, Q_ext: np.ndarray, delta_evals: np.ndarray):
    """Matrix-free V = B - J + U operator, U in low-rank form (Q_ext, delta_evals)."""
    Q_delta = Q_ext * delta_evals if Q_ext.size else Q_ext

    def matvec(x):
        x = np.asarray(x, dtype=float)
        is_1d = x.ndim == 1
        if is_1d:
            x = x[:, None]
        Bx = B_sparse_dense @ x
        Jx = np.mean(x, axis=0, keepdims=True)
        Ux = Q_delta @ (Q_ext.T @ x) if Q_ext.size else 0.0
        y = Bx - Jx - Ux
        return y[:, 0] if is_1d else y
    return matvec


def des_fmmc_admm(adj_matrix: np.ndarray, backend: str = "chebyshev", rho: float = 0.1,
                   num_iter: int = 500, k0: int = 4, delta_t: float = 1e-6,
                   eps_abs: float = 1e-6, eps_rel: float = 1e-4,
                   lambda2_gossip: float = 0.9, eps_gossip: float = 1e-4,
                   lanczos_reorthogonalize: bool = True,
                   init: str = "metropolis", verbose: bool = True,
                   slem_every: int = 1, track_B_history: bool = False,
                   adaptive_rho: bool = False, rho_mu: float = 10.0,
                   rho_tau: float = 2.0, rho_max_updates: int = 50,
                   fresh_oracle_rng: bool = False, oracle_rng_seed: int = 0,
                   tighten_lambda2: bool = False, graph_reg: float = 0.0,
                   require_spectral_stability: bool = False,
                   spectral_stability_window: int = 10,
                   spectral_stability_tol: float = 1e-5) -> DESResult:
    """
    graph_reg (one of two remedies for the Section 17 cycle_n50
    pathology): adds `graph_reg * ||w||_2^2` to
    the graph-projection subproblem, i.e. solves the tiny-regularized FMMC
    `min s + eps*||w||_2^2` with `eps = graph_reg * rho / 2` (see
    EdgeGraphProjection's docstring for the derivation). This breaks ties
    between equal-SLEM configurations that differ in edge-weight norm (e.g.
    cycle_n50's self-loop-free bipartite-parity point vs. its
    self-loop-having neighbor) by preferring the smaller-norm one. Default
    0.0 reproduces every previously-validated number in this paper unchanged.

    require_spectral_stability (the other remedy): see fmmc_exact_admm.py::fmmc_admm_exact's docstring for the
    full rationale -- the standard residual criterion never looks at SLEM
    directly, and Section 17 documents a trajectory where SLEM drifts
    smoothly from 0.9922 to 1.0 while both residuals are already tiny and
    shrinking. When True, convergence additionally requires SLEM to have
    moved by less than `spectral_stability_tol` over the last
    `spectral_stability_window` iterations. Forces per-iteration SLEM
    computation (ignores `slem_every` while active) since the window check
    needs every value, not a throttled subsample. Default off, changes
    nothing when unused.

    fresh_oracle_rng: THE root-cause fix for the Section 17 bipartite
    stuck-fixed-point pathology (cycle_n50: SLEM=1 "converged" for 130+
    iterations despite a reachable 0.992 solution). Root cause, diagnosed
    directly: this loop's call to oracle_chebyshev never passed an `rng`,
    so the oracle's internal default (`np.random.default_rng(0)`) reseeds
    to the IDENTICAL fixed seed on every single call. Once the ADMM state
    (B, s) stabilizes -- which happens exactly when the spectral-correction
    force vanishes near a coverage-error fixed point (Section 8's
    delta_t^coverage) -- the oracle draws the bit-identical random
    Chebyshev block every iteration forever, with ZERO possibility of ever
    re-sampling a block with better overlap on the missed eigendirection
    (verified directly: on cycle_n50's stuck B, the true top eigenvalue is
    exactly -1, a 1-dimensional "checkerboard" mode, while the oracle's
    default-seeded block only ever resolves the graph's 4-dimensional
    near-degenerate +-0.9921 eigenspace -- the block's limited capacity is
    consumed by the larger cluster, and re-trying the SAME seed can never
    discover the smaller one). This is not "unlucky randomness that should
    eventually self-correct" -- there was no fresh randomness being
    introduced at all. `fresh_oracle_rng=True` maintains ONE persistent
    `np.random.Generator` across the whole outer loop (seeded once, for
    reproducibility, by `oracle_rng_seed`) instead of a fresh
    `default_rng(0)` every call, restoring the random block's actual
    purpose: a chance to escape a coverage-error fixed point by
    re-sampling. Default False preserves EXACT existing behavior (and
    every previously-validated number in this paper) unchanged.

    adaptive_rho: standard residual-balancing penalty-parameter update
    (Boyd, Parikh, Chu, Peleato & Eckstein 2011, "Distributed Optimization
    and Statistical Learning via ADMM", Sec. 3.4.1) -- if primal_residual >
    rho_mu * dual_residual, increase rho by rho_tau (tightens graph-
    feasibility enforcement); if dual_residual > rho_mu * primal_residual,
    decrease rho by rho_tau (loosens it, tightens spectral consistency
    instead). Found empirically (Section 15.3.1) that a FIXED rho tuned for
    n=50 converges far more slowly at larger n -- rho=0.1 needed ~600+
    iterations at n=200 (vs. 298 at n=50) because larger n shifts the
    primal/dual residual balance rho=0.1 was tuned for. Default False
    preserves EXACT existing behavior (fixed rho) everywhere else in this
    paper; Theorem 1 (Section 10) is stated for fixed rho -- convergence
    theory for varying-penalty ADMM is classically understood only once the
    rho sequence stabilizes (Boyd et al. 2011 note this explicitly), so
    `rho_max_updates` caps how many times rho can change, after which it is
    held fixed for the remainder of the run -- turning this into a
    fixed-rho method (Theorem 1 applies cleanly) after a bounded warm-up
    adjustment phase, not an unboundedly-varying one.

    slem_every: compute_slem(B) does a FULL dense eigh (O(n^3)) purely for
    the history's diagnostic SLEM field -- at n=200 this was measured
    (cProfile) to be 35% of total wall-clock, dwarfing the matrix-free
    oracle it's supposed to be validating. Default 1 preserves EXACT prior
    behavior (every iteration, matching all previously-validated results
    in the technical report); pass e.g. 10 for large-n scale demonstrations
    where per-iteration SLEM tracking isn't needed. The final iteration
    always computes it regardless, so the returned result's last SLEM is
    always exact, never skipped.
    """
    assert backend in ("full_evd", "lanczos", "chebyshev")
    validate_graph(adj_matrix)
    adj = np.asarray(adj_matrix, dtype=float)
    n = adj.shape[0]
    num_edges = int(adj.sum() / 2)
    J = np.ones((n, n)) / n

    projector = EdgeGraphProjection(adj, graph_reg=graph_reg)
    if init.lower() == "metropolis":
        # Compute the full MH matrix ONCE and index into it -- previously this
        # was called inside the list comprehension below, i.e. once PER EDGE
        # (O(m) redundant O(n^2) recomputations, m=1960 calls measured at
        # n=200 via cProfile), a real quadratic-in-nothing-useful cost.
        w_mh = metropolis_hastings_weights(adj)
        w = np.array([w_mh[i, j] for (i, j) in projector.edges])
    else:
        w = np.zeros(projector.m)
    B = B_from_w(projector.edges, w, n)

    Q_ext = np.zeros((n, 0))
    delta_evals = np.zeros(0)
    U_dense = np.zeros((n, n))  # only for exact diagnostics (residuals); not used by matvec path
    Z = B - J

    history = []
    B_history = [] if track_B_history else None
    start = time.perf_counter()
    s = 0.0
    best_slem = float("inf")
    best_B = None
    best_iteration = -1
    rho_updates_used = 0
    rho_history = [rho] if adaptive_rho else None
    cheb_rng = np.random.default_rng(oracle_rng_seed) if fresh_oracle_rng else None

    for it in range(num_iter):
        M = J + Z - U_dense
        w, B, feas, success = projector.project(M, w0=w)
        if not success:
            raise RuntimeError(f"B-projection failed at iter {it}.")
        if track_B_history:
            B_history.append(B.copy())

        V_dense = B - J + U_dense
        V_dense = 0.5 * (V_dense + V_dense.T)
        matvec = _make_V_matvec(B, Q_ext, delta_evals)

        if backend == "full_evd":
            res: OracleResult = oracle_full_evd(V_dense, rho, delta_t=delta_t)
        elif backend == "lanczos":
            res = oracle_lanczos(matvec, n, rho, k0=k0, delta_t=delta_t)
        else:
            row_abs_sum = np.abs(V_dense).sum(axis=1)
            s_seed = s if s > 0 else 0.1
            res = oracle_chebyshev(matvec, n, rho, row_abs_sum, s_estimate=s_seed,
                                    delta_t=delta_t, block_size=k0, rng=cheb_rng,
                                    tighten_lambda2=tighten_lambda2)

        s = res.s_hat
        delta_arr = np.clip(res.Lambda_active, -s, s) - res.Lambda_active
        Q_ext, delta_evals = res.Q_active, delta_arr
        Z_next = V_dense + Q_ext @ np.diag(delta_evals) @ Q_ext.T if Q_ext.shape[1] else V_dense.copy()
        Z_next = 0.5 * (Z_next + Z_next.T)

        primal_matrix = B - J - Z_next
        U_dense = U_dense + primal_matrix
        primal_residual = float(la.norm(primal_matrix, "fro"))
        dual_residual = float(rho * la.norm(Z_next - Z, "fro"))
        eps_pri = float(n) * eps_abs + eps_rel * max(la.norm(B, "fro"), la.norm(Z_next + J, "fro"))
        eps_dual = float(n) * eps_abs + eps_rel * la.norm(rho * U_dense, "fro")
        will_stop = (it == num_iter - 1) or (primal_residual <= eps_pri and dual_residual <= eps_dual)

        if adaptive_rho and not will_stop and rho_updates_used < rho_max_updates:
            # Standard residual-balancing penalty update (Boyd et al. 2011,
            # Sec. 3.4.1). U_dense is the SCALED dual (its update rule, above,
            # has no rho factor) -- changing rho without rescaling it would
            # silently change what U_dense means between iterations, which
            # is the classic mistake this update must avoid: u_new =
            # u_old * (rho_old/rho_new).
            if primal_residual > rho_mu * dual_residual:
                U_dense /= rho_tau
                rho *= rho_tau
                rho_updates_used += 1
            elif dual_residual > rho_mu * primal_residual:
                U_dense *= rho_tau
                rho /= rho_tau
                rho_updates_used += 1
            if rho_history is not None:
                rho_history.append(rho)
        slem = compute_slem(B) if (require_spectral_stability or slem_every <= 1 or
                                    it % slem_every == 0 or will_stop) else float("nan")
        # Best-iterate tracking (additive, does not change B/s/behavior returned
        # as the primary result): the residual-based stopping criterion measures
        # aggregate Frobenius consistency between B and Z, which is NOT a
        # faithful proxy for SLEM near a near-degenerate/bipartite-parity
        # eigenvalue -- found on cycle_n50 (Section 17), where the run
        # oscillates between a self-loop-having SLEM~0.992 solution and a
        # zero-self-loop SLEM=1.0 one, and can satisfy the residual tolerance
        # right as it sits at the WORSE of the two. Tracking the best SLEM
        # actually observed makes the failure recoverable without touching the
        # ADMM dynamics or the stopping criterion itself.
        if not np.isnan(slem) and slem < best_slem:
            best_slem = slem
            best_B = B.copy()
            best_iteration = it

        if backend == "chebyshev":
            comm = communication_cost_chebyshev(res.q_t, res.k_active, num_edges, lambda2_gossip, eps_gossip)
        elif backend == "lanczos":
            comm = communication_cost_lanczos(res.q_t, res.k_active, num_edges, lambda2_gossip, eps_gossip,
                                               reorthogonalize=lanczos_reorthogonalize)
        else:
            comm = None

        elapsed = time.perf_counter() - start
        history.append(DESIterDiagnostics(
            iteration=it, s=float(s), slem=slem, primal_residual=primal_residual,
            dual_residual=dual_residual, eps_pri=eps_pri, eps_dual=eps_dual,
            k_active=res.k_active, q_t=res.q_t, backend=backend, certified=res.certified,
            comm=comm, total_time=elapsed,
        ))

        Z = Z_next
        if verbose and (it == 0 or (it + 1) % 25 == 0):
            extra = f" | rounds={comm.total_rounds:.1f}" if comm else ""
            print(f"[{backend}] iter={it+1:4d} | s={s:.6f} | SLEM={slem:.6f} | "
                  f"r_p={primal_residual:.3e} (eps={eps_pri:.3e}) | "
                  f"r_d={dual_residual:.3e} (eps={eps_dual:.3e}) | "
                  f"k={res.k_active} | q_t={res.q_t} | certified={res.certified}{extra}")

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
                    print(f"[{backend}] Converged at iteration {it + 1}.")
                break
            elif verbose and (it == 0 or (it + 1) % 25 == 0):
                print(f"[{backend}]   (residuals converged but SLEM not yet stable over last "
                      f"{spectral_stability_window} iters -- continuing)")

    return DESResult(B=B, s=float(s), history=history, elapsed=time.perf_counter() - start,
                      B_history=B_history, best_B=best_B, best_slem=best_slem,
                      best_iteration=best_iteration, rho_history=rho_history)


if __name__ == "__main__":
    np.random.seed(42)
    n = 50
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=42)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
    adj = nx.to_numpy_array(G, dtype=float)

    for backend in ["full_evd", "lanczos", "chebyshev"]:
        print(f"\n=== backend: {backend} ===")
        result = des_fmmc_admm(adj, backend=backend, rho=0.1, num_iter=400, verbose=True)
        last = result.history[-1]
        print(f"Final: s={result.s:.6f} SLEM={last.slem:.6f} iters={last.iteration+1} "
              f"elapsed={result.elapsed:.2f}s")
