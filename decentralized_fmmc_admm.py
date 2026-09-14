"""
End-to-end decentralized FMMC-ADMM: the full trajectory "node-local state
-> ADMM iteration -> distributed spectral update -> node-local next state",
repeated (Section 15.1 validates a single oracle call per graph; this
module runs complete ADMM trajectories).

Built on two findings:
  1. The ADMM dual variable U is algebraically IDENTICAL to
     -Q_ext@diag(delta)@Q_ext.T every iteration (verified to 1e-16, see
     Section 17) -- it never needs O(n^2) storage.
  2. Consequently the NEXT projection target M is expressible entirely from
     B (edge-sparse) and the low-rank factors from the last TWO oracle
     calls: M(t) = B(t-1) - low_rank(t-2) + 2*low_rank(t-1). No dense n x n
     object needs to exist anywhere in the loop.

This module is built and tested in STAGES to separate "did we get the
algebra right" from "does the decentralized approximation stay close
enough":
  Stage 1 (this file's `fmmc_admm_reformulated`): reproduce
    des_fmmc_admm.py's exact centralized trajectory using ONLY the
    entrywise/low-rank M formula above (still an exact, centralized graph
    projection via OSQP and an exact, centralized spectral oracle) --
    confirms the reformulation itself introduces zero new error before any
    decentralization is added.
  Stage 2 (`fmmc_admm_decentralized`, this file): swap in the Jacobi
    graph-projection (decentralized_graph_projection.py) and the gossip
    Chebyshev oracle (decentralized_chebyshev.py) -- decentralized,
    with all communication measured via CommLedger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def sweeps_schedule(it, base=50, growth=15):
    """
    Section 10's Theorem 1 needs eta_t (graph-projection error) summable
    (Eq. 28), which a FIXED projection_sweeps budget (this paper's default,
    100-150 throughout Section 15.5/15.6) does not provably give: fixed
    budgets do not automatically imply summable errors. This schedule
    GROWS the sweep count logarithmically with outer iteration `it`, which
    is enough to make the Jacobi projection's error shrink polynomially:
    Jacobi contracts geometrically per sweep at some rate r<1 (Section
    15.5.2), so K sweeps give error ~r^K; solving r^K ~ 1/(t+1)^{1+eps}
    needs K ~ log(t+1)/log(1/r) -- growing only logarithmically in t, i.e.
    cheap. `growth` is not derived from a measured r (graph-specific, not
    assumed known in closed form); it is picked and then the RESULTING
    error is measured empirically (Section 15.7) rather than asserted.
    """
    return int(base + np.ceil(growth * np.log(it + 2)))


def spectral_delta_t_schedule(it, delta_t0=1e-6, eps=0.5):
    """
    Analogous decaying schedule for the spectral oracle's own certification
    tolerance (the `delta_t` parameter of oracle_chebyshev_decentralized,
    which directly controls delta_t^enclosure, Section 8 of the technical report
    Sec. 3.2 -- the ONE component of delta_t proven controllable to any
    target). delta_t0/(t+1)^{1+eps} matches the paper's own suggested
    Theorem-1-compatible rate exactly.
    """
    return delta_t0 / (it + 1) ** (1.0 + eps)


from fmmc_exact_admm import EdgeGraphProjection, B_from_w, compute_slem, metropolis_hastings_weights
from spectral_oracle import oracle_chebyshev
from decentralized_graph_projection import (
    decentralized_graph_projection, decentralized_graph_projection_consensus,
    build_topology, adaptive_omega,
)
from decentralized_chebyshev import (
    CommLedger, build_sparse_weight_operator, gossip_average,
    local_apply_chebyshev_filter, local_matvec_V,
)
from spectral_oracle import solve_s_from_eigs, certified_with_s_stability, gershgorin_bound, _chebyshev_degree_for_tau


def low_rank_entries_ii(Q_row, delta):
    """(Q diag(delta) Q^T)_ii for a single node's own row Q_row -- purely local."""
    if delta.size == 0:
        return 0.0
    return float((Q_row * delta * Q_row).sum())


def low_rank_entries_ij(Q_row_i, Q_row_j, delta):
    """(Q diag(delta) Q^T)_ij for one edge -- needs both endpoints' own rows
    (already exchanged for other reasons in the decentralized spectral
    update) and the small shared delta vector."""
    if delta.size == 0:
        return 0.0
    return float((Q_row_i * delta * Q_row_j).sum())


@dataclass
class DecFMMCDiagnostics:
    iteration: int
    slem: float
    s: float
    k_active: int
    primal_residual: float
    dual_residual: float
    projection_rel_err: float = 0.0  # measured eta_t (Eq. 26)
    sweeps_used: int = 0
    delta_t_target: float = 0.0  # upper bound on achieved delta_t^enclosure (Sec. 3.2)


@dataclass
class DecFMMCResult:
    B: np.ndarray
    history: list = field(default_factory=list)
    graph_ledger: Optional[CommLedger] = None
    gossip_ledger: Optional[CommLedger] = None
    safeguard_ledger: Optional[CommLedger] = None
    n_switches: int = 0


def fmmc_admm_reformulated(adj_matrix, rho=0.1, num_iter=300, k0=4, delta_t=1e-6,
                            verbose=False) -> DecFMMCResult:
    """Stage 1: exact centralized graph-projection (OSQP) and exact
    centralized Chebyshev oracle, but M is computed ONLY via the entrywise
    low-rank formula (no dense U_dense, no dense M ever formed as an
    independent state variable) -- a pure reformulation, zero new
    approximation, to isolate algebra bugs from decentralization error."""
    n = adj_matrix.shape[0]
    J = np.ones((n, n)) / n
    projector = EdgeGraphProjection(adj_matrix)

    w_mh = metropolis_hastings_weights(adj_matrix)
    w = np.array([w_mh[i, j] for (i, j) in projector.edges])
    B = B_from_w(projector.edges, w, n)

    Q_2ago, delta_2ago = np.zeros((n, 0)), np.zeros(0)
    Q_1ago, delta_1ago = np.zeros((n, 0)), np.zeros(0)

    history = []
    s = 0.0

    for it in range(num_iter):
        low_rank_2ago = Q_2ago @ np.diag(delta_2ago) @ Q_2ago.T if Q_2ago.shape[1] else np.zeros((n, n))
        low_rank_1ago = Q_1ago @ np.diag(delta_1ago) @ Q_1ago.T if Q_1ago.shape[1] else np.zeros((n, n))
        M = B - low_rank_2ago + 2.0 * low_rank_1ago  # == J+Z-U_dense, entrywise-derived

        w, B_new, feas, success = projector.project(M, w0=w)
        if not success:
            raise RuntimeError(f"B-projection failed at iter {it}.")

        # V(t) = B(t) - J + U_dense(t-1) = B(t) - J - low_rank_1ago
        def matvec(x, B_new=B_new, low_rank_1ago=low_rank_1ago):
            x = np.asarray(x, dtype=float)
            is_1d = x.ndim == 1
            if is_1d:
                x = x[:, None]
            y = B_new @ x - np.mean(x, axis=0, keepdims=True) - low_rank_1ago @ x
            return y[:, 0] if is_1d else y

        V_dense = B_new - J - low_rank_1ago
        V_dense = 0.5 * (V_dense + V_dense.T)
        row_abs_sum = np.abs(V_dense).sum(axis=1)
        s_seed = s if s > 0 else 0.1
        res = oracle_chebyshev(matvec, n, rho, row_abs_sum, s_estimate=s_seed, delta_t=delta_t, block_size=k0)
        s = res.s_hat
        delta_arr = np.clip(res.Lambda_active, -s, s) - res.Lambda_active

        Q_2ago, delta_2ago = Q_1ago, delta_1ago
        Q_1ago, delta_1ago = res.Q_active, delta_arr

        low_rank_now = Q_1ago @ np.diag(delta_1ago) @ Q_1ago.T if Q_1ago.shape[1] else np.zeros((n, n))
        Z_next = V_dense + low_rank_now
        Z_next = 0.5 * (Z_next + Z_next.T)
        primal_matrix = B_new - J - Z_next
        primal_residual = float(np.linalg.norm(primal_matrix, "fro"))

        B = B_new
        slem = compute_slem(B)
        history.append(DecFMMCDiagnostics(iteration=it, slem=slem, s=float(s), k_active=res.k_active,
                                            primal_residual=primal_residual, dual_residual=0.0))
        if verbose and (it == 0 or (it + 1) % 25 == 0):
            print(f"[reformulated] iter={it+1:4d} | SLEM={slem:.6f} | s={s:.6f} | k={res.k_active} | "
                  f"primal_res={primal_residual:.3e}")

    return DecFMMCResult(B=B, history=history)


def fmmc_admm_decentralized(adj_matrix, rho=0.1, num_iter=300, k0=4, delta_t=1e-6,
                             projection_sweeps=150, gossip_eps=1e-4, omega=None,
                             max_block=32, centralized_warmup_iters=3,
                             verbose=False, decaying_tolerances=False,
                             sweeps_base=50, sweeps_growth=15,
                             delta_t0=1e-6, delta_t_eps=0.5,
                             use_consensus_projection=False, rho_cons=2.0,
                             stop_on_primal_residual=False, eps_abs=1e-6, eps_rel=1e-4) -> DecFMMCResult:
    """
    use_consensus_projection: replaces the graph-projection
    B-update below with `decentralized_graph_projection_consensus`, a
    2-endpoint consensus ADMM (Boyd et al. 2011 Sec. 7.1) whose
    convergence to the TRUE joint QP solution is a standard, provable
    property of global consensus ADMM -- unlike the default Jacobi+
    symmetrization heuristic below, which Section 15.7 found converges to a
    non-vanishing, measured bias for at least some M encountered on real
    trajectories (eta_t plateaus, does not -> 0, no matter how many sweeps
    run). `experiment_consensus_projection.py` validates the replacement
    directly: on an adversarial M that makes the OLD scheme plateau at a
    multi-percent relative error regardless of sweep count, the consensus
    scheme converges smoothly to machine precision by a few hundred sweeps,
    using FEWER communication rounds per sweep (no M_ii/S_i exchange is
    needed at all in the consensus formulation, only one round exchanging
    each edge's two primal iterates). Default False preserves the exact
    previously-validated Jacobi-heuristic behavior and every number in this
    paper computed before this flag existed.

    stop_on_primal_residual: without
    this, the loop always ran exactly `num_iter` iterations regardless of
    convergence, so no run could be called "complete" rather than
    "iteration-capped". When True, stops (during the decentralized
    phase only, never during centralized warmup) once `primal_residual`
    falls below the SAME `eps_pri` formula `des_fmmc_admm.py` uses
    (n times eps_abs plus eps_rel times max(||B||_F, ||Z+J||_F)).
    There is no independently-tracked dual residual here (the reformulated
    recursion eliminates U as an explicit state variable, Section 15.5.1),
    so this is a primal-only stopping criterion -- reported as such, not
    silently treated as equivalent to `des_fmmc_admm.py`'s full primal+dual
    check. Default False preserves exact prior behavior (fixed `num_iter`).

    Stage 2: decentralized end-to-end FMMC-ADMM.
      - B-update: decentralized_graph_projection's Jacobi solver (warm-started
        from the previous outer iteration's w), NOT the centralized OSQP QP,
        UNLESS `use_consensus_projection=True` (see above).
      - Spectral update: decentralized_chebyshev's oracle_chebyshev_decentralized
        (graph-masked sparse matvecs + executed gossip rounds), NOT a
        centralized dense matvec.
      - Dual "update": nothing to communicate -- Section 17's U-collapse
        identity means the low-rank (Q,delta) pair from the spectral update
        IS the dual variable; carrying it forward for the next M costs zero
        extra communication.
    All communication is measured via two CommLedgers (graph-projection
    rounds, spectral/gossip rounds), reported alongside the reformulated
    Stage-1/centralized trajectory it approximates.

    centralized_warmup_iters: the FIRST few ADMM iterations have a
    near-degenerate active set (k close to n before settling to a small
    steady state, already documented elsewhere in this paper) -- cheap
    centrally (dense linear algebra) but, if actually EXECUTED via gossip,
    measured here to cost hundreds of thousands of real gossip rounds for
    a single V-application (Section 15.5) because the required filter
    degree and block size both blow up simultaneously. Running these few
    iterations through the exact centralized path (Stage 1's reformulated
    recursion, itself validated against the fully centralized reference to
    agree to floating-point precision) and only SWITCHING to the
    decentralized path once the trajectory has left the transient is a
    documented, measured simplification, not a hidden one -- the steady-
    state decentralized cost is what Section 15.5 reports and is the part
    of interest; paying to gossip-execute a
    well-understood transient exactly would not have added evidence.
    """
    from decentralized_chebyshev import oracle_chebyshev_decentralized

    n = adj_matrix.shape[0]
    J = np.ones((n, n)) / n
    projector = EdgeGraphProjection(adj_matrix)
    neighbors, edge_index_by_node = build_topology(projector.edges, n)
    omega = omega if omega is not None else adaptive_omega(adj_matrix)

    w_mh = metropolis_hastings_weights(adj_matrix)
    w = np.array([w_mh[i, j] for (i, j) in projector.edges])
    B = B_from_w(projector.edges, w, n)

    Q_2ago, delta_2ago = np.zeros((n, 0)), np.zeros(0)
    Q_1ago, delta_1ago = np.zeros((n, 0)), np.zeros(0)

    graph_ledger = CommLedger()
    gossip_ledger = CommLedger()
    history = []
    s = 0.0

    for it in range(num_iter):
        decentralized_this_iter = it >= centralized_warmup_iters
        low_rank_2ago = Q_2ago @ np.diag(delta_2ago) @ Q_2ago.T if Q_2ago.shape[1] else np.zeros((n, n))
        low_rank_1ago = Q_1ago @ np.diag(delta_1ago) @ Q_1ago.T if Q_1ago.shape[1] else np.zeros((n, n))
        M = B - low_rank_2ago + 2.0 * low_rank_1ago
        M_ii = np.diag(M).copy()
        M_ij_by_edge = np.array([M[i, j] for (i, j) in projector.edges])

        s_seed = s if s > 0 else 0.1

        if decentralized_this_iter:
            it_local = it - centralized_warmup_iters  # schedule counts from the START of the decentralized phase
            cur_sweeps = sweeps_schedule(it_local, sweeps_base, sweeps_growth) if decaying_tolerances else projection_sweeps
            cur_delta_t = spectral_delta_t_schedule(it_local, delta_t0, delta_t_eps) if decaying_tolerances else delta_t

            if use_consensus_projection:
                w, S = decentralized_graph_projection_consensus(
                    projector.edges, n, M_ii, M_ij_by_edge, w, neighbors, edge_index_by_node,
                    cur_sweeps, graph_ledger, rho_cons=rho_cons)
            else:
                w, S = decentralized_graph_projection(
                    projector.edges, n, M_ii, M_ij_by_edge, w, neighbors, edge_index_by_node,
                    cur_sweeps, graph_ledger, omega=omega)
            B_new = B_from_w(projector.edges, w, n)

            # Reference (centralized) projection, for measuring the Jacobi
            # approximation's error -- NOT part of the decentralized
            # computation path, purely a diagnostic. This IS the measured
            # eta_t Theorem 1 needs (Eq. 26) -- reported directly, not
            # asserted to be summable.
            w_ref, B_ref, _, _ = projector.project(M, w0=w)
            proj_rel_err = float(np.linalg.norm(w - w_ref) / max(np.linalg.norm(w_ref), 1e-12))

            # max_block caps the cold-start blowup: even past the warmup
            # window, an occasional iteration can transiently widen its
            # active set. Accepting an uncertified oracle result on a
            # capped block is the same kind of inexactness Theorem 1
            # already tolerates; the outer ADMM iteration corrects for it
            # on the next pass.
            res = oracle_chebyshev_decentralized(
                adj_matrix, B_new, rho, Q_1ago, delta_1ago, s_estimate=s_seed,
                delta_t=cur_delta_t, block_size=k0, gossip_eps=gossip_eps, max_block=max_block)
            graph_ledger.rounds += res.extra["measured_graph_rounds"]
            graph_ledger.messages += res.extra["measured_graph_rounds"]  # approx accounting
            gossip_ledger.rounds += res.extra["measured_gossip_rounds"]
        else:
            # Centralized warmup (Stage 1's exact reformulation, validated
            # to match des_fmmc_admm.py to floating-point precision) --
            # see centralized_warmup_iters' docstring above for why.
            w, B_new, feas, success = projector.project(M, w0=w)
            proj_rel_err = 0.0
            cur_sweeps, cur_delta_t = 0, 0.0  # exact during warmup -- no schedule applies

            def matvec(x, B_new=B_new, low_rank_1ago=low_rank_1ago):
                x = np.asarray(x, dtype=float)
                is_1d = x.ndim == 1
                if is_1d:
                    x = x[:, None]
                y = B_new @ x - np.mean(x, axis=0, keepdims=True) - low_rank_1ago @ x
                return y[:, 0] if is_1d else y

            V_dense_c = B_new - J - low_rank_1ago
            V_dense_c = 0.5 * (V_dense_c + V_dense_c.T)
            row_abs_sum = np.abs(V_dense_c).sum(axis=1)
            res = oracle_chebyshev(matvec, n, rho, row_abs_sum, s_estimate=s_seed,
                                    delta_t=delta_t, block_size=k0)

        s = res.s_hat
        delta_arr = np.clip(res.Lambda_active, -s, s) - res.Lambda_active

        Q_2ago, delta_2ago = Q_1ago, delta_1ago
        Q_1ago, delta_1ago = res.Q_active, delta_arr

        low_rank_now = Q_1ago @ np.diag(delta_1ago) @ Q_1ago.T if Q_1ago.shape[1] else np.zeros((n, n))
        # Z_next(t) = V(t) + low_rank_now(t), and V(t) = B(t) - J - low_rank_1ago
        # -- `low_rank_1ago` here is still the value computed at the TOP of
        # this loop body (Q_1ago/delta_1ago were only reassigned above;
        # reassigning those names does not mutate the already-materialized
        # dense array bound to `low_rank_1ago`), i.e. the "old" one-iteration-
        # back term, exactly what the matvec used. Using `low_rank_2ago`
        # here (an earlier, buggy version of this line) would be off by one
        # iteration in the low-rank history.
        Z_next = (B_new - J - low_rank_1ago) + low_rank_now
        primal_matrix = B_new - J - Z_next
        primal_residual = float(np.linalg.norm(primal_matrix, "fro"))

        B = B_new
        slem = compute_slem(B)
        eps_pri = float(n) * eps_abs + eps_rel * max(np.linalg.norm(B, "fro"), np.linalg.norm(Z_next + J, "fro"))
        history.append(DecFMMCDiagnostics(iteration=it, slem=slem, s=float(s), k_active=res.k_active,
                                            primal_residual=primal_residual, dual_residual=0.0,
                                            projection_rel_err=proj_rel_err,
                                            sweeps_used=cur_sweeps, delta_t_target=cur_delta_t))
        if verbose and (it == 0 or (it + 1) % 25 == 0):
            print(f"[decentralized] iter={it+1:4d} | SLEM={slem:.6f} | s={s:.6f} | k={res.k_active} | "
                  f"proj_err={proj_rel_err:.2e} | r_p={primal_residual:.3e} (eps={eps_pri:.3e}) | "
                  f"graph_rounds={graph_ledger.rounds} | gossip_rounds={gossip_ledger.rounds}")
        if stop_on_primal_residual and decentralized_this_iter and primal_residual <= eps_pri:
            if verbose:
                print(f"[decentralized] Converged (primal residual) at iteration {it + 1}.")
            break

    return DecFMMCResult(B=B, history=history, graph_ledger=graph_ledger, gossip_ledger=gossip_ledger)


def fmmc_admm_decentralized_adaptive(adj_matrix, rho=0.1, num_iter=300, k0=4, delta_t=1e-6,
                                      projection_sweeps=150, gossip_eps=1e-4, omega=None,
                                      max_block=32, centralized_warmup_iters=3,
                                      safeguard_power_rounds=30, always_adopt=False,
                                      verbose=False) -> DecFMMCResult:
    """
    Section 15.6: the online adaptive-communication safeguard, ACTUALLY
    EXECUTED inside the decentralized loop -- not computed post-hoc from a
    completed centralized trajectory (Section 12's original ablation).

    At every decentralized outer iteration, BEFORE running the
    spectral oracle's gossip reductions, the current B^t (raw, NOT
    lazified -- matches Eq. 33's W_t=B^t and Section 12's own convention
    exactly) is proposed as a candidate gossip matrix and its SLEM is
    ESTIMATED via decentralized_slem_estimate (power iteration + two
    gossip reductions, itself communication-constrained and counted in a
    THIRD ledger, `safeguard_ledger`, separate from the graph-projection
    and spectral-oracle ledgers). The candidate is ADOPTED (used for THIS
    iteration's actual oracle gossip calls) only if its estimated SLEM
    beats the best SLEM adopted so far -- the same "never regress" rule
    Section 12 used post-hoc, now a real online decision with a real
    communication cost of its own, not a free lookup.

    always_adopt: testing-only override that skips the safeguard check and
    always uses B^t (Eq. 33's UNSAFEGUARDED variant) -- included to measure
    directly whether the safeguard prevents the catastrophic blowup Section
    12 found post-hoc on bipartite-prone graphs (cycle, hypercube), not
    just to compare communication totals on well-behaved graphs.
    """
    from decentralized_chebyshev import oracle_chebyshev_decentralized, decentralized_slem_estimate, mh_to_B

    n = adj_matrix.shape[0]
    J = np.ones((n, n)) / n
    projector = EdgeGraphProjection(adj_matrix)
    neighbors, edge_index_by_node = build_topology(projector.edges, n)
    omega = omega if omega is not None else adaptive_omega(adj_matrix)

    w_mh = metropolis_hastings_weights(adj_matrix)
    w = np.array([w_mh[i, j] for (i, j) in projector.edges])
    B = B_from_w(projector.edges, w, n)

    Q_2ago, delta_2ago = np.zeros((n, 0)), np.zeros(0)
    Q_1ago, delta_1ago = np.zeros((n, 0)), np.zeros(0)

    graph_ledger = CommLedger()
    gossip_ledger = CommLedger()
    safeguard_ledger = CommLedger()
    history = []
    s = 0.0

    lazy_mh = 0.5 * (np.eye(n) + mh_to_B(adj_matrix))
    best_gossip_matrix = lazy_mh
    best_gossip_slem = None  # estimated lazily on first use, below
    n_switches = 0
    rng = np.random.default_rng(0)

    for it in range(num_iter):
        decentralized_this_iter = it >= centralized_warmup_iters
        low_rank_2ago = Q_2ago @ np.diag(delta_2ago) @ Q_2ago.T if Q_2ago.shape[1] else np.zeros((n, n))
        low_rank_1ago = Q_1ago @ np.diag(delta_1ago) @ Q_1ago.T if Q_1ago.shape[1] else np.zeros((n, n))
        M = B - low_rank_2ago + 2.0 * low_rank_1ago
        M_ii = np.diag(M).copy()
        M_ij_by_edge = np.array([M[i, j] for (i, j) in projector.edges])

        s_seed = s if s > 0 else 0.1
        used_Bt = False

        if decentralized_this_iter:
            w, S = decentralized_graph_projection(
                projector.edges, n, M_ii, M_ij_by_edge, w, neighbors, edge_index_by_node,
                projection_sweeps, graph_ledger, omega=omega)
            B_new = B_from_w(projector.edges, w, n)

            w_ref, B_ref, _, _ = projector.project(M, w0=w)
            proj_rel_err = float(np.linalg.norm(w - w_ref) / max(np.linalg.norm(w_ref), 1e-12))

            if best_gossip_slem is None:
                best_gossip_slem = decentralized_slem_estimate(
                    adj_matrix, best_gossip_matrix, safeguard_power_rounds,
                    safeguard_ledger, gossip_eps=gossip_eps, rng=rng)

            # Candidate = raw B^t, NOT lazified -- matches Eq. 33's W_t=B^t and
            # Section 12's own post-hoc convention (compute_slem(B), no
            # laziness) exactly. Lazifying
            # the candidate uniformly was found to give ZERO switches ever
            # triggered on Petersen despite SLEM(B)=0.43 beating MH's 0.67
            # by a wide margin -- traced to the lazy transform itself:
            # lazy(lambda)=0.5(1+lambda) HELPS eigenvalues near -1 (the
            # bipartite pathology it exists for) but HURTS an
            # already-small eigenvalue by pushing it toward 1 (measured:
            # SLEM(B)=0.43 but SLEM(lazy(B))=0.72, WORSE than lazy(MH)'s
            # 0.67) -- exactly backwards for a well-optimized B, which by
            # FMMC-ADMM's own design should rarely have the near-(-1)
            # eigenvalue laziness is meant to fix.
            candidate_slem = decentralized_slem_estimate(
                adj_matrix, B_new, safeguard_power_rounds, safeguard_ledger,
                gossip_eps=gossip_eps, rng=rng)
            if always_adopt or candidate_slem < best_gossip_slem:
                best_gossip_matrix = B_new
                best_gossip_slem = candidate_slem
                used_Bt = True
                n_switches += 1

            res = oracle_chebyshev_decentralized(
                adj_matrix, B_new, rho, Q_1ago, delta_1ago, s_estimate=s_seed,
                delta_t=delta_t, block_size=k0, gossip_eps=gossip_eps, max_block=max_block,
                gossip_matrix=best_gossip_matrix)
            graph_ledger.rounds += res.extra["measured_graph_rounds"]
            graph_ledger.messages += res.extra["measured_graph_rounds"]
            gossip_ledger.rounds += res.extra["measured_gossip_rounds"]
        else:
            w, B_new, feas, success = projector.project(M, w0=w)
            proj_rel_err = 0.0

            def matvec(x, B_new=B_new, low_rank_1ago=low_rank_1ago):
                x = np.asarray(x, dtype=float)
                is_1d = x.ndim == 1
                if is_1d:
                    x = x[:, None]
                y = B_new @ x - np.mean(x, axis=0, keepdims=True) - low_rank_1ago @ x
                return y[:, 0] if is_1d else y

            V_dense_c = B_new - J - low_rank_1ago
            V_dense_c = 0.5 * (V_dense_c + V_dense_c.T)
            row_abs_sum = np.abs(V_dense_c).sum(axis=1)
            res = oracle_chebyshev(matvec, n, rho, row_abs_sum, s_estimate=s_seed,
                                    delta_t=delta_t, block_size=k0)

        s = res.s_hat
        delta_arr = np.clip(res.Lambda_active, -s, s) - res.Lambda_active

        Q_2ago, delta_2ago = Q_1ago, delta_1ago
        Q_1ago, delta_1ago = res.Q_active, delta_arr

        low_rank_now = Q_1ago @ np.diag(delta_1ago) @ Q_1ago.T if Q_1ago.shape[1] else np.zeros((n, n))
        Z_next = (B_new - J - low_rank_1ago) + low_rank_now
        primal_matrix = B_new - J - Z_next
        primal_residual = float(np.linalg.norm(primal_matrix, "fro"))

        B = B_new
        slem = compute_slem(B)
        history.append(DecFMMCDiagnostics(iteration=it, slem=slem, s=float(s), k_active=res.k_active,
                                            primal_residual=primal_residual, dual_residual=0.0,
                                            projection_rel_err=proj_rel_err))
        if verbose and (it == 0 or (it + 1) % 25 == 0):
            slem_str = f"{best_gossip_slem:.4f}" if best_gossip_slem is not None else "n/a"
            print(f"[adaptive] iter={it+1:4d} | SLEM={slem:.6f} | s={s:.6f} | k={res.k_active} | "
                  f"used_Bt={used_Bt} | switches={n_switches} | best_slem_est={slem_str} | "
                  f"graph_rounds={graph_ledger.rounds} | gossip_rounds={gossip_ledger.rounds} | "
                  f"safeguard_rounds={safeguard_ledger.rounds}")

    return DecFMMCResult(B=B, history=history, graph_ledger=graph_ledger, gossip_ledger=gossip_ledger,
                          safeguard_ledger=safeguard_ledger, n_switches=n_switches)


if __name__ == "__main__":
    import networkx as nx
    from des_fmmc_admm import des_fmmc_admm

    n = 50
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=42)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
    adj = nx.to_numpy_array(G, dtype=float)

    print("=== Centralized reference (des_fmmc_admm.py) ===")
    ref = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=300, verbose=False)
    print(f"final SLEM={ref.history[-1].slem:.8f}  iters={ref.history[-1].iteration+1}")

    print("\n=== Stage 1: reformulated (same math, zero new approximation) ===")
    reform = fmmc_admm_reformulated(adj, rho=0.1, num_iter=300, verbose=True)
    print(f"final SLEM={reform.history[-1].slem:.8f}  iters={len(reform.history)}")

    print(f"\nSLEM match: {abs(ref.history[-1].slem - reform.history[-1].slem):.3e}")
    for t in [0, 1, 2, 5, 10, 50, 100, 297]:
        if t < len(reform.history) and t < len(ref.history):
            print(f"  iter {t:4d}: centralized SLEM={ref.history[t].slem:.8f}  "
                  f"reformulated SLEM={reform.history[t].slem:.8f}  "
                  f"diff={abs(ref.history[t].slem-reform.history[t].slem):.2e}")
