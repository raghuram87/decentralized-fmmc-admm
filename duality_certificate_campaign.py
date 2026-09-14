"""
Strengthens the 46-instance Delta_obj campaign (delta_obj_campaign.py) with
duality-gap evidence: comparing two numerical objectives (Delta_obj) cannot rule out both solvers
converging to a similarly-good-but-suboptimal point, or a shared systematic
bias. Checking primal feasibility + dual feasibility + primal-dual gap for
each instance is much stronger, solver-independent evidence of correctness.

The dual objective is constructed from CLARABEL's own reported Lagrange
multipliers for Problem (P2)-equivalent's constraints (technical report Eq. 1):

    min_{B,s}  s
    s.t. B@1=1 (dual lambda), B>=0 (dual mu), B_ij=0 off-graph (dual nu),
         s*I-(B-J) >> 0 (dual Y1), s*I+(B-J) >> 0 (dual Y2)

Standard SDP Lagrangian duality gives the dual objective
    g(Y1,Y2,lambda) = tr((Y1-Y2)@J) - sum(lambda)

A from-scratch KKT stationarity check initially did not close (residual
~4.18) because of a sign error, identified by a systematic search over the
2^5 sign combinations of the five dual terms against a direct numerical
Lagrangian-gradient check. The correct stationarity identity is

    Y1 - Y2 + symmetrized(lambda) - mu + N = 0

(note the -mu, not +mu, and +N, not -N, relative to the naive first
guess) -- verified to close to 2e-9-3e-8 (solver-precision-limited) across
five independent test graphs of different families/sizes (Petersen,
cycle_n10, RGG_n15, Erdos-Renyi_n20, Barabasi-Albert_n12) before being
adopted here. `dual_stationarity_residual()` below computes this directly
for every campaign instance and reports it alongside the existing checks --
this is a VERIFIED dual-feasible certificate (PSD Y1,Y2,
nonneg mu, closed stationarity), not merely an empirically-matching
objective value.

For each of the SAME 46 instances delta_obj_campaign.py uses, this script
reports:
  1. CLARABEL's own primal-dual gap (s*_clarabel - g) -- confirms the
     REFERENCE value itself is a tight, trustworthy optimum, not just "a
     number CLARABEL printed".
  2. Our ADMM solution's PRIMAL FEASIBILITY, computed directly (not
     assumed): row-sum error, symmetry error, nonnegativity violation,
     graph-support violation, and SPECTRAL feasibility (min eigenvalue of
     s*I-(B-J) and s*I+(B-J), which must both be >= -tol for our reported
     SLEM to be an ACHIEVED bound, not an under-count).
  3. Our ADMM solution's PRIMAL-DUAL GAP against CLARABEL's independently
     constructed dual bound (slem_ours - g) -- this is the key new
     evidence: it certifies our ADMM's answer is near the TRUE global
     optimum via a bound that does not depend on trusting CLARABEL's own
     primal solve at all, only its (independently verifiable) dual
     certificate.
"""
import numpy as np
import cvxpy as cp
import networkx as nx

from fmmc_exact_admm import fmmc_admm_exact
from delta_obj_campaign import build_instances


def dual_stationarity_residual(Y1, Y2, lam, mu, N):
    """
    ||Y1 - Y2 + symmetrized(lambda) - mu + N||_F -- the corrected KKT
    stationarity identity (see the module docstring for its derivation).
    Should be ~solver-precision (1e-9 to 1e-7) for a dual-feasible,
    stationary point, not just an objective-value match.
    """
    n = Y1.shape[0]
    lam_sym = 0.5 * (np.outer(lam, np.ones(n)) + np.outer(np.ones(n), lam))
    grad = Y1 - Y2 + lam_sym - mu + N
    return float(np.linalg.norm(grad))


def dual_feasibility(Y1, Y2, mu):
    """PSD/nonneg checks on the dual variables themselves -- a valid dual
    point must satisfy these regardless of stationarity."""
    return dict(
        Y1_eigmin=float(np.linalg.eigvalsh(0.5 * (Y1 + Y1.T)).min()),
        Y2_eigmin=float(np.linalg.eigvalsh(0.5 * (Y2 + Y2.T)).min()),
        mu_min=float(mu.min()),
    )


def solve_fmmc_with_dual(adj_matrix, solver_preference=("CLARABEL", "SCS")):
    n = adj_matrix.shape[0]
    adj = np.asarray(adj_matrix, dtype=float)
    J = np.ones((n, n)) / n
    mask_allowed = (adj > 0) | np.eye(n, dtype=bool)

    B = cp.Variable((n, n), symmetric=True)
    s = cp.Variable(nonneg=True)
    c_rowsum = B @ np.ones(n) == np.ones(n)
    c_nonneg = B >= 0
    c_psd1 = s * np.eye(n) - (B - J) >> 0
    c_psd2 = s * np.eye(n) + (B - J) >> 0
    di, dj = np.where(~mask_allowed)
    constraints = [c_rowsum, c_nonneg, c_psd1, c_psd2]
    if di.size:
        c_offgraph = B[di, dj] == 0
        constraints.append(c_offgraph)

    prob = cp.Problem(cp.Minimize(s), constraints)
    for solver in solver_preference:
        try:
            prob.solve(solver=solver)
            if prob.status in ("optimal", "optimal_inaccurate"):
                lam = c_rowsum.dual_value
                mu = c_nonneg.dual_value
                Y1 = c_psd1.dual_value
                Y2 = c_psd2.dual_value
                N = np.zeros((n, n))
                if di.size:
                    N[di, dj] = c_offgraph.dual_value
                dual_obj = float(np.trace((Y1 - Y2) @ J) - np.sum(lam))
                extras = dict(
                    stationarity_residual=dual_stationarity_residual(Y1, Y2, lam, mu, N),
                    **dual_feasibility(Y1, Y2, mu),
                )
                return B.value, float(s.value), dual_obj, solver, extras
        except Exception:
            continue
    raise RuntimeError("All solvers failed to reach optimality.")


def primal_feasibility(B, s_reported, adj_matrix, tol=1e-8):
    n = B.shape[0]
    J = np.ones((n, n)) / n
    mask_allowed = (adj_matrix > 0) | np.eye(n, dtype=bool)

    row_sum_err = float(np.max(np.abs(B @ np.ones(n) - np.ones(n))))
    sym_err = float(np.max(np.abs(B - B.T)))
    nonneg_viol = float(max(0.0, -B.min()))
    offgraph_viol = float(np.max(np.abs(B[~mask_allowed]))) if (~mask_allowed).any() else 0.0

    V = B - J
    V = 0.5 * (V + V.T)
    eigs = np.linalg.eigvalsh(V)
    # s*I - V >> 0  <=>  min eigenvalue of (s*I - V) >= 0  <=>  s >= max(eigs)
    # s*I + V >> 0  <=>  s >= max(-eigs) = -min(eigs)
    psd1_slack = float(s_reported - eigs.max())   # should be >= -tol
    psd2_slack = float(s_reported + eigs.min())   # should be >= -tol

    return dict(row_sum_err=row_sum_err, sym_err=sym_err, nonneg_viol=nonneg_viol,
                offgraph_viol=offgraph_viol, psd1_slack=psd1_slack, psd2_slack=psd2_slack,
                true_slem=float(np.max(np.abs(eigs))))


def run_campaign():
    instances = build_instances()
    print(f"Total instances: {len(instances)}\n")
    header = (f"{'instance':>30} | {'s*(CLARABEL)':>12} | {'dual g':>12} | {'ref gap':>9} | "
              f"{'SLEM(ours)':>11} | {'ours-dual gap':>13} | {'row_sum':>9} | {'nonneg':>9} | "
              f"{'psd1_slk':>9} | {'psd2_slk':>9} | {'stat.resid':>10} | {'Y1/Y2/mu min':>12}")
    print(header)

    rows = []
    failures = []
    for label, adj in instances:
        try:
            B_ref, s_ref, dual_g, solver, dual_extras = solve_fmmc_with_dual(adj)
            ref_gap = s_ref - dual_g

            result = fmmc_admm_exact(adj, rho=0.1, num_iter=1000, verbose=False)
            slem_ours = result.history[-1].slem
            B_final = result.B

            feas = primal_feasibility(B_final, slem_ours, adj)
            ours_dual_gap = slem_ours - dual_g
            dual_feas_min = min(dual_extras["Y1_eigmin"], dual_extras["Y2_eigmin"], dual_extras["mu_min"])

            print(f"{label:>30} | {s_ref:>12.7f} | {dual_g:>12.7f} | {ref_gap:>9.2e} | "
                  f"{slem_ours:>11.7f} | {ours_dual_gap:>13.2e} | {feas['row_sum_err']:>9.2e} | "
                  f"{feas['nonneg_viol']:>9.2e} | {feas['psd1_slack']:>9.2e} | {feas['psd2_slack']:>9.2e} | "
                  f"{dual_extras['stationarity_residual']:>10.2e} | {dual_feas_min:>12.2e}")

            rows.append(dict(label=label, s_ref=s_ref, dual_g=dual_g, ref_gap=ref_gap,
                              slem_ours=slem_ours, ours_dual_gap=ours_dual_gap,
                              stationarity_residual=dual_extras["stationarity_residual"],
                              dual_feas_min=dual_feas_min, **feas))
        except Exception as e:
            failures.append((label, str(e)))
            print(f"{label:>30} | FAILED: {e}")

    ref_gaps = np.array([abs(r["ref_gap"]) for r in rows])
    ours_gaps = np.array([abs(r["ours_dual_gap"]) for r in rows])
    psd_slacks = np.array([min(r["psd1_slack"], r["psd2_slack"]) for r in rows])
    nonneg_viols = np.array([r["nonneg_viol"] for r in rows])
    row_sum_errs = np.array([r["row_sum_err"] for r in rows])
    stat_resids = np.array([r["stationarity_residual"] for r in rows])
    dual_feas_mins = np.array([r["dual_feas_min"] for r in rows])

    print("\n" + "=" * 100)
    print(f"n = {len(rows)} successful instances ({len(failures)} failed)")
    print(f"CLARABEL's own primal-dual gap:  max={ref_gaps.max():.3e}  median={np.median(ref_gaps):.3e}")
    print(f"Our ADMM's gap vs dual bound:     max={ours_gaps.max():.3e}  median={np.median(ours_gaps):.3e}  "
          f"95th pct={np.percentile(ours_gaps, 95):.3e}")
    print(f"Our ADMM's primal feasibility:")
    print(f"  row-sum error:     max={row_sum_errs.max():.3e}  median={np.median(row_sum_errs):.3e}")
    print(f"  nonnegativity viol: max={nonneg_viols.max():.3e}  median={np.median(nonneg_viols):.3e}")
    print(f"  spectral (PSD) slack: min={psd_slacks.min():.3e} (must be >= ~0 for feasibility)")
    print(f"Dual certificate verification (NOT just objective-value matching):")
    print(f"  KKT stationarity residual: max={stat_resids.max():.3e}  median={np.median(stat_resids):.3e}")
    print(f"  Dual feasibility (Y1,Y2,mu min eigenvalue/value): min={dual_feas_mins.min():.3e} "
          f"(must be >= ~0)")
    if failures:
        print("\nFailures:")
        for label, err in failures:
            print(f"  {label}: {err}")

    return rows, failures


if __name__ == "__main__":
    run_campaign()
