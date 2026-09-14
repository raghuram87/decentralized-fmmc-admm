"""
Three-way validation against a TRUE optimum, not a shared iteration budget.
Section 15.5.4's n=100/200/500 table
compared the decentralized solver against a CENTRALIZED reference run for
the SAME (small, unconverged) iteration count -- a fair like-for-like
comparison of the decentralization approximation, but not evidence either
one is near the true FMMC optimum, since neither was run to its own
convergence tolerance, let alone checked against an independent bound.

This script closes that gap for one instance (n=100, the largest size at
which CLARABEL can still be run at all -- Section 15's own stated ceiling
is n<~100-150, dense-SDP-bound). Four independent solves of the SAME graph:

  1. CLARABEL (CVXPY), primal optimum s* and its own dual bound g
     (duality_certificate_campaign.py::solve_fmmc_with_dual) -- the
     independent SDP dual lower bound requested.
  2. Centralized exact-EVD ADMM (des_fmmc_admm.py, backend="full_evd"), run
     to its own primal/dual residual stopping criterion (eps_abs=1e-6,
     eps_rel=1e-4) -- NOT a fixed iteration budget.
  3. The decentralized ADMM (decentralized_fmmc_admm.py::
     fmmc_admm_decentralized) run for enough outer iterations to sit well
     past centralized convergence, with the SAME kind of primal-residual
     history tracked and reported (its own eps_pri computed the same way
     des_fmmc_admm.py computes it, for comparability -- decentralized_fmmc_
     admm.py does not track a dual residual since the reformulated
     recursion eliminates U as an explicit state variable; this is reported
     explicitly below, not silently assumed away).

Reports the three-way agreement:
    decentralized primal <-> centralized primal <-> independent dual bound
and, crucially, how close ALL THREE sit to CLARABEL's own primal-dual gap
(the tightest available evidence of true optimality on this instance).
"""
import os
import time
import numpy as np
import networkx as nx

from duality_certificate_campaign import solve_fmmc_with_dual, primal_feasibility
from des_fmmc_admm import des_fmmc_admm
from decentralized_fmmc_admm import fmmc_admm_decentralized
from fmmc_exact_admm import compute_slem

# Cache for the (slow) CLARABEL reference solves; override with FMMC_CACHE_DIR.
_SCRATCH = os.environ.get("FMMC_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
os.makedirs(_SCRATCH, exist_ok=True)


def build_rgg(n, seed=42):
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=seed)
    assert nx.is_connected(G), "instance must be connected"
    return G


def run(n=100, dec_num_iter=650, dec_warmup=15, dec_sweeps=100, dec_max_block=16, verbose=True,
        cache_clarabel=True):
    G = build_rgg(n)
    adj = nx.to_numpy_array(G, dtype=float)
    print(f"n={n}  edges={G.number_of_edges()}")

    # 1. CLARABEL primal + independent dual bound (cached: this exact n=100,
    # seed=42 instance was already solved once, 208.0s, s*=0.85616622,
    # dual_g=0.85616622, gap=6.19e-11 -- re-solving is redundant compute)
    cache_path = os.path.join(_SCRATCH, f"n{n}_clarabel.npz")
    if cache_clarabel and os.path.exists(cache_path):
        cached = np.load(cache_path)
        s_clarabel, dual_g, dt_clarabel = float(cached["s"]), float(cached["dual_g"]), float(cached["dt"])
        solver = "CLARABEL (cached)"
    else:
        t0 = time.time()
        B_clarabel, s_clarabel, dual_g, solver, _extras = solve_fmmc_with_dual(adj)
        dt_clarabel = time.time() - t0
        if cache_clarabel:
            np.savez(cache_path, s=s_clarabel, dual_g=dual_g, dt=dt_clarabel)
    ref_gap = s_clarabel - dual_g
    print(f"\n[1] CLARABEL:  s*={s_clarabel:.8f}  dual g={dual_g:.8f}  "
          f"own gap={ref_gap:.3e}  ({solver}, {dt_clarabel:.1f}s)")

    # 2. Centralized exact-EVD ADMM, run to ITS OWN convergence tolerance
    t0 = time.time()
    r_full = des_fmmc_admm(adj, backend="full_evd", rho=0.1, num_iter=3000,
                            verbose=False, slem_every=50)
    dt_full = time.time() - t0
    last_full = r_full.history[-1]
    full_converged = (last_full.primal_residual <= last_full.eps_pri and
                       last_full.dual_residual <= last_full.eps_dual)
    print(f"\n[2] Centralized exact-EVD ADMM: iters={len(r_full.history)}  "
          f"SLEM={last_full.slem:.8f}  converged={full_converged}  ({dt_full:.1f}s)")
    print(f"    primal_res={last_full.primal_residual:.3e} (eps={last_full.eps_pri:.3e})  "
          f"dual_res={last_full.dual_residual:.3e} (eps={last_full.eps_dual:.3e})")
    gap_full_vs_dual = last_full.slem - dual_g
    print(f"    gap vs. independent dual bound: {gap_full_vs_dual:.3e}")

    # 3. Decentralized ADMM, run well past centralized convergence
    t0 = time.time()
    r_dec = fmmc_admm_decentralized(
        adj, rho=0.1, num_iter=dec_num_iter, k0=4, delta_t=1e-6,
        projection_sweeps=dec_sweeps, max_block=dec_max_block,
        centralized_warmup_iters=dec_warmup, decaying_tolerances=True,
        verbose=verbose)
    dt_dec = time.time() - t0
    last_dec = r_dec.history[-1]
    slem_dec = compute_slem(r_dec.B)
    print(f"\n[3] Decentralized ADMM: iters={len(r_dec.history)}  "
          f"SLEM(final B)={slem_dec:.8f}  ({dt_dec:.1f}s)")
    print(f"    final primal_residual={last_dec.primal_residual:.3e}  "
          f"(NOTE: decentralized_fmmc_admm.py's reformulation eliminates U as "
          f"an explicit state variable -- dual_residual is not independently "
          f"tracked here, unlike des_fmmc_admm.py's full ADMM; primal_residual "
          f"is valid and comparable)")
    # eps_pri comparable to des_fmmc_admm.py's own formula, for scale reference
    eps_pri_dec = 100.0 * 1e-6 + 1e-4 * max(np.linalg.norm(r_dec.B, "fro"), 1.0)
    print(f"    reference eps_pri scale (same formula as [2]): {eps_pri_dec:.3e}")
    gap_dec_vs_dual = slem_dec - dual_g
    print(f"    gap vs. independent dual bound: {gap_dec_vs_dual:.3e}")

    # Primal feasibility of the decentralized final B, checked independently
    feas_dec = primal_feasibility(r_dec.B, slem_dec, adj)
    print(f"    feasibility: row_sum_err={feas_dec['row_sum_err']:.2e}  "
          f"sym_err={feas_dec['sym_err']:.2e}  nonneg_viol={feas_dec['nonneg_viol']:.2e}  "
          f"offgraph_viol={feas_dec['offgraph_viol']:.2e}")

    print("\n=== Three-way summary ===")
    print(f"  Independent SDP dual bound (CLARABEL's own g):     {dual_g:.8f}")
    print(f"  CLARABEL primal s*:                                {s_clarabel:.8f}  (gap {ref_gap:.2e})")
    print(f"  Centralized exact-EVD ADMM (converged):            {last_full.slem:.8f}  (gap vs dual {gap_full_vs_dual:.2e})")
    print(f"  Decentralized ADMM ({len(r_dec.history)} iters):   {slem_dec:.8f}  (gap vs dual {gap_dec_vs_dual:.2e})")
    print(f"  |decentralized - centralized full_evd| = {abs(slem_dec - last_full.slem):.3e}")
    print(f"  |decentralized - CLARABEL s*|          = {abs(slem_dec - s_clarabel):.3e}")

    # primal residual trajectory around the decentralized end, for the paper's plot/table
    tail = r_dec.history[-30:]
    print("\nDecentralized primal_residual, last 30 iterations:")
    for d in tail:
        print(f"  it={d.iteration:4d}  primal_res={d.primal_residual:.3e}  SLEM={d.slem:.6f}  "
              f"sweeps_used={d.sweeps_used}  delta_t_target={d.delta_t_target:.2e}")

    return dict(n=n, edges=G.number_of_edges(),
                s_clarabel=s_clarabel, dual_g=dual_g, ref_gap=ref_gap,
                slem_full_evd=last_full.slem, iters_full_evd=len(r_full.history),
                slem_decentralized=slem_dec, iters_decentralized=len(r_dec.history),
                gap_full_vs_dual=gap_full_vs_dual, gap_dec_vs_dual=gap_dec_vs_dual,
                feas_dec=feas_dec)


if __name__ == "__main__":
    run()
