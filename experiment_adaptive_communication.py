"""
W_t adaptive-communication ablation (technical report Section 12, RQ6): does
using the EVOLVING ADMM iterate B^t as the gossip/reduction matrix for the
spectral oracle -- instead of a fixed Metropolis-Hastings matrix -- reduce
total communication, at matched final SLEM accuracy (same ADMM trajectory,
only the reported communication cost differs)?

Three strategies for W_t (Eq. 33-35), all evaluated on the SAME single ADMM
run per graph -- W_t only feeds the communication-cost MODEL (Eq. 30-31),
not the ADMM update itself, so one real trajectory per graph is enough to
compare all three post-hoc, no need to re-run ADMM three times per graph:

  1. Fixed Metropolis-Hastings (baseline): W_t = W_MH, constant. Same lazy-
     matrix convention used everywhere else in this paper
     (experiment_backend_comparison.py::graph_gossip_lambda2) for bipartite
     safety.
  2. W_t = B^t (Eq. 33): use the CURRENT ADMM iterate directly. Since B^t
     converges toward the FMMC-optimal transition matrix, its own SLEM
     should improve as ADMM progresses -- but Eq. 33's own text warns ADMM
     iterates are NOT guaranteed to improve SLEM monotonically, so this can
     also regress mid-run.
  3. Safeguarded interpolation (Eq. 35): the paper defines the INTERPOLATION
     FORM but not a schedule for theta_t. We adopt the natural "safeguarded"
     reading -- never regress: beta_used[t] = min(beta_used[t-1], SLEM(B^t)),
     i.e. only adopt B^t once it demonstrably beats the best mixing rate
     seen so far, otherwise keep the previous (safer) W_{t-1}. This is a
     hard theta_t in {0,1} rather than a continuous blend, but it is the
     most defensible literal reading of "safeguarded" and is stated
     explicitly here since the paper does not pin the schedule down.

Total communication for a strategy = sum over ADMM iterations of
q_t * c0_t * R_t(beta_t) (Eq. 31), using each iteration's ALREADY-MEASURED
q_t and c0 (from the Chebyshev oracle's own communication-cost accounting,
communication_model.py) -- only beta_t/R_t differs across the three
strategies.

Run on the SAME four graphs used throughout this paper
(experiment_backend_comparison.py::build_graphs) rather than the single RGG
instance the first pass of this experiment used -- one graph cannot support
a general claim about whether the safeguard ever matters (on RGG, raw B^t
never actually regressed below the fixed-MH baseline, so the safeguard never
triggered there; testing the harder-mixing graphs already known to have
larger active sets in this paper -- cycle_n50, hypercube_n64 -- checks
whether that finding is a general pattern or an artifact of one easy graph).
"""
import numpy as np

from des_fmmc_admm import des_fmmc_admm
from fmmc_exact_admm import metropolis_hastings_weights
from communication_model import gossip_rounds_for_tolerance
from experiment_backend_comparison import build_graphs

EPS_GOSSIP = 1e-4


def lazy_mh_slem(adj):
    n = adj.shape[0]
    W = metropolis_hastings_weights(adj)
    W_lazy = 0.5 * (np.eye(n) + W)
    J = np.ones((n, n)) / n
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * ((W_lazy - J) + (W_lazy - J).T)))))


def run_on_graph(gname, adj, num_iter=300, rho=0.1, verbose=True):
    result = des_fmmc_admm(adj, backend="chebyshev", rho=rho, num_iter=num_iter,
                            verbose=False, track_B_history=True)
    history = result.history

    beta_fixed_mh = lazy_mh_slem(adj)
    beta_Bt = np.array([h.slem for h in history])
    n_worse_than_mh = int(np.sum(beta_Bt > beta_fixed_mh))
    n_iters = len(beta_Bt)

    diffs = np.diff(beta_Bt)
    increases = diffs[diffs > 0]
    n_regressions = len(increases)
    max_regression = float(increases.max()) if n_regressions else 0.0

    beta_safeguarded = np.minimum.accumulate(np.concatenate([[beta_fixed_mh], beta_Bt]))[1:]

    strategies = {
        "fixed_mh": np.full(n_iters, beta_fixed_mh),
        "raw_Bt": beta_Bt,
        "safeguarded": beta_safeguarded,
    }

    totals = {}
    for key, betas in strategies.items():
        total_rounds = 0
        for h, beta_t in zip(history, betas):
            if h.comm is None:
                continue
            R_t = gossip_rounds_for_tolerance(float(beta_t), EPS_GOSSIP)
            total_rounds += h.q_t * h.comm.c0_per_application * R_t
        totals[key] = total_rounds

    ratio_raw = totals["raw_Bt"] / totals["fixed_mh"] if totals["fixed_mh"] else float("nan")
    ratio_safe = totals["safeguarded"] / totals["fixed_mh"] if totals["fixed_mh"] else float("nan")

    row = dict(graph=gname, n=adj.shape[0], iters=n_iters, final_slem=history[-1].slem,
               beta_fixed_mh=beta_fixed_mh, beta_Bt_min=beta_Bt.min(), beta_Bt_max=beta_Bt.max(),
               n_worse_than_mh=n_worse_than_mh, n_regressions=n_regressions,
               max_regression=max_regression, totals=totals, ratio_raw=ratio_raw, ratio_safe=ratio_safe)

    if verbose:
        print(f"{gname:>22} | iters={n_iters:4d} | final SLEM={history[-1].slem:.6f} | "
              f"fixed-MH beta={beta_fixed_mh:.4f} | B^t range=[{beta_Bt.min():.4f},{beta_Bt.max():.4f}] | "
              f"worse-than-MH={n_worse_than_mh}/{n_iters} | regressions={n_regressions}/{n_iters-1} "
              f"(max {max_regression:.5f}) | raw B^t={ratio_raw:.3f}x | safeguarded={ratio_safe:.3f}x")
    return row


def run():
    graphs = build_graphs()
    print(f"{'graph':>22} | {'iters':>9} | {'final SLEM':>10} | {'fixed-MH beta':>13} | "
          f"{'B^t range':>17} | {'worse-than-MH':>13} | {'regressions':>19} | {'raw B^t':>8} | {'safeguarded':>11}")
    rows = []
    for gname, adj in graphs.items():
        # 800 iterations, matching experiment_backend_comparison.py's own
        # setup for these graphs (cycle_n50/hypercube_n64 need more than
        # 300 to converge; using a different budget than the paper's other
        # tables would make this ablation's beta_t trajectory unrepresentative).
        rows.append(run_on_graph(gname, adj, num_iter=800))

    print(f"\n{'summary':>22} | {'raw B^t vs fixed-MH':>20} | {'safeguarded vs fixed-MH':>24}")
    for r in rows:
        print(f"{r['graph']:>22} | {r['ratio_raw']:>19.3f}x | {r['ratio_safe']:>23.3f}x")

    return rows


if __name__ == "__main__":
    run()
