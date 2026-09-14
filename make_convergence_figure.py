"""
Diagnostic figure: eta_t,
delta_t (target), and the primal residual over one complete decentralized
FMMC-ADMM run with the consensus-projection fix (Section 15.8) and
decaying tolerances (Section 15.7/10), n=50, so Section 10's theory and
Section 15.9's result connect visually, not just in prose tables.

Data points below are the actual values printed by a real run
(`decentralized_fmmc_admm.py::fmmc_admm_decentralized`, rho=0.1,
projection_sweeps=100 base with decaying_tolerances=True,
use_consensus_projection=True, rho_cons=2.0, centralized_warmup_iters=15),
recorded directly from its own verbose output every 25 iterations (this
session's live run, stopped at iteration 600 once the primal residual
visibly entered a stable oscillation band rather than continuing to
shrink -- Section 15.9 reports and explains this). delta_t_target
is NOT separately logged in the verbose print; it is recomputed here from
the known, deterministic schedule formula
(`spectral_delta_t_schedule(it_local, delta_t0=1e-6, eps=0.5)`,
it_local = it - 15) rather than re-run, since that formula is exact and
was already verified against real execution in Section 15.7.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (iteration, SLEM, eta_t=projection_rel_err, r_p=primal_residual)
DATA = [
    (1, 0.897972, 0.0, 1.737e+00),
    (25, 0.788041, 4.99e-09, 1.200e-01),
    (50, 0.775071, 4.73e-10, 7.407e-02),
    (75, 0.772749, 1.08e-09, 5.325e-02),
    (100, 0.779349, 1.20e-08, 1.385e-02),
    (125, 0.772667, 1.01e-09, 8.661e-03),
    (150, 0.771844, 4.35e-10, 5.058e-03),
    (175, 0.771619, 2.61e-10, 3.216e-03),
    (200, 0.771556, 6.46e-10, 2.005e-03),
    (225, 0.771532, 7.50e-10, 1.358e-03),
    (250, 0.771521, 3.43e-09, 9.679e-04),
    (275, 0.771514, 2.67e-09, 7.542e-04),
    (300, 0.771508, 2.58e-09, 7.520e-04),
    (325, 0.771504, 5.12e-10, 7.250e-04),
    (350, 0.771500, 1.86e-09, 6.053e-04),
    (375, 0.771498, 1.24e-09, 5.405e-04),
    (400, 0.771497, 6.54e-10, 5.861e-04),
    (425, 0.771495, 3.16e-10, 5.940e-04),
    (450, 0.771493, 1.31e-10, 6.133e-04),
    (475, 0.771491, 1.30e-10, 4.409e-04),
    (500, 0.771489, 1.14e-09, 6.052e-04),
    (525, 0.771488, 1.81e-10, 5.983e-04),
    (550, 0.771487, 7.16e-11, 5.996e-04),
    (575, 0.771486, 7.54e-11, 4.537e-04),
    (600, 0.771485, 1.43e-10, 5.982e-04),
]

TRUE_OPT = 0.77146797826736
CENTRALIZED_ADMM = 0.77150137
WARMUP = 15
EPS_PRI_APPROX = 3.47e-4  # observed eps_pri, stable across steady state


def delta_t_target(it, delta_t0=1e-6, eps=0.5, warmup=WARMUP):
    it_local = max(it - warmup, 0)
    return delta_t0 / (it_local + 1) ** (1.0 + eps)


def make_figure():
    it = np.array([d[0] for d in DATA])
    slem = np.array([d[1] for d in DATA])
    eta_t = np.array([d[2] for d in DATA])
    r_p = np.array([d[3] for d in DATA])
    delta_t = np.array([delta_t_target(i) for i in it])

    eta_plot = np.clip(eta_t, 1e-11, None)  # floor for the it=1 exact-zero point (log scale)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax = axes[0]
    ax.semilogy(it, eta_plot, "o-", label=r"$\eta_t$ (graph-projection error, consensus fix)", color="#1f77b4")
    ax.semilogy(it, delta_t, "--", label=r"$\delta_t$ target (spectral tolerance schedule)", color="#ff7f0e")
    ax.semilogy(it, r_p, "o-", label=r"primal residual $r_p^t$", color="#2ca02c")
    ax.axhline(EPS_PRI_APPROX, color="#2ca02c", linestyle=":", linewidth=1,
               label=r"$r_p^t$ stopping tolerance ($\approx 3.47\times10^{-4}$)")
    ax.set_ylabel("value (log scale)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(r"$n=50$: consensus projection + decaying tolerances (real run, stopped at $t=600$)")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    ax2.plot(it, slem, "o-", color="#d62728", label="SLEM (decentralized)")
    ax2.axhline(TRUE_OPT, color="black", linestyle="--", linewidth=1,
                label=f"true optimum, CLARABEL ({TRUE_OPT:.6f})")
    ax2.axhline(CENTRALIZED_ADMM, color="gray", linestyle=":", linewidth=1,
                label=f"centralized exact-EVD ADMM ({CENTRALIZED_ADMM:.6f})")
    ax2.set_xlabel(r"outer ADMM iteration $t$")
    ax2.set_ylabel("SLEM")
    ax2.set_ylim(0.7710, 0.79)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convergence_diagnostics_n50.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")
    print(f"final (t={it[-1]}): eta_t={eta_t[-1]:.3e} delta_t_target={delta_t[-1]:.3e} "
          f"r_p={r_p[-1]:.3e} slem={slem[-1]:.6f}  gap_vs_true={slem[-1]-TRUE_OPT:.3e}")


if __name__ == "__main__":
    make_figure()
