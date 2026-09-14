"""
Block-size sweep (technical report Section 17, "Multiplicity" limitation --
"the block size must be chosen at least as large as the expected active-set
size -- which is itself not known a priori"). This experiment quantifies
the COST of guessing wrong: for each of the four graphs already used
throughout this project (experiment_backend_comparison.py's build_graphs,
each with a known steady-state active-set size k*), sweep the INITIAL block
size b0 across values below, at, and above k*, and measure how much extra
work (q_t, growth iterations) the adaptive block-growth mechanism
(spectral_oracle.py::oracle_chebyshev) has to do to recover from an
under-sized guess.

The adaptive mechanism (block_size doubles on every failed s-stability
certification, spectral_oracle.py's oracle_chebyshev while-loop) already
makes under-sizing SAFE -- it just costs more oracle applications. This
sweep is the first place that cost is measured directly rather than only
asserted.
"""
import numpy as np

from spectral_oracle import oracle_chebyshev, gershgorin_bound
from experiment_backend_comparison import build_graphs
from fmmc_exact_admm import metropolis_hastings_weights
from decentralized_chebyshev import mh_to_B


B0_VALUES = [1, 2, 4, 8, 16, 32]


def make_matvec(B):
    def matvec(x):
        x = np.asarray(x, dtype=float)
        Bx = B @ x
        Jx = np.mean(x, axis=0, keepdims=True) if x.ndim > 1 else np.mean(x)
        return Bx - Jx
    return matvec


def run():
    graphs = build_graphs()
    print(f"{'graph':>22} | {'b0':>4} | {'final b':>7} | {'growth iters (est.)':>19} | "
          f"{'k_active':>8} | {'q_t':>5} | {'certified':>9}")
    for gname, adj in graphs.items():
        n = adj.shape[0]
        B = mh_to_B(adj)
        matvec = make_matvec(B)
        row_abs_sum = np.abs(B).sum(axis=1) + 1.0
        for b0 in B0_VALUES:
            if b0 >= n:
                continue
            rng = np.random.default_rng(0)
            res = oracle_chebyshev(matvec, n, rho=1.0, row_abs_sum=row_abs_sum,
                                    s_estimate=0.5, block_size=b0, block_growth=2, rng=rng)
            final_b = res.extra["block_size"]
            growth_iters = round(np.log2(max(final_b / b0, 1.0))) + 1
            print(f"{gname:>22} | {b0:>4} | {final_b:>7} | {growth_iters:>19} | "
                  f"{res.k_active:>8} | {res.q_t:>5} | {str(res.certified):>9}")
        print()


if __name__ == "__main__":
    run()
