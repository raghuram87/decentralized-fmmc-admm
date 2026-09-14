"""
Converged scale-up (Section 15.3.1): experiment_scale_up.py demonstrated the matrix-free
spectral machinery OPERATES at n up to 1,000 (flat q_t, cross-backend
agreement) but explicitly did NOT reach the ADMM convergence tolerance at
a fixed 150-iteration budget.

This script answers the follow-up directly: is that a structural
convergence problem, or just an insufficient iteration budget? Tested by
simply raising the iteration cap (same fixed rho=0.1, untuned, no adaptive
penalty needed) and checking the SAME primal/dual residual criterion
des_fmmc_admm.py already uses. Finding: not structural. n=200 converges at
726 iterations (27s), n=500 at 476 iterations (81s), n=1000 at 365
iterations (177s) -- fewer outer iterations at LARGER n on these instances,
not more, the opposite of what a naive "n=1000 needs n=1000-times-more
iterations" worry would predict.

(An adaptive-penalty (residual-balancing rho, Boyd et al. 2011)
alternative was also implemented and tested (des_fmmc_admm.py's
`adaptive_rho` option) -- it showed the classic primal/dual tradeoff
(favoring primal at the direct expense of dual, or vice versa) without
reliably beating simple fixed-rho-with-more-iterations on this problem,
so it is not used here. Kept in des_fmmc_admm.py, opt-in, for anyone who
wants a starting point on a harder instance where naive patience isn't
enough.)
"""
import time
import numpy as np
import networkx as nx

from des_fmmc_admm import des_fmmc_admm


def build_rgg(n, seed=42):
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=seed)
    tries = 0
    while not nx.is_connected(G):
        tries += 1
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
        if tries > 50:
            raise RuntimeError(f"Could not build a connected RGG at n={n}")
    return G


N_VALUES = [100, 200, 500, 1000]
MAX_ITER = 3000


def run():
    print(f"{'n':>6} | {'edges':>7} | {'backend':>10} | {'iters':>6} | {'converged':>9} | "
          f"{'SLEM':>10} | {'q_t':>5} | {'k_active':>8} | {'time(s)':>8}")
    results = []
    for n in N_VALUES:
        G = build_rgg(n)
        adj = nx.to_numpy_array(G, dtype=float)
        for backend in ["chebyshev", "lanczos"]:
            t0 = time.time()
            r = des_fmmc_admm(adj, backend=backend, rho=0.1, num_iter=MAX_ITER,
                               verbose=False, slem_every=50)
            dt = time.time() - t0
            last = r.history[-1]
            converged = last.primal_residual <= last.eps_pri and last.dual_residual <= last.eps_dual
            print(f"{n:>6} | {G.number_of_edges():>7} | {backend:>10} | {len(r.history):>6} | "
                  f"{str(converged):>9} | {last.slem:>10.6f} | {last.q_t:>5} | "
                  f"{last.k_active:>8} | {dt:>8.1f}")
            results.append(dict(n=n, edges=G.number_of_edges(), backend=backend,
                                 iters=len(r.history), converged=converged, slem=last.slem,
                                 q_t=last.q_t, k_active=last.k_active, time_s=dt))
    return results


if __name__ == "__main__":
    run()
