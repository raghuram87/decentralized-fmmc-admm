"""
Push Chebyshev-ADMM / Lanczos-ADMM alone (no CVXPY/RK reference -- both are
inherently capped at small/moderate n by their own architectures, see
Section 15/15.2) well past the n=64 ceiling of experiment_backend_comparison.py.

Two fixes (des_fmmc_admm.py) made this practical rather than just slow:
  1. metropolis_hastings_weights(adj) was being recomputed once PER EDGE in
     the init list comprehension (O(m) redundant O(n^2) calls) -- hoisted
     out to a single call.
  2. compute_slem(B) -- a full dense eigh, O(n^3) -- ran every iteration
     purely for the diagnostic history, dwarfing the matrix-free oracle it
     was meant to be measuring (35% of total wall-clock at n=200, per
     cProfile). `slem_every` throttles this; default still 1 (unchanged
     behavior everywhere else in the codebase), used here as 10.

No sparse-B rewrite was needed: profiling showed compute_slem's O(n^3) eigh
dominated, not the O(n^2) dense Z/U dual-variable bookkeeping the ADMM
splitting itself requires (Z, U are dense n x n objects in this
formulation, not graph-sparse -- see Section 15.1 of the technical
report).
"""
import time
import numpy as np
import networkx as nx

from des_fmmc_admm import des_fmmc_admm


N_VALUES = [100, 200, 500, 1000]
NUM_ITER = 150
SLEM_EVERY = 20


def build_rgg(n, seed=42):
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=seed)
    tries = 0
    while not nx.is_connected(G):
        tries += 1
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
        if tries > 50:
            raise RuntimeError(f"Could not build a connected RGG at n={n}")
    return G, nx.to_numpy_array(G, dtype=float)


def run():
    print(f"{'n':>6} | {'edges':>7} | {'backend':>10} | {'SLEM':>10} | {'iters':>6} | "
          f"{'q_t(final)':>10} | {'k(final)':>8} | {'time(s)':>8}")
    results = []
    for n in N_VALUES:
        G, adj = build_rgg(n)
        for backend in ["chebyshev", "lanczos"]:
            t0 = time.time()
            try:
                result = des_fmmc_admm(adj, backend=backend, rho=0.1, num_iter=NUM_ITER,
                                        verbose=False, slem_every=SLEM_EVERY)
            except Exception as e:
                print(f"{n:>6} | {G.number_of_edges():>7} | {backend:>10} | FAILED: {e}")
                continue
            dt = time.time() - t0
            last = result.history[-1]
            print(f"{n:>6} | {G.number_of_edges():>7} | {backend:>10} | {last.slem:>10.6f} | "
                  f"{last.iteration + 1:>6} | {last.q_t:>10} | {last.k_active:>8} | {dt:>8.1f}")
            results.append(dict(n=n, edges=G.number_of_edges(), backend=backend,
                                 slem=last.slem, iters=last.iteration + 1,
                                 q_t=last.q_t, k_active=last.k_active, time_s=dt))
    return results


if __name__ == "__main__":
    run()
