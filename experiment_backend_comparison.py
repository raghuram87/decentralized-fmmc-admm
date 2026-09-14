"""
Experiment: full-EVD vs. Lanczos vs. Chebyshev, measured by communication
rounds (the primary metric), with q_t and c0 reported separately, across
four graphs spanning the symmetry/degeneracy spectrum: a random geometric
graph (no automorphisms -- the friendliest case), a cycle, a hypercube,
and the Petersen graph (strongly regular, only 3 distinct eigenvalues --
the most degenerate case).

Steady-state communication cost is reported (averaged over the last few
ADMM iterations), since the first iteration's cold-start active set is
transient and not representative (near-degenerate warm starts can have k
close to n initially,
settling to a small steady-state k as ADMM converges).
"""
import numpy as np
import networkx as nx

from des_fmmc_admm import des_fmmc_admm
from fmmc_exact_admm import metropolis_hastings_weights, compute_slem


def build_graphs():
    graphs = {}
    n = 50
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=42)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
    graphs["random_geometric_n50"] = nx.to_numpy_array(G, dtype=float)

    graphs["cycle_n50"] = nx.to_numpy_array(nx.cycle_graph(50), dtype=float)

    H = nx.convert_node_labels_to_integers(nx.hypercube_graph(6))  # n=64
    graphs["hypercube_n64"] = nx.to_numpy_array(H, dtype=float)

    graphs["petersen_n10"] = nx.to_numpy_array(nx.petersen_graph(), dtype=float)
    return graphs


def graph_gossip_lambda2(adj):
    """
    SLEM of the graph's own LAZY Metropolis-Hastings matrix, W_lazy =
    (I+W)/2 -- the standard fix for bipartite graphs (e.g. the
    hypercube), whose plain MH matrix has an eigenvalue at exactly -1
    (lambda2_gossip=1, naive gossip never converges -- confirmed
    numerically: eigvalsh gave 1.0000000000000009 for hypercube_n64).
    Lazifying halves every eigenvalue's distance from 1, turning -1 into
    0, without changing the fixed point. This is what any real
    implementation would actually use, so it's the fair number to report.
    """
    n = adj.shape[0]
    W = metropolis_hastings_weights(adj)
    W_lazy = 0.5 * (np.eye(n) + W)
    J = np.ones((n, n)) / n
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * ((W_lazy - J) + (W_lazy - J).T)))))


def steady_state_summary(history, tail=5):
    tail_hist = history[-tail:] if len(history) >= tail else history
    q_t_avg = np.mean([h.q_t for h in tail_hist])
    k_avg = np.mean([h.k_active for h in tail_hist])
    rounds_avg = np.mean([h.comm.total_rounds for h in tail_hist if h.comm is not None]) if tail_hist[0].comm else None
    c0_avg = np.mean([h.comm.c0_per_application for h in tail_hist if h.comm is not None]) if tail_hist[0].comm else None
    return q_t_avg, k_avg, rounds_avg, c0_avg


def run():
    graphs = build_graphs()
    results = {}

    for gname, adj in graphs.items():
        n = adj.shape[0]
        lambda2 = graph_gossip_lambda2(adj)
        print("=" * 100)
        print(f"Graph: {gname}  (n={n}, edges={int(adj.sum()/2)}, MH gossip lambda2={lambda2:.4f})")
        print("=" * 100)
        print(f"{'backend':>10} | {'iters':>6} | {'SLEM':>10} | {'k* (avg tail)':>13} | "
              f"{'q_t (avg tail)':>14} | {'c0 (avg tail)':>13} | {'rounds (avg tail)':>17} | {'time(s)':>8}")

        graph_results = {}
        for backend in ["full_evd", "lanczos", "chebyshev"]:
            result = des_fmmc_admm(adj, backend=backend, rho=0.1, num_iter=800,
                                    lambda2_gossip=lambda2, eps_gossip=1e-4,
                                    verbose=False)
            last = result.history[-1]
            q_t_avg, k_avg, rounds_avg, c0_avg = steady_state_summary(result.history)
            graph_results[backend] = {
                "iters": last.iteration + 1, "slem": last.slem,
                "q_t_avg": q_t_avg, "k_avg": k_avg, "rounds_avg": rounds_avg, "c0_avg": c0_avg,
                "elapsed": result.elapsed,
            }
            rounds_str = f"{rounds_avg:,.0f}" if rounds_avg is not None else "n/a (centralized)"
            c0_str = f"{c0_avg:.2f}" if c0_avg is not None else "n/a"
            print(f"{backend:>10} | {last.iteration+1:>6} | {last.slem:>10.6f} | {k_avg:>13.1f} | "
                  f"{q_t_avg:>14.1f} | {c0_str:>13} | {rounds_str:>17} | {result.elapsed:>8.2f}")
        results[gname] = graph_results

        # Cross-backend accuracy check
        slems = [graph_results[b]["slem"] for b in ["full_evd", "lanczos", "chebyshev"]]
        spread = max(slems) - min(slems)
        print(f"SLEM spread across backends: {spread:.2e}  "
              f"({'OK -- all backends solve the same problem' if spread < 1e-2 else 'WARNING: large spread'})")
        print()

    print("=" * 100)
    print("Summary: Chebyshev vs. Lanczos communication-round ratio at steady state")
    print("=" * 100)
    for gname, r in results.items():
        if r["lanczos"]["rounds_avg"] and r["chebyshev"]["rounds_avg"]:
            ratio = r["lanczos"]["rounds_avg"] / r["chebyshev"]["rounds_avg"]
            print(f"{gname:>22}: Lanczos/Chebyshev rounds ratio = {ratio:,.1f}x "
                  f"(k*={r['chebyshev']['k_avg']:.1f})")

    return results


if __name__ == "__main__":
    run()
