"""
Statistical Delta_obj campaign (Section 14.4): max/median/95th-percentile
across MANY graph instances. For each instance, compares
our exact ADMM solver (fmmc_exact_admm.py) against the independent CVXPY
reference (centralized_reference.py) on the SAME graph.

Graph families (matching a subset of Section 14.3's list, kept small for
CVXPY solve speed -- this campaign targets statistical breadth over many
SMALL instances, not scale; scale is addressed separately):
  - random geometric  (no automorphisms)
  - cycle             (vertex-transitive, only n varies -- no seed)
  - Erdos-Renyi        (near the connectivity threshold)
  - random regular    (d=3, tests moderate symmetry)
  - Barabasi-Albert   (m=2, always connected, heterogeneous degree)
"""
import numpy as np
import networkx as nx

from centralized_reference import solve_fmmc_cvxpy, compute_slem as compute_slem_ref
from fmmc_exact_admm import fmmc_admm_exact


N_VALUES = [10, 15, 20, 25]
SEEDS_PER_CONFIG = 3


def connected_or_retry(builder, max_tries=50):
    for attempt in range(max_tries):
        G = builder(attempt)
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not build a connected instance after many retries.")


def build_instances():
    instances = []  # list of (label, adj)

    for n in N_VALUES:
        for seed in range(SEEDS_PER_CONFIG):
            G = connected_or_retry(lambda a, n=n, seed=seed: nx.random_geometric_graph(
                n, radius=np.sqrt(8.0 / n), seed=1000 * seed + a))
            instances.append((f"random_geometric_n{n}_s{seed}", nx.to_numpy_array(G, dtype=float)))

    for n in N_VALUES:
        instances.append((f"cycle_n{n}", nx.to_numpy_array(nx.cycle_graph(n), dtype=float)))

    for n in N_VALUES:
        for seed in range(SEEDS_PER_CONFIG):
            p = min(1.0, 2.0 * np.log(n) / n + 0.1)
            G = connected_or_retry(lambda a, n=n, seed=seed, p=p: nx.erdos_renyi_graph(
                n, p, seed=1000 * seed + a))
            instances.append((f"erdos_renyi_n{n}_s{seed}", nx.to_numpy_array(G, dtype=float)))

    for n in N_VALUES:
        if (n * 3) % 2 != 0:
            continue  # random_regular_graph requires n*d even
        for seed in range(SEEDS_PER_CONFIG):
            G = connected_or_retry(lambda a, n=n, seed=seed: nx.random_regular_graph(
                3, n, seed=1000 * seed + a))
            instances.append((f"random_regular3_n{n}_s{seed}", nx.to_numpy_array(G, dtype=float)))

    for n in N_VALUES:
        for seed in range(SEEDS_PER_CONFIG):
            G = nx.barabasi_albert_graph(n, 2, seed=1000 * seed + 7)
            instances.append((f"barabasi_albert_n{n}_s{seed}", nx.to_numpy_array(G, dtype=float)))

    return instances


def run_campaign():
    instances = build_instances()
    print(f"Total instances: {len(instances)}\n")
    print(f"{'instance':>30} | {'SLEM (ref)':>11} | {'SLEM (ours)':>12} | {'Delta_obj':>10} | {'solver':>8}")

    deltas = []
    failures = []
    for label, adj in instances:
        try:
            B_ref, s_ref, status, solver = solve_fmmc_cvxpy(adj)
            slem_ref = compute_slem_ref(B_ref)
            result = fmmc_admm_exact(adj, rho=0.1, num_iter=1000, verbose=False)
            slem_ours = result.history[-1].slem
            delta = abs(slem_ref - slem_ours)
            deltas.append(delta)
            print(f"{label:>30} | {slem_ref:>11.7f} | {slem_ours:>12.7f} | {delta:>10.2e} | {solver:>8}")
        except Exception as e:
            failures.append((label, str(e)))
            print(f"{label:>30} | FAILED: {e}")

    deltas = np.array(deltas)
    print("\n" + "=" * 70)
    print(f"n = {len(deltas)} successful instances ({len(failures)} failed)")
    print(f"max Delta_obj    : {deltas.max():.3e}")
    print(f"median Delta_obj : {np.median(deltas):.3e}")
    print(f"95th pct Delta_obj: {np.percentile(deltas, 95):.3e}")
    print(f"mean Delta_obj   : {deltas.mean():.3e}")
    if failures:
        print("\nFailures:")
        for label, err in failures:
            print(f"  {label}: {err}")

    return deltas, failures


if __name__ == "__main__":
    run_campaign()
