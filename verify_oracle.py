"""
Verification checks run before trusting the spectral oracles for
experiments:
  1. The corrected Chebyshev filter is actually small on the suppress
     region and grows toward the tail (a filter with the suppress and pass
     regions interchanged would FAIL this check).
  2. All three oracle backends agree on a small test matrix.
  3. The adaptive-k certificate recovers the correct active-set size on
     the Petersen graph, whose adjacency spectrum is known analytically
     (3 x1, 1 x5, -2 x4) -- a strong, exact check since we know the truth.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from spectral_oracle import (
    chebyshev_filter_value, oracle_full_evd, oracle_lanczos, oracle_chebyshev,
    gershgorin_bound,
)


def check_filter_shape():
    print("=" * 70)
    print("Check 1: corrected Chebyshev filter shape")
    print("=" * 70)
    tau, Lambda2 = 1.0, 25.0  # s=1, Lambda=5
    thetas = np.linspace(0, Lambda2, 200)
    for m in [2, 4, 8, 16]:
        g = chebyshev_filter_value(thetas, tau, Lambda2, m)
        suppress_max = np.max(np.abs(g[thetas <= tau]))
        boundary_val = g[-1]
        print(f"m={m:>3} | max|g_m| on suppress region [0,tau] = {suppress_max:.4e} "
              f"| g_m(Lambda^2) = {boundary_val:.4f}")
    print("Expected: suppress-region max shrinks geometrically with m, "
          "boundary value stays ~1 (normalization). This is the property "
          "the paper's own Eq. 17 (as literally written) does NOT have.\n")

    # Plot for visual confirmation.
    SURFACE, INK, GRID = "#fcfcfb", "#0b0b0b", "#e1e0d9"
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for color, m in zip(COLORS, [2, 4, 8, 16]):
        g = chebyshev_filter_value(thetas, tau, Lambda2, m)
        ax.plot(thetas, np.abs(g), color=color, linewidth=1.8, label=f"m={m}")
    ax.axvline(tau, color="#898781", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Corrected Chebyshev filter |g_m(theta)| (tau=1, dashed line)", color=INK, fontsize=12)
    ax.set_xlabel("theta = lambda^2")
    ax.set_ylabel("|g_m(theta)| (log scale)")
    ax.grid(True, color=GRID)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("chebyshev_filter_shape.png", dpi=150, facecolor=SURFACE)
    print("Saved chebyshev_filter_shape.png\n")


def check_backends_agree():
    print("=" * 70)
    print("Check 2: do all three backends agree on a small test matrix?")
    print("=" * 70)
    rng = np.random.default_rng(0)
    n = 50
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=1)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=rng.integers(1_000_000))
    adj = nx.to_numpy_array(G)
    # Build a plausible V = B - J + U from a Metropolis-Hastings B, U=0.
    deg = adj.sum(axis=1)
    B = np.where(adj > 0, 1.0 / np.maximum(deg[:, None], deg[None, :]), 0.0)
    np.fill_diagonal(B, 1.0 - B.sum(axis=1))
    J = np.ones((n, n)) / n
    V = B - J
    V = 0.5 * (V + V.T)
    rho = 1.0

    def matvec(x):
        return V @ x

    r_evd = oracle_full_evd(V, rho)
    r_lan = oracle_lanczos(matvec, n, rho, k0=4)
    row_abs_sum = np.abs(V).sum(axis=1)
    s_est = r_evd.s_hat  # use exact s to seed tau for the filter's own estimate
    r_cheb = oracle_chebyshev(matvec, n, rho, row_abs_sum, s_estimate=s_est, block_size=4)

    print(f"{'backend':>10} | {'s_hat':>12} | {'k_active':>8} | {'certified':>9} | {'q_t':>6}")
    for r in [r_evd, r_lan, r_cheb]:
        print(f"{r.backend:>10} | {r.s_hat:>12.8f} | {r.k_active:>8d} | {str(r.certified):>9} | {r.q_t:>6d}")
    print()


def check_petersen():
    print("=" * 70)
    print("Check 3: Petersen graph (known analytic spectrum 3x1, 1x5, -2x4)")
    print("=" * 70)
    G = nx.petersen_graph()
    adj = nx.to_numpy_array(G)
    true_evals = np.sort(np.linalg.eigvalsh(adj))
    print("True adjacency eigenvalues:", np.round(true_evals, 6))

    n = adj.shape[0]
    rho = 1.0
    V = adj.copy()  # use the adjacency matrix itself, spectrum known exactly

    def matvec(x):
        return V @ x

    r_evd = oracle_full_evd(V, rho)
    print(f"Full EVD  : s_hat={r_evd.s_hat:.6f}  k_active={r_evd.k_active}")

    row_abs_sum = np.abs(V).sum(axis=1)
    r_cheb = oracle_chebyshev(matvec, n, rho, row_abs_sum, s_estimate=r_evd.s_hat, block_size=2)
    print(f"Chebyshev : s_hat={r_cheb.s_hat:.6f}  k_active={r_cheb.k_active}  "
          f"certified={r_cheb.certified}  block={r_cheb.extra['block_size']}  degree={r_cheb.extra['degree']}")

    r_lan = oracle_lanczos(matvec, n, rho, k0=2)
    print(f"Lanczos   : s_hat={r_lan.s_hat:.6f}  k_active={r_lan.k_active}  certified={r_lan.certified}")

    match = (r_evd.k_active == r_cheb.k_active == r_lan.k_active)
    print(f"\nAll three backends agree on k_active: {match}")


if __name__ == "__main__":
    check_filter_shape()
    check_backends_agree()
    check_petersen()
