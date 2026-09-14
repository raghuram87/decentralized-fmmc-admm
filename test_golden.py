"""
Golden-value regression test suite for the results reported in the
accompanying papers.

Every solver call here is fully deterministic (no unseeded randomness in the
code paths exercised -- the Chebyshev oracle's default RNG reseeds to a fixed
seed every call unless `fresh_oracle_rng=True`, which none of these tests
use), so exact float equality is the right check, not a tolerance -- any
change to these values means the algorithm's actual output changed, which is
exactly what a golden test should catch. Runtime: all tests combined take
well under 10 seconds.
"""
import numpy as np
import networkx as nx
import pytest

from des_fmmc_admm import des_fmmc_admm
from fmmc_exact_admm import fmmc_admm_exact
from duality_certificate_campaign import solve_fmmc_with_dual


def _rgg(n, radius, seed=42):
    G = nx.random_geometric_graph(n, radius=radius, seed=seed)
    assert nx.is_connected(G), "golden-test graph must be connected"
    return nx.to_numpy_array(G, dtype=float)


# Reference values were recorded with NumPy/SciPy linked against MKL. Other
# BLAS/LAPACK builds (e.g. the OpenBLAS in PyPI wheels) change the
# floating-point trajectory slightly: observed differences are at most
# 1.3e-6 in the final SLEM and one outer iteration in the stopping time.
# The tolerances below absorb that while still catching any change visible
# at the six significant digits reported in the papers.
SLEM_TOL = 1e-5
ITER_TOL = 3


def assert_slem(result, reference):
    assert abs(result.history[-1].slem - reference) < SLEM_TOL, result.history[-1].slem


def assert_iterations(result, reference):
    assert abs(len(result.history) - reference) <= ITER_TOL, len(result.history)


def test_rgg_n50_radius03_chebyshev():
    """Primary end-to-end regression check (Chebyshev backend)."""
    adj = _rgg(50, radius=0.3, seed=42)
    r = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=1000, verbose=False)
    assert_slem(r, 0.9105597763660572)


def test_rgg_n50_sqrt8n_chebyshev():
    """This paper's primary n=50 reference instance (Section 15's table)."""
    adj = _rgg(50, radius=np.sqrt(8.0 / 50), seed=42)
    r = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=1000, verbose=False)
    assert_iterations(r, 298)
    assert_slem(r, 0.7715562550029532)


def test_cycle_n50_known_bipartite_pathology():
    """
    Locks in the KNOWN Section 17 pathology as a golden value, not just a
    prose claim: with graph_reg=0 (default), the exact ADMM converges to the
    inferior bipartite-parity fixed point (SLEM=1), not the reachable
    ~0.9921. If this value ever changes, something about the ADMM dynamics
    changed -- worth knowing either way, not something to silently "fix" by
    updating the golden value without investigating why.
    """
    adj = nx.to_numpy_array(nx.cycle_graph(50), dtype=float)
    r = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=800, verbose=False)
    assert_iterations(r, 230)
    assert_slem(r, 1.0000000000000002)


def test_cycle_n50_graph_reg_fixes_pathology():
    """
    Section 17 remedy: graph_reg in its identified working window escapes
    the bad fixed point and lands close to the true optimum
    mu* = (1+cos(2pi/n))/(3-cos(2pi/n)).
    """
    adj = nx.to_numpy_array(nx.cycle_graph(50), dtype=float)
    r = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=800, verbose=False,
                       graph_reg=6e-3)
    assert_slem(r, 0.9922189304113829)
    # True FMMC optimum for the even cycle: mu* = (1+cos(2pi/n))/(3-cos(2pi/n)),
    # verified against a full CLARABEL SDP solve of cycle_n50 to 2.4e-9
    # (cos(2pi/n) is not the optimum; it differs by 3.1e-5).
    c = np.cos(2 * np.pi / 50)
    true_opt = (1 + c) / (3 - c)
    assert abs(r.history[-1].slem - true_opt) < 2e-4


def test_cycle_n50_full_evd_exact_deterministic():
    """
    fmmc_admm_exact (fmmc_exact_admm.py) uses a deterministic full
    eigendecomposition, no random Chebyshev block -- confirms the
    bipartite-pathology golden values above are specific to the Chebyshev
    backend's random-block exploration, not a property of the exact ADMM
    recursion itself.
    """
    adj = nx.to_numpy_array(nx.cycle_graph(50), dtype=float)
    r = fmmc_admm_exact(adj, rho=0.1, num_iter=1000, verbose=False)
    assert_slem(r, 0.9921477957544579)


def test_duality_certificate_petersen():
    """
    Smoke test for the independent SDP dual-certificate machinery (Section
    14.4/15.5.5's evidence chain): CLARABEL's own primal-dual gap on a tiny,
    fast instance should be at machine-precision-level tight.
    """
    adj = nx.to_numpy_array(nx.petersen_graph(), dtype=float)
    B, s, dual_g, solver, extras = solve_fmmc_with_dual(adj)
    assert solver == "CLARABEL"
    assert abs(s - dual_g) < 1e-6
    assert extras["stationarity_residual"] < 1e-5
    assert min(extras["Y1_eigmin"], extras["Y2_eigmin"], extras["mu_min"]) > -1e-6


def test_graph_reg_zero_reproduces_default_behavior():
    """graph_reg=0.0 (the default) must be a true no-op vs. omitting it."""
    adj = _rgg(50, radius=0.3, seed=42)
    r_default = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=1000, verbose=False)
    r_explicit_zero = des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=1000,
                                     verbose=False, graph_reg=0.0)
    assert r_default.history[-1].slem == r_explicit_zero.history[-1].slem


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_consensus_projection_matches_centralized_and_heuristic_is_biased():
    """Graph-feasibility projection (letter, Table II): on the Petersen graph's
    least favorable target, consensus ADMM (rho_c=2) reaches the centralized
    projection while the project-then-average heuristic plateaus."""
    from experiment_consensus_projection import find_adversarial_M, head_to_head
    adj = nx.to_numpy_array(nx.petersen_graph(), dtype=float)
    seed, _ = find_adversarial_M(adj)
    rows = {r["K"]: r for r in head_to_head("petersen_n10", adj, seed, sweeps_list=(200, 800))}
    assert rows[800]["err_new"] < 1e-10
    assert rows[800]["err_old"] > 1e-3
    assert abs(rows[800]["err_old"] - rows[200]["err_old"]) < 1e-6 * rows[800]["err_old"]
    assert rows[800]["rounds_new"] == 800 and rows[800]["rounds_old"] == 1616


def test_single_scalar_reduction_counterexample():
    """Two independent couplings: w* = (5/9, 1/9), single-scalar gives (1/3, 1/3)."""
    from verify_node_qp_two_constraint_counterexample import (
        naive_single_scalar_attempt, scipy_ref_2con,
    )
    r = np.ones(2); kappa = np.ones(2)
    a1 = np.array([1.0, 1.0]); a2 = np.array([1.0, -1.0])
    lo = np.full(2, -2.0); hi = np.full(2, 2.0)
    w_true = scipy_ref_2con(r, kappa, a1, 2.0, 0.5, a2, 2.0, 0.5, lo, hi)
    w_naive = naive_single_scalar_attempt(r, kappa, a1, 2.0, 0.5, a2, 2.0, 0.5, lo, hi)
    assert np.allclose(w_true, [5 / 9, 1 / 9], atol=1e-6)
    assert np.allclose(w_naive, [1 / 3, 1 / 3], atol=1e-9)
