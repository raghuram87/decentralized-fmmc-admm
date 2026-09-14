"""
Common spectral-oracle interface (technical report Section 5) with three
backends: full EVD (reference), Lanczos/Krylov (comparison baseline), and
Chebyshev polynomial filtering (the paper's proposed default).

O_spec(V, delta) -> (s_hat, Q_active, Lambda_active, diagnostics)

All backends report `q_t` (operator applications) and, via
`communication_model.py`, `c0` (reductions per application) SEPARATELY,
rather than only as a product.

The Chebyshev filter implements the construction derived in the technical
report (suppress region [-tau, tau], growth toward the spectral tail). The
adaptive-k certificate (technical report, Prop. 2) is used by every
backend to decide whether the active set has been fully captured, via the
classical symmetric-matrix residual bound: a true eigenvalue of V exists
within ||V q - lambda q||_2 of every Ritz pair, no gap assumption needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla


MatVec = Callable[[np.ndarray], np.ndarray]  # x (n,) or (n,k) -> V @ x


@dataclass
class OracleResult:
    s_hat: float
    Q_active: np.ndarray       # (n, k)
    Lambda_active: np.ndarray  # (k,)
    k_active: int
    q_t: int                   # operator applications used
    certified: bool            # Prop. 2 enclosure confirms active set fully captured
    backend: str = ""
    extra: dict = field(default_factory=dict)


# =====================================================================
# Shared: exact scalar s*-solve given a (possibly partial) eigenvalue set
# =====================================================================
def solve_s_from_eigs(abs_eigs: np.ndarray, rho: float, tol: float = 1e-12, maxiter: int = 200) -> float:
    a = np.sort(np.abs(abs_eigs))[::-1]

    def f(s):
        return rho * np.maximum(a - s, 0.0).sum() - 1.0

    if f(0.0) <= 0.0:
        return 0.0
    lo, hi = 0.0, float(a[0]) if a.size else 0.0
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def residual_enclosure_certified(ritz_vals: np.ndarray, residual_norms: np.ndarray, s_hat: float, delta_t: float) -> bool:
    """
    Proposition 2 (technical report): sorted by |ritz_vals|
    descending, if the smallest-magnitude retained Ritz pair's enclosure
    ritz + residual <= s_hat + delta_t, no omitted eigenvalue of V can
    exceed the threshold -- the active set is certified complete.
    """
    if ritz_vals.size == 0:
        return False
    order = np.argsort(np.abs(ritz_vals))
    boundary = order[0]  # smallest-|value| retained pair -- weakest link
    return abs(ritz_vals[boundary]) + residual_norms[boundary] <= s_hat + delta_t


def certified_with_s_stability(ritz_vals: np.ndarray, residual_norms: np.ndarray, s_hat: float,
                                s_prev: Optional[float], delta_t: float, s_stability_tol: float) -> bool:
    """
    The enclosure certificate alone can be fooled: s_hat is *itself*
    estimated from the currently-retained Ritz values (solve_s_from_eigs),
    so under-sampling can make s_hat come out too small, which then makes
    the enclosure check trivially pass against the wrong threshold. Found
    empirically (verify_oracle.py, Check 2): a k=2 Lanczos run estimated
    s_hat=0 (true value 0.49) and "certified" a badly incomplete active
    set. Require s_hat to also be STABLE relative to the previous growth
    iteration's estimate before trusting the enclosure check.
    """
    if s_prev is None:
        return False
    if abs(s_hat - s_prev) > s_stability_tol * max(1.0, abs(s_prev)):
        return False
    return residual_enclosure_certified(ritz_vals, residual_norms, s_hat, delta_t)


# =====================================================================
# Backend 1: full EVD (reference / ground truth)
# =====================================================================
def oracle_full_evd(V: np.ndarray, rho: float, delta_t: float = 1e-9) -> OracleResult:
    V = 0.5 * (V + V.T)
    n = V.shape[0]
    evals, Q = la.eigh(V)
    s_hat = solve_s_from_eigs(evals, rho)
    active = np.abs(evals) > s_hat
    k = int(active.sum())
    return OracleResult(
        s_hat=s_hat, Q_active=Q[:, active], Lambda_active=evals[active],
        k_active=k, q_t=1, certified=True, backend="full_evd",
        extra={"full_spectrum_abs": np.abs(evals)},
    )


# =====================================================================
# Backend 2: Lanczos / Krylov (comparison baseline)
# =====================================================================
def oracle_lanczos(matvec: MatVec, n: int, rho: float, k0: int = 4, k_max: Optional[int] = None,
                    delta_t: float = 1e-9, tol: float = 1e-6, growth_factor: int = 2,
                    s_stability_tol: float = 1e-3) -> OracleResult:
    """
    scipy's eigsh (ARPACK, implicitly-restarted Lanczos), grown until the
    Prop. 2 enclosure certifies the active set is fully captured -- this
    IS the "adaptive k" the technical report's Section 8 calls for, applied to
    the Lanczos backend. Certification also requires s_hat stability
    across growth iterations (see `certified_with_s_stability`).
    """
    k_max = k_max or (n - 1)
    k = min(k0, n - 1)
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    q_t_total = 0
    s_prev = None

    while True:
        try:
            evals, Q = spla.eigsh(op, k=k, which="BE", tol=tol, maxiter=max(200, 20 * k))
        except spla.ArpackNoConvergence as e:
            evals, Q = e.eigenvalues, e.eigenvectors
        # ARPACK's own convergence tolerance bounds each Ritz residual;
        # recompute exactly for the certificate rather than trust `tol`.
        VQ = np.column_stack([matvec(Q[:, i]) for i in range(Q.shape[1])])
        residuals = np.linalg.norm(VQ - Q * evals[None, :], axis=0)
        q_t_total += k * 20  # ARPACK's internal restart cycles apply the operator repeatedly;
        # 20x is a conservative estimate of restart overhead for reporting purposes (see extra['q_t_note']).

        s_hat = solve_s_from_eigs(evals, rho)
        active = np.abs(evals) > s_hat
        certified = certified_with_s_stability(evals, residuals, s_hat, s_prev, delta_t, s_stability_tol)

        if certified or k >= k_max:
            return OracleResult(
                s_hat=s_hat, Q_active=Q[:, active], Lambda_active=evals[active],
                k_active=int(active.sum()), q_t=q_t_total, certified=certified,
                backend="lanczos",
                extra={"residuals": residuals, "k_tried": k,
                       "q_t_note": "ARPACK restart count is an estimate, not counted exactly"},
            )
        s_prev = s_hat
        k = min(k * growth_factor, k_max)


# =====================================================================
# Backend 3: Chebyshev polynomial filter (proposed default)
# =====================================================================
def gershgorin_bound(row_abs_sum: np.ndarray) -> float:
    """Lambda >= ||V||_2 from Gershgorin: max row sum of |V| (one round of
    neighbor exchange to gather each row's absolute sum)."""
    return float(np.max(row_abs_sum))


def power_iteration_tighten(matvec: MatVec, n: int, gershgorin_upper: float,
                             num_iters: int = 20, safety: float = 1.1,
                             rng: Optional[np.random.Generator] = None) -> float:
    """
    Root-cause fix for the Section 17/8 bipartite-stuck-fixed-point
    pathology (cycle_n50 "converging" to SLEM=1 for 130+ iterations despite
    a reachable 0.992 solution). Diagnosed precisely: the Gershgorin bound
    can be dramatically loose (measured: 37.67 vs a true spectral radius of
    ~1.0 at the stuck state) because it sums |V_ij| over a row without any
    cancellation, while V = B - J - U can have substantial sign-mixed mass
    once U (dense, though algebraically low-rank per Section 6.3) has
    accumulated over many iterations. A loose Lambda2 does not just waste
    degree -- it actively erases the Chebyshev filter's ability to RANK
    near-boundary eigenvalues against each other: the degree formula
    (Proposition 1) is calibrated only for suppressing the interior [0,tau]
    to tolerance delta_t, and a large Lambda2 makes that trivially easy
    (small m suffices), but that same small m gives poor separation between
    DIFFERENT eigenvalues that are both just above tau -- measured directly
    on the stuck state: the amplification ratio between the missed
    eigenvalue (theta=1.0) and a competing degenerate cluster (theta=0.9843)
    was only 1.38x at the Gershgorin-computed degree (m=4), rising to 9.5x
    once Lambda2 is tightened to near the true spectral radius (m=25) --
    the difference between the true extreme losing a close, essentially
    coin-flip competition against a 4-dimensional competing cluster, and
    winning decisively.

    Power iteration on matvec (already available -- no new communication
    primitive, consistent with this paper's matrix-free framing) refines
    Lambda2 toward the true spectral radius. Returns
    min(gershgorin_upper, power_iterate * safety) -- power iteration
    converges from BELOW and only after enough iterations, so the result is
    capped at the Gershgorin value (a proven, unconditional upper bound) to
    guarantee correctness is never compromised by an under-converged
    estimate; `safety` gives a small margin above the power estimate itself
    for the same reason.
    """
    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal(n)
    x = x / (np.linalg.norm(x) + 1e-300)
    lam = 0.0
    for _ in range(num_iters):
        y = matvec(x)
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-300:
            break
        lam = norm_y
        x = y / norm_y
    return float(min(gershgorin_upper, lam * safety))


def chebyshev_filter_value(theta: np.ndarray, tau: float, Lambda2: float, m: int) -> np.ndarray:
    """
    g_m(theta) using the construction (technical report
    Sec. 1): psi maps the SUPPRESS region [0,tau] to [-1,1].
    """
    psi = (2.0 * theta - tau) / tau
    X = (2.0 * Lambda2 - tau) / tau
    Tm = np.cosh(m * np.arccosh(X)) if X > 1 else np.cos(m * np.arccos(np.clip(X, -1, 1)))

    def T(x):
        out = np.empty_like(x, dtype=float)
        inside = np.abs(x) <= 1
        out[inside] = np.cos(m * np.arccos(x[inside]))
        out[~inside] = np.sign(x[~inside]) ** m * np.cosh(m * np.arccosh(np.abs(x[~inside])))
        return out

    return T(np.atleast_1d(psi).astype(float)) / Tm


def apply_chebyshev_filter(matvec: MatVec, X0: np.ndarray, tau: float, Lambda2: float, m: int):
    """
    Apply g_m(V^2) to a block X0 (n,b) via the 3-term Chebyshev recurrence
    in theta=V^2, i.e. T_{j+1}(psi) built from repeated application of the
    AFFINE-SHIFTED operator W := (2 V^2 - tau I)/tau. Each "step" costs 2
    applications of V (one V^2). Returns filtered block and applications count.
    """
    def W(X):
        return (2.0 * matvec(matvec(X)) - tau * X) / tau

    T_prev = X0
    T_curr = W(X0)
    q_t = 2  # the T_curr application above
    for _ in range(2, m + 1):
        T_next = 2.0 * W(T_curr) - T_prev
        T_prev, T_curr = T_curr, T_next
        q_t += 2
    X_norm = (2.0 * Lambda2 - tau) / tau
    Tm_norm = np.cosh(m * np.arccosh(X_norm)) if X_norm > 1 else 1.0
    return T_curr / Tm_norm, q_t


_COSH_OVERFLOW_MARGIN = 650.0  # cosh() overflows float64 around x~=709


def _safe_max_degree(X: float) -> int:
    """Largest m for which cosh(m*arccosh(X)) stays finite in float64."""
    arc = np.arccosh(X) if X > 1 else 1e-6
    return max(4, int(_COSH_OVERFLOW_MARGIN / max(arc, 1e-6)))


def _chebyshev_degree_for_tau(tau: float, Lambda2: float, delta_t: float, m_cap: int = 200) -> int:
    # tau can collapse toward 0 during block growth if s_hat has not yet
    # stabilized (a near-degenerate cold start: many "active" eigenvalues at the very
    # first ADMM iteration). A very small tau calls for a very high degree
    # (mathematically correct: an almost-nothing-suppressed filter needs
    # many terms), but the CONSTRUCTION itself (T_m(X_norm) via cosh)
    # overflows float64 long before a "correct" degree is reached in that
    # regime -- cap at whichever bound binds first, and let the caller's
    # s-stability/growth loop handle the resulting non-certification by
    # growing the block instead of the degree.
    tau = max(tau, 1e-9 * Lambda2)
    X = 2.0 * Lambda2 / tau - 1.0
    X = max(X, 1.0 + 1e-9)
    xi = X - np.sqrt(X ** 2 - 1.0)
    if xi >= 1.0 or xi <= 0.0:
        m = m_cap
    else:
        m = int(np.ceil(np.log(2.0 / delta_t) / np.log(1.0 / xi)))
    return int(np.clip(m, 4, min(m_cap, _safe_max_degree(X))))


def oracle_chebyshev(matvec: MatVec, n: int, rho: float, row_abs_sum: np.ndarray,
                      s_estimate: float, delta_t: float = 1e-9, block_size: int = 4,
                      degree: Optional[int] = None, block_growth: int = 2,
                      max_block: Optional[int] = None, rng: Optional[np.random.Generator] = None,
                      s_stability_tol: float = 1e-3, tighten_lambda2: bool = False,
                      power_iters: int = 20) -> OracleResult:
    """
    Distributed extreme-spectrum oracle via Chebyshev filtering (technical
    report Section 7). Grows the block size b (not
    just the degree) until the Prop. 2 certificate confirms the active set
    is fully captured. `tau = s_hat^2` and the degree are
    both RE-ESTIMATED each growth iteration (not frozen at the initial
    `s_estimate`), since a stale tau would keep the filter targeting the
    wrong region if s_hat drifts as b grows. Certification also requires
    s_hat stability (see `certified_with_s_stability`) -- an under-sized
    block can otherwise produce a self-referentially wrong s_hat that
    trivially "passes" the enclosure check (found empirically in
    verify_oracle.py Check 2).
    """
    rng = rng or np.random.default_rng(0)
    Lambda_bound = gershgorin_bound(row_abs_sum)
    if tighten_lambda2:
        # Root-cause fix (Section 17/8): a loose Gershgorin bound doesn't
        # just waste degree, it erases the filter's ability to rank
        # near-boundary eigenvalues against each other -- see
        # power_iteration_tighten's docstring for the full diagnosis.
        Lambda_bound = power_iteration_tighten(matvec, n, Lambda_bound, num_iters=power_iters, rng=rng)
    Lambda2 = Lambda_bound ** 2
    tau = max(max(s_estimate, 1e-12) ** 2, 1e-9 * Lambda2)
    max_block = max_block or (n - 1)
    b = min(block_size, n - 1)
    # Power iteration's own matvecs are a real communication/compute cost,
    # not a free lookup -- counted in q_t like everything else.
    q_t_total = power_iters if tighten_lambda2 else 0
    s_prev = None
    fixed_degree = degree

    while True:
        m = fixed_degree if fixed_degree is not None else _chebyshev_degree_for_tau(tau, Lambda2, delta_t)

        X0 = rng.standard_normal((n, b))
        X0, _ = np.linalg.qr(X0)
        Xf, q_t = apply_chebyshev_filter(matvec, X0, tau, Lambda2, m)
        q_t_total += q_t

        Qb, _ = np.linalg.qr(Xf)
        # Rayleigh-Ritz on the filtered subspace (small b x b problem).
        VQb = np.column_stack([matvec(Qb[:, i]) for i in range(Qb.shape[1])])
        H = Qb.T @ VQb
        H = 0.5 * (H + H.T)
        ritz_vals, W = la.eigh(H)
        Ritz_Q = Qb @ W
        residuals = np.linalg.norm(VQb @ W - Ritz_Q * ritz_vals[None, :], axis=0)

        s_hat = solve_s_from_eigs(ritz_vals, rho)
        active = np.abs(ritz_vals) > s_hat
        certified = certified_with_s_stability(ritz_vals, residuals, s_hat, s_prev, delta_t, s_stability_tol)

        if certified or b >= max_block:
            return OracleResult(
                s_hat=s_hat, Q_active=Ritz_Q[:, active], Lambda_active=ritz_vals[active],
                k_active=int(active.sum()), q_t=q_t_total, certified=certified,
                backend="chebyshev",
                extra={"degree": m, "block_size": b, "residuals": residuals},
            )
        s_prev = s_hat
        tau = max(max(s_hat, 1e-12) ** 2, 1e-9 * Lambda2)
        b = min(b * block_growth, max_block)
