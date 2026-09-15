"""
Decentralized solver for the graph-feasibility projection
(fmmc_exact_admm.py::EdgeGraphProjection), the one piece of the ADMM loop
that was still centralized (solved via a single global OSQP QP). This
closes that gap via a Jacobi-style local iteration, derived directly from
EdgeGraphProjection's own KKT conditions -- not a new formulation, a
decentralized SOLVER for the exact same QP.

Derivation. EdgeGraphProjection solves
    min_w  sum_i (1-M_ii-(Aw)_i)^2 + 2 sum_e (w_e-M_ij)^2
    s.t.   w>=0, Aw<=1
Setting the unconstrained gradient (w.r.t. w_e=(i,j)) to zero and splitting
off w_e's own diagonal contribution (P_ee=8, from P=2A^TA+4I) gives the
correctly-diagonal-split Jacobi fixed-point relation
    w_ij = 0.25*(1-M_ii) + 0.25*(1-M_jj) + 0.5*M_ij
           - 0.25*S_i^{other} - 0.25*S_j^{other}                        (*)
where S_i^{other} = S_i - w_ij is node i's row sum over its OTHER edges
(excluding w_ij itself). Naively using the FULL S_i (including w_ij) here
introduces a spurious self-coupling term that makes the iteration a period-2
oscillator, not a contraction -- found empirically (first version of this
file oscillated between two answers every other sweep, never converging).
A Jacobi sweep updates every edge from (*) using the row sums from BEFORE
the sweep (parallelizable, symmetric), each requiring exactly one exchange
of (M_ii, S_i) with each neighbor.

The box+row-sum constraint {w>=0, sum_j w_ij<=1} is NOT separable across a
node's edges after this unconstrained step, so it is enforced by a LOCAL
"capped simplex" projection of each node's own row (standard Euclidean
projection onto {x>=0, sum(x)<=1} via bisection on the shift threshold --
pure local computation, no communication). Projecting independently at
both endpoints of an edge can disagree on the shared w_ij, so each sweep
ends with one more exchange to SYMMETRIZE: w_ij_agreed =
0.5*(w_ij_from_i + w_ij_from_j).

This does not reproduce OSQP's solution exactly after a few sweeps -- it
is an INEXACT decentralized projection, exactly the kind of inexactness
Theorem 1 (Section 10) already covers, provided the resulting errors are
summable. The approximation gap is measured directly here against the
centralized OSQP reference, not asserted away.
"""
from __future__ import annotations

import numpy as np

from decentralized_chebyshev import CommLedger


def solve_node_qp(d: np.ndarray, t: np.ndarray, M_ii: float, rho_cons: float,
                   cap: float = 1.0) -> np.ndarray:
    """
    EXACT local solve of node i's own consensus-ADMM subproblem, used by
    the consensus graph projection that replaces the biased
    Jacobi+symmetrization heuristic:

        minimize_{w>=0, sum(w)<=cap}
            sum_e (w_e - d_e)^2 + (1 - M_ii - sum_e w_e)^2
            + (rho_cons/2) sum_e (w_e - t_e)^2

    d_e is the edge's own target (M_ij, known to both endpoints), t_e is
    this node's current consensus target z_e - u_i^e for edge e.

    This is NOT the same operation as "solve the unconstrained stationarity
    for sum_e w_e treating every edge as free, then apply a plain Euclidean
    capped-simplex projection to the result" -- that simpler two-step
    recipe (what an earlier draft of this fix, and structurally the
    ORIGINAL Jacobi scheme's own per-node step, both do) is only exact when
    the sum<=cap constraint is the ONLY active constraint; empirically
    verified here to be WRONG (by up to 0.42 in a direct comparison against
    a constrained QP solve) whenever nonnegativity clips some but not all
    coordinates while sum(w)<cap. The two-case construction below is
    verified exact against `scipy.optimize.minimize` (SLSQP) to
    ~1.8e-7 (solver-tolerance-limited) across 500 random trials spanning
    both regimes.

    Derivation: the separable part of the objective, sum_e(w_e-d_e)^2 +
    (rho_cons/2)(w_e-t_e)^2, is itself a per-coordinate quadratic minimized
    at r_e = (2 d_e + rho_cons t_e)/(2+rho_cons) with combined curvature
    (2+rho_cons). Writing c=1-M_ii, the full objective becomes
    (2+rho_cons)/2 * ||w-r||^2 + (sum(w)-c)^2 + const, i.e. an ISOTROPIC
    distance-to-r term plus a SEPARATE penalty purely on the sum -- unlike
    the sum<=cap CONSTRAINT's Lagrange multiplier, this penalty is present
    whether or not the constraint binds, so it must be folded in for both
    regimes, not just when cap is active.

    Case 1 (sum constraint inactive): every free coordinate satisfies
    w_e=max(0,r_e-shift) for a SINGLE shared `shift`, self-consistently
    solving shift = 2*(S(shift)-c)/(2+rho_cons) where S(shift)=sum_e
    max(0,r_e-shift) -- a monotone scalar equation, solved by bisection.
    If the resulting S(shift) <= cap, this IS the constrained optimum.

    Case 2 (sum constraint active, S=cap): the same per-coordinate form
    applies with shift = 2*(cap-c)/(2+rho_cons) + extra, extra>=0 the sum
    constraint's own multiplier (scaled) -- i.e. a plain capped-simplex
    projection of the ALREADY-SHIFTED vector r-2*(cap-c)/(2+rho_cons), not
    of r itself (dropping this fixed shift is exactly the bug the
    empirical check above caught).
    """
    d = np.asarray(d, dtype=float)
    t = np.asarray(t, dtype=float)
    k = d.size
    if k == 0:
        return d
    r = (2.0 * d + rho_cons * t) / (2.0 + rho_cons)
    c = 1.0 - M_ii

    def S_of(rvec, shift):
        return np.maximum(rvec - shift, 0.0).sum()

    def F(shift):
        return shift - 2.0 * (S_of(r, shift) - c) / (2.0 + rho_cons)

    span = 10.0 * (np.abs(r).max() + abs(c) + 1.0) if k else 1.0
    lo, hi = -span, span
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if F(mid) < 0:
            lo = mid
        else:
            hi = mid
    shift1 = 0.5 * (lo + hi)
    S1 = S_of(r, shift1)
    if S1 <= cap + 1e-10:
        return np.maximum(r - shift1, 0.0)

    base_shift = 2.0 * (cap - c) / (2.0 + rho_cons)
    r2 = r - base_shift
    lo2, hi2 = 0.0, float(max(np.max(r2), 0.0)) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo2 + hi2)
        if S_of(r2, mid) > cap:
            lo2 = mid
        else:
            hi2 = mid
    extra = 0.5 * (lo2 + hi2)
    return np.maximum(r2 - extra, 0.0)


def decentralized_graph_projection_consensus(edges, n, M_ii, M_ij_by_edge, w_init,
                                              neighbors, edge_index_by_node,
                                              num_sweeps, ledger: CommLedger,
                                              rho_cons: float = 1.0, callback=None,
                                              lam_init=None, return_duals: bool = False):
    """
    Warm starts: `w_init` initializes the consensus variable z and
    `lam_init` (default zeros) the per-edge scaled dual u_i^e of the
    lower-index endpoint (the other endpoint holds -lam_init, preserving
    u_i^e + u_j^e = 0). With return_duals=True, returns (z, S, lam).

    Distributed CONSENSUS ADMM for the graph-feasibility projection
    (replaces the biased `decentralized_graph_projection` heuristic
    above).

    Each edge e=(i,j) gets TWO local copies, w_i^e (node i's own opinion)
    and w_j^e (node j's), reconciled by a consensus variable z_e and
    per-node scaled dual u_i^e -- NOT by naively averaging two independent
    projections, which Section 15.7 found converges to a measured,
    non-vanishing bias for at least some M encountered on real trajectories.
    This is the standard global-consensus ADMM construction (Boyd, Parikh,
    Chu, Peleato & Eckstein 2011, Sec. 7.1 -- already reference #3 in this
    paper), specialized to N=2 endpoints per shared variable, applied
    edge-by-edge across the whole graph:

        w_i^{k+1} = argmin_{w_i feasible} f_i(w_i) + (rho_cons/2)||w_i - z^k + u_i^k||^2   (LOCAL, exact via solve_node_qp)
        z_e^{k+1} = (w_i^e + u_i^e + w_j^e + u_j^e) / 2                                    (needs ONE exchange)
        u_i^{k+1} = u_i^k + w_i^{k+1} - z^{k+1}                                            (LOCAL)

    Standard global-consensus ADMM theory (convex f_i, convex per-node
    feasible sets -- both true here) guarantees w_i^e, w_j^e -> the SAME
    value and that value solves the TRUE joint QP `EdgeGraphProjection`
    solves centrally -- i.e. eta_t -> 0 as num_sweeps -> infinity, the
    property Section 15.7 found the OLD heuristic structurally lacks.

    A simplification, not an approximation: because there are only
    2 "copies" per consensus variable, the invariant u_j^e = -u_i^e holds
    for all k whenever both start at 0 (a standard fact about global
    consensus ADMM: sum_i u_i^k is invariant and starts at 0) -- so only
    ONE scaled dual per edge needs to be tracked (`lam`, owned by the
    LOWER-indexed endpoint), and it turns out z_e's u-terms cancel exactly:
    z_e = (w_i^e+u_i^e+w_j^e+u_j^e)/2 = (w_i^e+w_j^e)/2. This also means
    this scheme needs LESS per-sweep communication than the heuristic it
    replaces: no exchange of (M_ii, S_i) is needed at all (S_i is now
    purely local, each node using only its OWN private copy), only ONE
    round exchanging each edge's two primal iterates -- the same single
    exchange the old scheme's "symmetrization" step already required.

    Returns (w, S) after `num_sweeps` ADMM sweeps, where w=z (the converged
    consensus values) -- both endpoints agree on it BY CONSTRUCTION once
    consensus is reached, unlike the old scheme's persistently-disagreeing
    endpoint views.
    """
    m = len(edges)
    lam = np.zeros(m) if lam_init is None else np.asarray(lam_init, dtype=float).copy()   # lam[e] = u_i^e for i = edges[e][0] (lower index); u_j^e = -lam[e]
    z = w_init.copy()

    for _ in range(num_sweeps):
        # LOCAL step at every node -- NO communication needed here (a
        # reduction vs. the old scheme, which needed an S_i exchange first).
        x_lo = np.empty(m)
        x_hi = np.empty(m)
        for i in range(n):
            idxs = edge_index_by_node[i]
            if not idxs:
                continue
            d_local = np.empty(len(idxs))
            t_local = np.empty(len(idxs))
            for pos, e_idx in enumerate(idxs):
                lo, hi = edges[e_idx]
                sign = 1.0 if lo == i else -1.0
                u_i_e = sign * lam[e_idx]
                d_local[pos] = M_ij_by_edge[e_idx]
                t_local[pos] = z[e_idx] - u_i_e
            x_i = solve_node_qp(d_local, t_local, M_ii[i], rho_cons, cap=1.0)
            for pos, e_idx in enumerate(idxs):
                lo, hi = edges[e_idx]
                if lo == i:
                    x_lo[e_idx] = x_i[pos]
                else:
                    x_hi[e_idx] = x_i[pos]

        # ONE round: exchange each endpoint's own primal iterate for the
        # shared edge -- the only communication this scheme needs per sweep.
        ledger.log_round(2 * m, 1)
        z_new = 0.5 * (x_lo + x_hi)
        lam = lam + x_lo - z_new
        z = z_new
        # Optional per-sweep observer; returning True stops early.
        if callback is not None and callback(z):
            break

    S = np.zeros(n)
    for i in range(n):
        for e_idx in edge_index_by_node[i]:
            S[i] += z[e_idx]
    if return_duals:
        return z, S, lam
    return z, S


def project_capped_simplex(t: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Euclidean projection of t onto {x>=0, sum(x)<=cap} -- pure local
    computation given a node's own raw target values for its edges."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return t
    clipped = np.maximum(t, 0.0)
    if clipped.sum() <= cap:
        return clipped
    lo, hi = 0.0, float(np.max(t))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.maximum(t - mid, 0.0).sum() > cap:
            lo = mid
        else:
            hi = mid
    return np.maximum(t - hi, 0.0)


def decentralized_graph_projection(edges, n, M_ii, M_ij_by_edge, w_init,
                                    neighbors, edge_index_by_node,
                                    num_sweeps, ledger: CommLedger, omega: float = 1.0,
                                    rescale_passes: int = 8):
    """
    rescale_passes: maximum number of final local rescale+re-symmetrize
    passes (see the comment below); 0 returns the raw sweep output.
    omega: Jacobi damping/relaxation factor. Plain Jacobi (omega=1) is only
    guaranteed contractive when 2D-P is positive definite (D=diag(P)); for
    P=2A^TA+4I this requires the LINE GRAPH's degree to stay bounded --
    found empirically to fail on a denser RGG (n=30, m=207, average degree
    ~14): the iteration stagnates around 0.9 relative error instead of
    converging. Damping (omega<1) shrinks the effective iteration matrix's
    spectral radius and restores contraction at the cost of more sweeps.

    edges: list of (i,j) with i<j (matches EdgeGraphProjection.edges order).
    M_ii: (n,) array, node i's own diagonal target -- already local.
    M_ij_by_edge: (m,) array, aligned with `edges` -- known to both endpoints.
    w_init: (m,) warm start (e.g. previous outer iteration's w).
    neighbors: list of length n, neighbors[i] = sorted list of node indices.
    edge_index_by_node: list of length n, edge_index_by_node[i][pos] = index
      into `edges`/`w` for the pos-th neighbor in neighbors[i] (handles the
      i<j edge-storage convention transparently).
    Returns (w, S) after `num_sweeps` Jacobi sweeps.
    """
    m = len(edges)
    w = w_init.copy()
    S = np.zeros(n)
    for i in range(n):
        for e_idx in edge_index_by_node[i]:
            S[i] += w[e_idx]

    for _ in range(num_sweeps):
        # Round 1: exchange (M_ii, S_i) with every neighbor -- 1 round, 2 floats/message.
        ledger.log_round(sum(len(nb) for nb in neighbors), 2)
        w_raw = np.empty(m)
        for e_idx, (i, j) in enumerate(edges):
            S_i_other = S[i] - w[e_idx]
            S_j_other = S[j] - w[e_idx]
            w_unconstrained = (0.25 * (1.0 - M_ii[i]) + 0.25 * (1.0 - M_ii[j]) + 0.5 * M_ij_by_edge[e_idx]
                               - 0.25 * S_i_other - 0.25 * S_j_other)
            w_raw[e_idx] = (1.0 - omega) * w[e_idx] + omega * w_unconstrained

        # Local step: each node projects its OWN row (no communication).
        w_from_i = np.empty(m)  # node i's (smaller-index endpoint) projected view
        w_from_j = np.empty(m)  # node j's (larger-index endpoint) projected view
        for i in range(n):
            idxs = edge_index_by_node[i]
            if not idxs:
                continue
            raw_row = w_raw[idxs]
            proj_row = project_capped_simplex(raw_row, cap=1.0)
            for pos, e_idx in enumerate(idxs):
                ii, jj = edges[e_idx]
                if ii == i:
                    w_from_i[e_idx] = proj_row[pos]
                else:
                    w_from_j[e_idx] = proj_row[pos]

        # Round 2: exchange each endpoint's projected value for the shared edge -- 1 round.
        ledger.log_round(2 * m, 1)
        w = 0.5 * (w_from_i + w_from_j)

        S = np.zeros(n)
        for i in range(n):
            for e_idx in edge_index_by_node[i]:
                S[i] += w[e_idx]

    # Final safety rescale: the symmetrization step (w = average of both
    # endpoints projections) can push a node's AGGREGATE row sum above 1
    # even though each endpoint's OWN projected row individually satisfied
    # the constraint -- found empirically running the full ADMM loop
    # (decentralized_fmmc_admm.py), where B ended up with entries as
    # negative as -0.14 (B_ii=1-S_i going negative for S_i>1), despite this
    # isolated test never showing it for simpler M. Any node whose row sum
    # still exceeds 1 after the sweeps uniformly shrinks its OWN edges by
    # 1/S_i -- a purely local, always-feasible correction, avoiding
    # w_ii<0 -- then a final exchange re-symmetrizes the (slightly)
    # asymmetric result. Each pass costs 2 rounds (up to 2*rescale_passes
    # in total), regardless of how many sweeps ran.
    # Iterated (not single-shot): rescaling both endpoints then
    # re-symmetrizing can still leave the averaged row sum slightly above 1.
    # Each pass roughly halves the largest violation (measured on the ADMM
    # target of experiment_heuristic_bias.py: 5.0e-2 without correction,
    # 1.9e-4 after 8 passes), so a few passes make it small but not zero.
    for _ in range(rescale_passes):
        scale = np.ones(n)
        for i in range(n):
            if S[i] > 1.0:
                scale[i] = 1.0 / S[i]
        ledger.log_round(sum(len(nb) for nb in neighbors), 1)
        w_from_i = np.empty(m)
        w_from_j = np.empty(m)
        for e_idx, (i, j) in enumerate(edges):
            w_from_i[e_idx] = w[e_idx] * scale[i]
            w_from_j[e_idx] = w[e_idx] * scale[j]
        ledger.log_round(2 * m, 1)
        w = 0.5 * (w_from_i + w_from_j)
        S = np.zeros(n)
        for i in range(n):
            for e_idx in edge_index_by_node[i]:
                S[i] += w[e_idx]
        if S.max() <= 1.0 + 1e-10:
            break

    return w, S


def adaptive_omega(adj: np.ndarray, safety: float = 4.0) -> float:
    """Degree-adaptive Jacobi damping. Plain Jacobi (omega=1) is only
    guaranteed contractive when 2D-P is positive definite (D=diag(P)=8I);
    P's off-diagonal mass at a degree-d node scales with d, so damping by
    ~1/max_degree keeps the damped iteration contractive regardless of
    graph density -- found empirically: max_degree=24 needed omega<=0.3
    (omega*max_degree<=7.2) to converge; `safety` sets the margin below
    that threshold (default 4.0, i.e. omega*max_degree<=4 -- comfortably
    inside the empirically-found bound). Computing the graph-wide
    max_degree decentrally needs a max-consensus reduction (flooding,
    O(diameter) rounds) -- but only ONCE at setup, not per outer ADMM
    iteration, so its cost is negligible against the many repeated
    B-projection/spectral rounds that follow."""
    deg = adj.sum(axis=1)
    max_deg = float(deg.max())
    return min(1.0, safety / max(max_deg, 1.0))


def build_topology(edges, n):
    neighbors = [[] for _ in range(n)]
    edge_index_by_node = [[] for _ in range(n)]
    for e_idx, (i, j) in enumerate(edges):
        neighbors[i].append(j)
        neighbors[j].append(i)
        edge_index_by_node[i].append(e_idx)
        edge_index_by_node[j].append(e_idx)
    return neighbors, edge_index_by_node


if __name__ == "__main__":
    import networkx as nx
    from fmmc_exact_admm import EdgeGraphProjection

    def check(gname, adj, num_sweeps_list, omega=1.0):
        n = adj.shape[0]
        projector = EdgeGraphProjection(adj)
        neighbors, edge_index_by_node = build_topology(projector.edges, n)
        rng = np.random.default_rng(0)
        M = rng.standard_normal((n, n)) * 0.3
        M = 0.5 * (M + M.T) + np.eye(n) * 0.5
        M_ii = np.diag(M).copy()
        M_ij_by_edge = np.array([M[i, j] for (i, j) in projector.edges])

        w_ref, B_ref, feas, success = projector.project(M)
        deg = adj.sum(axis=1)

        print(f"\n{gname} (n={n}, m={projector.m}, max degree={int(deg.max())}, omega={omega})")
        for K in num_sweeps_list:
            ledger = CommLedger()
            w_init = np.zeros(projector.m)
            w_dec, S = decentralized_graph_projection(
                projector.edges, n, M_ii, M_ij_by_edge, w_init,
                neighbors, edge_index_by_node, K, ledger, omega=omega)
            err = np.linalg.norm(w_dec - w_ref) / max(np.linalg.norm(w_ref), 1e-12)
            feas_viol = max(0.0, S.max() - 1.0 - 1e-6)
            print(f"  sweeps={K:4d} | rel. ||w_dec-w_ref||={err:.4e} | "
                  f"max row-sum viol={feas_viol:.2e} | rounds={ledger.rounds}")

    from experiment_backend_comparison import build_graphs

    print("=" * 70)
    print("Validating adaptive_omega on the paper's actual headline graphs")
    print("=" * 70)
    for gname, adj in build_graphs().items():
        omega = adaptive_omega(adj)
        check(f"{gname} (adaptive omega={omega:.4f})", adj, [10, 50, 100, 200, 500], omega=omega)
