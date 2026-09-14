"""
Locality-enforcing decentralized simulator for the Chebyshev spectral
oracle. It complements communication_model.py's analytical formula with
MEASURED rounds from a simulation in which a node structurally cannot read
another node's data except through an explicit gossip round.

spectral_oracle.py::oracle_chebyshev is algorithmically local-friendly
(matvecs only touch graph edges) but runs in one process with instant
access to global numpy arrays -- no round count reported anywhere in this
project is otherwise MEASURED rather than modeled. This module provides
measured counts for the Chebyshev backend (Lanczos stays modeled, since
decentralizing ARPACK would require reimplementing Lanczos).

Locality is enforced structurally: the weight
operator is masked to the graph's own sparsity pattern (a node's row has
nonzero entries ONLY at its 1-hop neighbors), so a sparse matvec against
it is algebraically IDENTICAL to n independent local combinations -- not
an approximation of locality, an enforcement of it via the data structure
itself.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import networkx as nx

from spectral_oracle import (
    solve_s_from_eigs, certified_with_s_stability, gershgorin_bound,
    _chebyshev_degree_for_tau, OracleResult,
)
from fmmc_exact_admm import metropolis_hastings_weights


def mh_to_B(adj_matrix):
    """metropolis_hastings_weights returns only the off-diagonal edge
    weights w_ij (feasible w>=0, sum_j w_ij<=1); B=I-L(w) needs the
    diagonal derived as B_ii = 1 - sum_j w_ij (fmmc_exact_admm.py::B_from_w's
    convention) -- using the raw w matrix directly as if it were already
    row-stochastic silently breaks gossip averaging (rows don't sum to 1)."""
    w = metropolis_hastings_weights(adj_matrix)
    return np.eye(adj_matrix.shape[0]) - np.diag(w.sum(axis=1)) + w


# =====================================================================
# Locality-enforcing primitives
# =====================================================================
class CommLedger:
    """A 'round' is one synchronous exchange of messages between 1-hop
    neighbors -- tracks rounds/messages/floats for the local simulator."""

    def __init__(self):
        self.rounds = 0
        self.messages = 0
        self.floats = 0

    def log_round(self, num_directed_messages, floats_per_message):
        self.rounds += 1
        self.messages += num_directed_messages
        self.floats += num_directed_messages * floats_per_message


def build_sparse_weight_operator(adj_matrix, weight_matrix):
    """Graph-masked sparse weight operator: W[i,j] != 0 only if j is a
    1-hop neighbor of i (or j==i). A sparse matvec against it is
    algebraically identical to n independent local combinations of a
    node's own row with values received from its neighbors."""
    n = adj_matrix.shape[0]
    mask = (adj_matrix > 0) | np.eye(n, dtype=bool)
    W_sparse = sp.csr_matrix(np.where(mask, weight_matrix, 0.0))
    directed_messages = W_sparse.nnz - n
    return W_sparse, directed_messages


def local_sparse_matvec(W_sparse, directed_messages, x, ledger):
    """One communication round: (Wx)_i computed using ONLY node i's own
    weight row and values RECEIVED from its 1-hop neighbors -- enforced
    by W_sparse's graph-masked sparsity pattern, not just by convention."""
    x = np.asarray(x, dtype=float)
    is_1d = x.ndim == 1
    if is_1d:
        x = x[:, None]
    y = W_sparse @ x
    ledger.log_round(directed_messages, x.shape[1])
    return y[:, 0] if is_1d else y


def gossip_average(x, W_sparse, directed_messages, ledger, eps=1e-4, max_rounds=3000):
    """Decentralized computation of the global column-wise mean via
    repeated 1-hop Metropolis-Hastings gossip rounds -- every reduction
    the Chebyshev filter needs (the J-mean, the U-correction inner
    products) is really one of these, not a free operation."""
    x = np.asarray(x, dtype=float)
    true_mean = np.mean(x, axis=0)
    y = x.copy()
    for r in range(1, max_rounds + 1):
        y = local_sparse_matvec(W_sparse, directed_messages, y, ledger)
        if np.max(np.abs(y - true_mean)) < eps:
            return y, r
    return y, max_rounds


# =====================================================================
# Locality-enforced V-matvec (the same operator spectral_oracle.py's
# _make_V_matvec computes centrally, here built only from local rounds)
# =====================================================================
def local_matvec_V(B_sparse, B_messages, gossip_sparse, gossip_messages,
                    X, Q_ext, delta_evals, graph_ledger, gossip_ledger, gossip_eps):
    """
    (V @ X)_i = (B - J - Q_ext diag(delta) Q_ext^T) @ X, X is (n, k).
      - B @ X: 1 round, exactly 1-hop.
      - mean(X) and Q_ext^T @ X: global reductions, batched into ONE
        joint gossip run (k mean-channels + k*k cross-channels), since
        gossip round count depends on the network's spectral gap, not
        channel count.
    """
    n, k = X.shape
    k_active = Q_ext.shape[1]
    BX = local_sparse_matvec(B_sparse, B_messages, X, graph_ledger)

    # cross_channels[i,c,j] = X[i,c]*Q_ext[i,j] -- k_active can differ from
    # k (X's own block size), e.g. once U carries forward a k_active-column
    # correction from a PRIOR oracle call while the CURRENT filter block has
    # a different size. Reshaping with `k` instead of `k_active` here is dimensionally silent-wrong
    # whenever k_active != k -- never caught by the original single-call
    # validation, since that test always had Q_ext empty (k_active=0).
    cross_channels = np.einsum('ic,ij->icj', X, Q_ext).reshape(n, k * k_active) if k_active else np.zeros((n, 0))
    joint = np.concatenate([X, cross_channels], axis=1)

    reduced, gossip_rounds = gossip_average(joint, gossip_sparse, gossip_messages,
                                             gossip_ledger, eps=gossip_eps)
    mean_X = reduced[:, :k]
    if k_active:
        # reduced[:,k:] holds mean_i(X[i,c]*Q_ext[i,j]) = (1/n)(X^T@Q_ext)[c,j];
        # scaling by n and transposing gives Q_ext^T@X, shape (k_active,k) --
        # matching des_fmmc_admm.py::_make_V_matvec's `Q_ext.T @ x` exactly.
        XtQ = (reduced[0, k:].reshape(k, k_active)) * n  # (k, k_active) = X^T@Q_ext
        QtX = XtQ.T  # (k_active, k) = Q_ext^T@X
        Q_delta = Q_ext * delta_evals
        UX = Q_delta @ QtX
    else:
        UX = 0.0

    return BX - mean_X - UX, gossip_rounds


def local_apply_chebyshev_filter(B_sparse, B_messages, gossip_sparse, gossip_messages,
                                  X0, tau, Lambda2, m, Q_ext, delta_evals,
                                  graph_ledger, gossip_ledger, gossip_eps):
    """Same 3-term recurrence as spectral_oracle.py::apply_chebyshev_filter,
    but every V-application goes through local_matvec_V (measured rounds)
    instead of a dense/instant matvec."""
    def W(X):
        VX, _ = local_matvec_V(B_sparse, B_messages, gossip_sparse, gossip_messages,
                                X, Q_ext, delta_evals, graph_ledger, gossip_ledger, gossip_eps)
        V2X, _ = local_matvec_V(B_sparse, B_messages, gossip_sparse, gossip_messages,
                                 VX, Q_ext, delta_evals, graph_ledger, gossip_ledger, gossip_eps)
        return (2.0 * V2X - tau * X) / tau

    T_prev = X0
    T_curr = W(X0)
    q_t = 2
    for _ in range(2, m + 1):
        T_next = 2.0 * W(T_curr) - T_prev
        T_prev, T_curr = T_curr, T_next
        q_t += 2

    X_norm = (2.0 * Lambda2 - tau) / tau
    Tm_norm = np.cosh(m * np.arccosh(X_norm)) if X_norm > 1 else 1.0
    return T_curr / Tm_norm, q_t


# =====================================================================
# Decentralized oracle: same growth/certification logic as
# spectral_oracle.py::oracle_chebyshev, but every number is measured.
# =====================================================================
def decentralized_slem_estimate(adj_matrix, candidate_matrix, num_power_rounds,
                                 ledger, gossip_eps=1e-4, rng=None):
    """
    Decentralized estimate of SLEM(candidate_matrix), for the
    online adaptive-communication safeguard (Section 12/15.6): decide
    whether to ADOPT a new gossip/consensus matrix without secretly relying
    on a centralized eigendecomposition to make that decision.

    Design: deflate the trivial eigenvalue-1 direction ONCE via a gossip
    mean computed on a KNOWN-reliable matrix (lazy MH -- the existing safe
    fallback, never the candidate itself, which might be a poor/unknown
    mixer -- that is exactly what is being estimated). Then run PURE LOCAL
    power iteration (local_sparse_matvec only, no further gossip) on the
    CANDIDATE matrix for `num_power_rounds`, and estimate the decay rate
    via two norm-squared gossip reductions (start, end) -- not one per
    round, keeping the estimate cheap relative to the full spectral oracle.
    Validated (this file's __main__) against the true SLEM on this paper's
    four standard graphs: 0.6-4.5% relative error at 60 power rounds,
    3-15% at 10 -- accurate enough for a switch/no-switch decision, not a
    substitute for the oracle's own certified accuracy.
    """
    n = adj_matrix.shape[0]
    rng = rng or np.random.default_rng(0)

    lazy_mh = 0.5 * (np.eye(n) + mh_to_B(adj_matrix))
    safe_sparse, safe_messages = build_sparse_weight_operator(adj_matrix, lazy_mh)
    x0 = rng.standard_normal(n)
    mean0, _ = gossip_average(x0, safe_sparse, safe_messages, ledger, eps=gossip_eps)
    x = x0 - mean0

    cand_sparse, cand_messages = build_sparse_weight_operator(adj_matrix, candidate_matrix)

    def norm_sq(v):
        sq = (v ** 2)[:, None]
        reduced, _ = gossip_average(sq, safe_sparse, safe_messages, ledger, eps=gossip_eps)
        return float(reduced[0, 0] * n)

    start_norm_sq = norm_sq(x)
    for _ in range(num_power_rounds):
        x = local_sparse_matvec(cand_sparse, cand_messages, x, ledger)
    end_norm_sq = norm_sq(x)

    if start_norm_sq <= 0:
        return 0.0
    ratio = end_norm_sq / start_norm_sq
    if ratio <= 0:
        return 0.0
    return float(ratio ** (1.0 / (2 * num_power_rounds)))


def oracle_chebyshev_decentralized(adj_matrix, B_matrix, rho, Q_ext, delta_evals,
                                    s_estimate, delta_t=1e-6, block_size=4,
                                    block_growth=2, max_block=None,
                                    gossip_eps=1e-4, rng=None, gossip_matrix=None):
    n = adj_matrix.shape[0]
    rng = rng or np.random.default_rng(0)
    max_block = max_block or (n - 1)

    B_sparse, B_messages = build_sparse_weight_operator(adj_matrix, B_matrix)
    if gossip_matrix is None:
        # Default: lazy MH matrix for the gossip/reduction channel
        # specifically (NOT for the B-operator itself): plain MH has SLEM
        # -> 1 on poor mixers (cycle) and exactly -1 on bipartite graphs
        # (hypercube), the same pathology already documented in Section 17
        # and fixed the same way in
        # experiment_backend_comparison.py::graph_gossip_lambda2. Without
        # this, gossip_average never converges within max_rounds on those
        # graphs. Overriding via `gossip_matrix` lets a caller supply a
        # DIFFERENT (e.g. safeguard-selected, Section 12/15.6) consensus
        # matrix for the reduction channel while B_matrix still drives the
        # V operator -- the two roles were always conceptually separate
        # (Section 15.1), this parameter just exposes that separation.
        gossip_matrix = 0.5 * (np.eye(n) + mh_to_B(adj_matrix))
    gossip_sparse, gossip_messages = build_sparse_weight_operator(adj_matrix, gossip_matrix)

    row_abs_sum = np.abs(B_matrix).sum(axis=1) + 2.0  # B row-sum(|.|) + |J|,|U| row bound proxy
    Lambda2 = gershgorin_bound(row_abs_sum) ** 2
    tau = max(max(s_estimate, 1e-12) ** 2, 1e-9 * Lambda2)

    b = min(block_size, n - 1)
    s_prev = None
    graph_ledger, gossip_ledger = CommLedger(), CommLedger()

    while True:
        m = _chebyshev_degree_for_tau(tau, Lambda2, delta_t)
        X0 = rng.standard_normal((n, b))
        X0, _ = np.linalg.qr(X0)

        Xf, _ = local_apply_chebyshev_filter(
            B_sparse, B_messages, gossip_sparse, gossip_messages,
            X0, tau, Lambda2, m, Q_ext, delta_evals,
            graph_ledger, gossip_ledger, gossip_eps)

        Qb, _ = np.linalg.qr(Xf)
        # Rayleigh-Ritz: H = Qb^T V Qb, via a local matvec + one more
        # joint gossip reduction for the b x b inner-product matrix.
        VQb, _ = local_matvec_V(B_sparse, B_messages, gossip_sparse, gossip_messages,
                                 Qb, Q_ext, delta_evals, graph_ledger, gossip_ledger, gossip_eps)
        cross = np.einsum('ic,ij->icj', Qb, VQb).reshape(n, b * b)
        reduced, _ = gossip_average(cross, gossip_sparse, gossip_messages, gossip_ledger, eps=gossip_eps)
        H = (reduced[0].reshape(b, b) * n)
        H = 0.5 * (H + H.T)

        ritz_vals, W = np.linalg.eigh(H)
        Ritz_Q = Qb @ W
        # Residuals: reuse VQb @ W - Ritz_Q * ritz_vals (cheap local recombination).
        residuals = np.linalg.norm(VQb @ W - Ritz_Q * ritz_vals[None, :], axis=0)

        s_hat = solve_s_from_eigs(ritz_vals, rho)
        active = np.abs(ritz_vals) > s_hat
        certified = certified_with_s_stability(ritz_vals, residuals, s_hat, s_prev, delta_t, 1e-3)

        if certified or b >= max_block:
            return OracleResult(
                s_hat=s_hat, Q_active=Ritz_Q[:, active], Lambda_active=ritz_vals[active],
                k_active=int(active.sum()), q_t=graph_ledger.rounds, certified=certified,
                backend="chebyshev_decentralized",
                extra={"degree": m, "block_size": b,
                       "measured_graph_rounds": graph_ledger.rounds,
                       "measured_gossip_rounds": gossip_ledger.rounds,
                       "measured_graph_floats": graph_ledger.floats,
                       "measured_gossip_floats": gossip_ledger.floats},
            )
        s_prev = s_hat
        tau = max(max(s_hat, 1e-12) ** 2, 1e-9 * Lambda2)
        b = min(b * block_growth, max_block)


if __name__ == "__main__":
    from spectral_oracle import oracle_chebyshev
    from communication_model import communication_cost_chebyshev, gossip_rounds_for_tolerance

    graphs = {}
    n = 50
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=42)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(n, radius=np.sqrt(8.0 / n), seed=np.random.randint(1_000_000))
    graphs["random_geometric_n50"] = nx.to_numpy_array(G, dtype=float)
    graphs["cycle_n50"] = nx.to_numpy_array(nx.cycle_graph(50), dtype=float)
    H = nx.convert_node_labels_to_integers(nx.hypercube_graph(6))
    graphs["hypercube_n64"] = nx.to_numpy_array(H, dtype=float)
    graphs["petersen_n10"] = nx.to_numpy_array(nx.petersen_graph(), dtype=float)

    print(f"{'graph':>22} | {'s_hat(cent)':>11} | {'s_hat(decent)':>13} | {'k(cent)':>7} | {'k(decent)':>9} | "
          f"{'measured rounds':>16} | {'modeled rounds':>16} | {'ratio meas/model':>17}")
    for gname, adj in graphs.items():
        n_ = adj.shape[0]
        Wmh = mh_to_B(adj)
        rho = 1.0
        Q_ext0 = np.zeros((n_, 0))
        delta0 = np.zeros(0)
        num_edges = int(np.count_nonzero(np.triu(adj, k=1)))

        def matvec(x):
            x = np.asarray(x, dtype=float)
            Bx = Wmh @ x
            Jx = np.mean(x, axis=0, keepdims=True) if x.ndim > 1 else np.mean(x)
            return Bx - Jx

        r_cent = oracle_chebyshev(matvec, n_, rho, np.abs(Wmh).sum(axis=1) + 2.0,
                                   s_estimate=0.5, block_size=4)
        r_decent = oracle_chebyshev_decentralized(adj, Wmh, rho, Q_ext0, delta0,
                                                    s_estimate=0.5, block_size=4)

        # Modeled rounds: same gossip matrix (lazy MH), same eps, fed the
        # measured q_t/k_active -- isolates how well the ANALYTICAL R_t
        # formula alone tracks the round count a real gossip simulation needed.
        lazy_gossip = 0.5 * (np.eye(n_) + Wmh)
        lambda2_gossip = float(np.sort(np.abs(np.linalg.eigvalsh(lazy_gossip)))[-2])
        modeled = communication_cost_chebyshev(r_decent.q_t, r_decent.k_active, num_edges,
                                                 lambda2_gossip, eps_gossip=1e-4)

        measured = r_decent.extra['measured_gossip_rounds']
        print(f"{gname:>22} | {r_cent.s_hat:>11.6f} | {r_decent.s_hat:>13.6f} | "
              f"{r_cent.k_active:>7d} | {r_decent.k_active:>9d} | "
              f"{measured:>16,.0f} | "
              f"{modeled.total_rounds:>16,.0f} | "
              f"{measured / max(modeled.total_rounds, 1):>17.2f}")
