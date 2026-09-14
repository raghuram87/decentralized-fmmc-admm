"""
Decentralized communication accounting for the two approximate oracle
backends. Applying the graph
term B@x is always 1-hop/free; every global reduction (mean for J, inner
products for U) costs R_t gossip rounds, where R_t is governed by the
gossip matrix's own spectral gap -- the same quantity the whole paper is
about.

Reports q_t (operator applications) and c0 (reductions per application)
SEPARATELY for both backends rather than only their product.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def gossip_rounds_for_tolerance(lambda2_gossip: float, eps_gossip: float) -> int:
    """R_t = ceil(log(1/eps) / log(1/lambda2)) -- linear convergence of
    repeated application of a symmetric doubly-stochastic gossip matrix.

    On BIPARTITE graphs (e.g. the hypercube), the unmodified
    Metropolis-Hastings matrix has an eigenvalue at exactly -1, so
    lambda2_gossip = 1 mathematically -- naive gossip never converges at
    all (a real, known pathology, not a numerical artifact). Floating
    point pushes the computed value to a hair ABOVE 1
    (e.g. 1.0000000000000009 was observed for hypercube_n64), which would
    otherwise send log(1/lambda2) negative and blow up the round count
    into nonsense. Clip defensively and report a large-but-finite round
    count as the "gossip is effectively broken here" signal.
    """
    lambda2_gossip = min(lambda2_gossip, 1.0 - 1e-6)
    if lambda2_gossip <= 0:
        return 1
    return int(np.ceil(np.log(1.0 / eps_gossip) / np.log(1.0 / lambda2_gossip)))


@dataclass
class CommunicationCost:
    backend: str
    q_t: int
    c0_per_application: float
    reorthog_tax_total: float
    R_t: int
    total_reduction_groups: float
    total_rounds: float
    total_floats: float


def communication_cost_chebyshev(q_t: int, k_active: int, num_edges: int,
                                  lambda2_gossip: float, eps_gossip: float) -> CommunicationCost:
    """
    Every V-application needs: 1 reduction-group for the mean (J term) +
    k_active reduction-groups for the low-rank U correction. No
    orthogonalization tax -- the filter's 3-term recurrence has fixed
    O(1) memory, it never builds/maintains an explicit orthogonal basis.
    """
    c0 = 1 + k_active
    R_t = gossip_rounds_for_tolerance(lambda2_gossip, eps_gossip)
    total_groups = q_t * c0
    total_rounds = total_groups * R_t
    total_floats = total_rounds * 2 * num_edges
    return CommunicationCost("chebyshev", q_t, c0, 0.0, R_t, total_groups, total_rounds, total_floats)


def communication_cost_lanczos(q_t: int, k_active: int, num_edges: int,
                                lambda2_gossip: float, eps_gossip: float,
                                reorthogonalize: bool = True) -> CommunicationCost:
    """
    Same per-application O(1+k) cost as Chebyshev (both need the mean and
    U-correction reductions to apply V at all), PLUS Lanczos-specific
    overhead: at minimum 1 extra reduction-group per step for alpha_j
    (q_j^T V q_j); with full re-orthogonalization (needed in practice --
    the technical report notes gossip returns inner products
    inexactly, which corrupts orthogonality, Sec. 5.1/7.2), step j needs
    ~j additional reduction-groups (reorthogonalizing against every prior
    Krylov vector), giving O(q_t^2) total reorthogonalization cost versus
    Chebyshev's O(q_t) -- the qualitative, not just constant-factor,
    advantage that the experiments are designed to verify.
    """
    c0_base = 1 + k_active
    if reorthogonalize:
        reorthog_tax = q_t * (q_t + 1) / 2.0
    else:
        reorthog_tax = float(q_t)
    R_t = gossip_rounds_for_tolerance(lambda2_gossip, eps_gossip)
    total_groups = q_t * c0_base + reorthog_tax
    total_rounds = total_groups * R_t
    total_floats = total_rounds * 2 * num_edges
    return CommunicationCost("lanczos", q_t, c0_base, reorthog_tax, R_t, total_groups, total_rounds, total_floats)
