"""
Does solve_node_qp's two-case closed-form-via-scalar-bisection construction
generalize beyond the FMMC-specific instance (uniform edge curvature,
a=all-ones sum functional, [0,inf) box)?

Algebraic claim to test: minimize_w  sum_e (kappa_e/2)(w_e-r_e)^2
                                     + beta*(a^T w - c)^2
                         s.t.        lo_e <= w_e <= hi_e  (per-edge box)
                                     a^T w <= cap           (ONE linear cap)

KKT stationarity (ignoring box first): kappa_e*(w_e-r_e) + 2*beta*a_e*(a^T w - c) = 0
=> w_e = r_e - (2*beta*a_e/kappa_e)*psi,  psi := a^T w - c
With box clipping added, standard theory (separable quadratic + box + ONE
linear constraint) says w_e(psi) = clip(r_e - (2*beta*a_e/kappa_e)*psi, lo_e, hi_e)
is monotone in psi (since a_e/kappa_e same sign per e), so psi solves a
1-D monotone equation -> bisection. This is the classical "weighted
water-filling / continuous knapsack" structure -- should generalize to
ANY a_e, kappa_e>0, box. Testing it here against scipy before trusting it.

Also testing whether it breaks for TWO independent linear constraints per
node (not reducible to one scalar) -- expected to fail / not be expressible
this way.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

rng = np.random.default_rng(0)

def solve_general_single_constraint(r, kappa, a, beta, c, lo, hi, cap):
    """Generalized node QP: min sum kappa_e/2 (w_e-r_e)^2 + beta*(a.w-c)^2
    s.t. lo<=w<=hi, a.w<=cap.  Solve via bisection on multiplier psi
    (soft penalty, always active) then, if a.w>cap, on extra multiplier
    for the hard constraint."""
    r = np.asarray(r,float); kappa=np.asarray(kappa,float); a=np.asarray(a,float)
    lo=np.asarray(lo,float); hi=np.asarray(hi,float)

    def w_of_psi(psi, extra=0.0):
        # stationarity with soft penalty psi and (optional) hard-constraint multiplier `extra`
        return np.clip(r - (2*beta*a/kappa)*psi - extra*a/kappa, lo, hi)

    def F(psi):
        w = w_of_psi(psi)
        return psi - (a @ w - c)

    span = 10*(np.abs(r).max()+abs(c)+np.abs(hi-lo).max()+1)
    lo_s, hi_s = -span, span
    for _ in range(200):
        mid = 0.5*(lo_s+hi_s)
        if F(mid) < 0: lo_s = mid
        else: hi_s = mid
    psi1 = 0.5*(lo_s+hi_s)
    w1 = w_of_psi(psi1)
    if a @ w1 <= cap + 1e-9:
        return w1

    # hard constraint active: solve for `extra` at fixed psi=psi1 (soft penalty
    # remains satisfied approximately; re-derive: with hard constraint active,
    # stationarity for w_e picks up BOTH the soft term at its own consistent
    # psi AND a constant "extra" Lagrange multiplier for a.w=cap).
    # Full joint solve: two nested monotone equations.
    def G(extra):
        # inner solve for psi given extra
        lo2, hi2 = -span, span
        for _ in range(200):
            mid = 0.5*(lo2+hi2)
            w = w_of_psi(mid, extra)
            val = mid - (a @ w - c)
            if val < 0: lo2 = mid
            else: hi2 = mid
        psi = 0.5*(lo2+hi2)
        w = w_of_psi(psi, extra)
        return a @ w - cap, w

    lo_e, hi_e_ = 0.0, span
    for _ in range(200):
        mid = 0.5*(lo_e+hi_e_)
        val, w = G(mid)
        if val > 0: lo_e = mid
        else: hi_e_ = mid
    _, w_final = G(0.5*(lo_e+hi_e_))
    return w_final


def scipy_ref(r, kappa, a, beta, c, lo, hi, cap):
    k = len(r)
    def obj(w):
        return np.sum(0.5*kappa*(w-r)**2) + beta*(a@w-c)**2
    def grad(w):
        return kappa*(w-r) + 2*beta*a*(a@w-c)
    cons = LinearConstraint(a.reshape(1,-1), -np.inf, cap)
    bounds = Bounds(lo, hi)
    x0 = np.clip(r, lo, hi)
    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=[cons],
                    method='SLSQP', options={'maxiter':500,'ftol':1e-14})
    return res.x

max_err = 0.0
n_trials = 500
for trial in range(n_trials):
    k = rng.integers(2, 7)
    r = rng.uniform(-2, 2, k)
    kappa = rng.uniform(0.3, 5.0, k)       # heterogeneous curvature
    a = rng.uniform(0.2, 3.0, k)           # heterogeneous linear-functional coeffs (all positive)
    beta = rng.uniform(0.1, 3.0)
    c = rng.uniform(-1, 1)
    lo = rng.uniform(-1, 0, k)             # general box, not just [0, inf)
    hi = lo + rng.uniform(0.5, 3.0, k)
    cap = rng.uniform(0.0, 2.0)

    w_mine = solve_general_single_constraint(r, kappa, a, beta, c, lo, hi, cap)
    w_ref = scipy_ref(r, kappa, a, beta, c, lo, hi, cap)
    err = np.max(np.abs(w_mine - w_ref))
    max_err = max(max_err, err)
    if err > 1e-4:
        print(f"trial {trial}: err={err:.3e}  mine={w_mine}  ref={w_ref}")

print(f"\nSingle-constraint generalization (heterogeneous kappa_e, a_e, general box): "
      f"max err over {n_trials} trials = {max_err:.3e}")
