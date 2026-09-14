"""
Does the single-scalar-bisection reduction survive TWO independent
node-local linear constraints (not reducible to one scalar)? Expectation:
no -- each additional independent linear coupling needs its own
multiplier, and the problem becomes a joint fixed point in >=2 dimensions,
not a single monotone scalar equation. Testing this directly: try to
solve a 2-constraint node QP using the SAME single-multiplier machinery
(collapsing the two constraints' identity terms into one scalar via a
naive combination) and show it disagrees with the true (2-multiplier)
solution -- i.e. the construction's scope stops at ONE node-local
linear constraint plus box bounds, not "any number."
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

rng = np.random.default_rng(1)

def scipy_ref_2con(r, kappa, a1, beta1, c1, a2, beta2, c2, lo, hi):
    def obj(w):
        return (np.sum(0.5*kappa*(w-r)**2)
                + beta1*(a1@w-c1)**2 + beta2*(a2@w-c2)**2)
    def grad(w):
        return kappa*(w-r) + 2*beta1*a1*(a1@w-c1) + 2*beta2*a2*(a2@w-c2)
    bounds = Bounds(lo, hi)
    x0 = np.clip(r, lo, hi)
    res = minimize(obj, x0, jac=grad, bounds=bounds, method='L-BFGS-B',
                    options={'maxiter':2000,'ftol':1e-16,'gtol':1e-12})
    return res.x

def naive_single_scalar_attempt(r, kappa, a1, beta1, c1, a2, beta2, c2, lo, hi):
    """Wrong attempt: treat the combined pull as if it reduced to ONE
    scalar psi via a1 alone (ignoring a2's independent direction) --
    this is what you'd get if you incorrectly assumed the single-multiplier
    machinery still applied."""
    a_eff = a1  # pretend only constraint 1 exists structurally
    beta_eff = beta1
    c_eff = c1
    def w_of_psi(psi):
        return np.clip(r - (2*beta_eff*a_eff/kappa)*psi, lo, hi)
    def F(psi):
        w = w_of_psi(psi)
        return psi - (a_eff @ w - c_eff)
    span = 10*(np.abs(r).max()+abs(c_eff)+np.abs(hi-lo).max()+1)
    lo_s, hi_s = -span, span
    for _ in range(200):
        mid = 0.5*(lo_s+hi_s)
        if F(mid) < 0: lo_s = mid
        else: hi_s = mid
    return w_of_psi(0.5*(lo_s+hi_s))

max_err = 0.0
worst = None
for trial in range(200):
    k = rng.integers(3, 7)
    r = rng.uniform(-2, 2, k)
    kappa = rng.uniform(0.5, 3.0, k)
    a1 = rng.uniform(0.2, 2.0, k)
    a2 = rng.uniform(0.2, 2.0, k)  # independent direction, not parallel to a1
    beta1 = rng.uniform(0.5, 2.0); beta2 = rng.uniform(0.5, 2.0)
    c1 = rng.uniform(-1, 1); c2 = rng.uniform(-1, 1)
    lo = rng.uniform(-1, 0, k); hi = lo + rng.uniform(0.5, 3.0, k)

    w_true = scipy_ref_2con(r, kappa, a1, beta1, c1, a2, beta2, c2, lo, hi)
    w_naive = naive_single_scalar_attempt(r, kappa, a1, beta1, c1, a2, beta2, c2, lo, hi)
    err = np.max(np.abs(w_true - w_naive))
    if err > max_err:
        max_err = err; worst = trial

print(f"Two-independent-constraint case: naive single-scalar reduction vs. true "
      f"2-multiplier optimum, max err over 200 trials = {max_err:.3e} "
      f"(large error expected -- confirms single-scalar machinery does NOT "
      f"cover >1 independent node-local linear constraint)")

# Explicit two-edge instance reported in the letter: r=(1,1), kappa=(1,1),
# a1=(1,1), a2=(1,-1), beta1=beta2=2, c1=c2=1/2, inactive box. The joint
# stationarity conditions give w* = (5/9, 1/9); the single-scalar reduction
# with the first coupling gives psi = 1/6 and w = (1/3, 1/3).
r = np.array([1.0, 1.0]); kappa = np.array([1.0, 1.0])
a1 = np.array([1.0, 1.0]); a2 = np.array([1.0, -1.0])
lo = np.full(2, -2.0); hi = np.full(2, 2.0)
w_true = scipy_ref_2con(r, kappa, a1, 2.0, 0.5, a2, 2.0, 0.5, lo, hi)
w_naive = naive_single_scalar_attempt(r, kappa, a1, 2.0, 0.5, a2, 2.0, 0.5, lo, hi)
assert np.allclose(w_true, [5/9, 1/9], atol=1e-6), w_true
assert np.allclose(w_naive, [1/3, 1/3], atol=1e-9), w_naive
print(f"Explicit instance: w* = {w_true} (5/9, 1/9); single-scalar = {w_naive} (1/3, 1/3)")
