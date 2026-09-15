# decentralized-fmmc-admm

[![tests](https://github.com/raghuram87/decentralized-fmmc-admm/actions/workflows/tests.yml/badge.svg)](https://github.com/raghuram87/decentralized-fmmc-admm/actions/workflows/tests.yml)

Python implementation and reproduction scripts for two related results on decentralized
design of consensus weights:

1. **Graph-constrained doubly stochastic projection.** An exact global-consensus ADMM for
   the Euclidean projection of a symmetric matrix onto the symmetric, nonnegative, doubly
   stochastic matrices supported on a graph. It includes the node-local scalar-bisection
   solver and the project-then-average baseline, which converges to a biased fixed point.
   Paper: *A Provably Exact Distributed ADMM Projection onto Graph-Constrained Doubly
   Stochastic Matrices*.
2. **Decentralized fastest mixing Markov chain (FMMC).** An edge-local ADMM for FMMC in
   which the spectral update uses a matrix-free, Chebyshev-filtered extreme-spectrum
   estimate and communication is measured in gossip rounds. Includes a
   Rokade–Kalaimani baseline and an independent SDP reference. Paper: *Communication-Aware
   Decentralized FMMC Optimization via Edge-Local ADMM and Chebyshev Polynomial Filtering*.

Section numbers in docstrings refer to the extended technical report of the second paper.

## Installation

Python 3.11 or later.

```bash
pip install -r requirements.txt
pytest test_golden.py -q        # about 15 s
```

All scripts are run from the repository root, e.g. `python experiment_consensus_projection.py`.
Random seeds are fixed throughout.

## Graph-constrained doubly stochastic projection

| Result | Script |
|---|---|
| Heuristic, consensus ADMM, and node-local solver | `decentralized_graph_projection.py` |
| Centralized reference projection (OSQP) | `fmmc_exact_admm.py` (`EdgeGraphProjection`) |
| Persistent bias on an ADMM target (error constant over 100–3000 sweeps; variant without rescaling; Metropolis–Hastings target) | `experiment_heuristic_bias.py` |
| Five-node cycle, per-edge values | `experiment_five_node_cycle.py` |
| Four-graph validation table (error and rounds at 800 sweeps) | `experiment_consensus_projection.py` |
| Convergence-curve figure | `make_lcss_figure.py` |
| Sensitivity to the consensus penalty ρ_c | `experiment_rho_sensitivity.py` |
| Dependence on graph size (cycles, hypercubes, random geometric graphs up to n = 800) | `experiment_projection_scaling.py` |
| Warm starts along an outer ADMM trajectory (and per-target figure) | `experiment_warm_start.py`, `make_warm_start_figure.py` |
| Solver for heterogeneous curvature, coefficients, and box bounds (500 trials vs. SLSQP) | `verify_node_qp_generalization.py` |
| Counterexample with two independent couplings, w* = (5/9, 1/9) | `verify_node_qp_two_constraint_counterexample.py` |

## Decentralized FMMC

| Component or result | Script |
|---|---|
| ADMM with swappable spectral backend (full EVD / Lanczos / Chebyshev) | `des_fmmc_admm.py` |
| Spectral-update backends and the active-set certificate | `spectral_oracle.py` |
| Locality-enforcing simulator with measured gossip rounds | `decentralized_chebyshev.py` |
| Analytical communication model | `communication_model.py` |
| End-to-end decentralized ADMM | `decentralized_fmmc_admm.py` |
| Exact centralized edge-weight ADMM | `fmmc_exact_admm.py` |
| Independent SDP reference (CVXPY + CLARABEL) | `centralized_reference.py` |
| Rokade–Kalaimani baseline | `baseline_rokade_kalaimani.py` |
| Backend comparison on four graphs | `experiment_backend_comparison.py` |
| Scale-up (fixed budget / converged) | `experiment_scale_up.py`, `experiment_scale_up_converged.py` |
| Validation against a converged SDP optimum | `experiment_true_optimum_validation.py` |
| Objective-gap campaign and dual certificates | `delta_obj_campaign.py`, `duality_certificate_campaign.py` |
| Block-size sweep | `experiment_block_size_sweep.py` |
| Adaptive gossip matrix ablation | `experiment_adaptive_communication.py` |
| Convergence diagnostics figure (n = 50) | `make_convergence_figure.py` |
| Peak memory per method, with import-only baselines (requires GNU `/usr/bin/time`) | `measure_memory.py` |
| Topological scope of the suboptimal fixed point on long cycles | `experiment_bipartite_scope.py` |
| Spectral-update sanity checks | `verify_oracle.py` |

`experiment_true_optimum_validation.py` caches the slow CLARABEL reference solves in
`./cache` (override with the `FMMC_CACHE_DIR` environment variable).

## Tests

`test_golden.py` holds the regression tests for the reported results. Its reference values
were recorded with NumPy/SciPy linked against MKL. Builds with other BLAS/LAPACK
libraries, such as the OpenBLAS in PyPI wheels, follow slightly different floating-point
trajectories: the final SLEM differs by at most about 1e-6 and stopping times by about one
iteration. The tests use tolerances that absorb these differences but still catch any change
visible at the six significant digits reported in the papers.

## Citation

If you use this code, please cite the software (see `CITATION.cff`) and the corresponding paper.

## License

MIT; see `LICENSE`.
