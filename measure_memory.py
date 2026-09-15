"""
Peak-memory measurement for the Section 15 results table ("Memory"
column). Each method is run in its OWN subprocess (via
`/usr/bin/time -v`) rather than sequentially in one process, since peak
RSS only ever grows within a process -- running multiple methods
back-to-back in-process would just report a monotonically increasing
combined peak, not per-method figures.

DES-FMMC (Chebyshev)/(Krylov) reuse the Chebyshev-ADMM/Krylov-ADMM
measurements directly: both pairs run the identical computation (only
the Section 15 table's row LABELS differ, since DES-FMMC additionally
reports the communication-cost *model*'s output on top of the same
centralized run -- see the note already in Section 15).
"""
import os
import subprocess
import re
import sys

GRAPH_SETUP = """
import numpy as np, networkx as nx
n = 50
G = nx.random_geometric_graph(n, radius=np.sqrt(8.0/n), seed=42)
while not nx.is_connected(G):
    G = nx.random_geometric_graph(n, radius=np.sqrt(8.0/n), seed=np.random.randint(1_000_000))
adj = nx.to_numpy_array(G, dtype=float)
"""

METHOD_SNIPPETS = {
    "MOSEK-substitute (CLARABEL)": GRAPH_SETUP + """
from centralized_reference import solve_fmmc_cvxpy
solve_fmmc_cvxpy(adj)
""",
    "Exact-EVD ADMM": GRAPH_SETUP + """
from fmmc_exact_admm import fmmc_admm_exact
fmmc_admm_exact(adj, rho=0.1, num_iter=300, verbose=False)
""",
    "Krylov-ADMM": GRAPH_SETUP + """
from des_fmmc_admm import des_fmmc_admm
des_fmmc_admm(adj, backend="lanczos", rho=0.1, num_iter=300, verbose=False)
""",
    "Chebyshev-ADMM": GRAPH_SETUP + """
from des_fmmc_admm import des_fmmc_admm
des_fmmc_admm(adj, backend="chebyshev", rho=0.1, num_iter=300, verbose=False)
""",
}


def measure_peak_rss_kb(snippet: str) -> float:
    proc = subprocess.run(
        ["/usr/bin/time", "-v", sys.executable, "-c", snippet],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True,
    )
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", proc.stderr)
    if not m:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("Could not parse peak RSS from /usr/bin/time output.")
    return float(m.group(1))


def import_only(snippet: str) -> str:
    """The same snippet with its final (solve) line removed: graph setup and
    imports only, so that peak RSS attributable to the solve itself can be
    separated from interpreter and library-import overhead."""
    lines = snippet.strip().splitlines()
    return "\n".join(lines[:-1]) + "\n"


if __name__ == "__main__":
    results = {}
    print(f"{'method':>28} | {'peak RSS (MB)':>14} | {'imports only':>12} | {'solve increment':>15}")
    for name, snippet in METHOD_SNIPPETS.items():
        mb = measure_peak_rss_kb(snippet) / 1024.0
        base = measure_peak_rss_kb(import_only(snippet)) / 1024.0
        results[name] = {"peak_mb": mb, "imports_only_mb": base, "increment_mb": mb - base}
        print(f"{name:>28} | {mb:>14.1f} | {base:>12.1f} | {mb - base:>15.1f}", flush=True)
