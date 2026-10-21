"""
Where the loss and the error of a trained jump network sit, from its values at the training nodes.

The network's node values come from a .vts written with the evaluation grid equal to the training grid
(gridstates.Nx_eval = solver.Nx_tr), so its "U" field is exactly what the rows saw in training. The rows
are rebuilt with the NumPy replica (tools/jump_replica3d.py), which tests/check_jump_rows.py showed equal
to the JAX rows (fixed scheme, Bias Slow, interface weights, alternating tetrahedra).

    python3 tools/jump_node_breakdown.py nodes_jump_lbfgs.vts [more.vts ...] --nx 32

For each file it prints
  * the node error, inside and outside the sphere;
  * the loss 0.5 mean(r^2) (should equal the JAX run's iteration-0 loss) and its split by row kind;
  * the error split into a smooth part (best degree-10 polynomial per side) and a rough remainder,
    with the loss each part causes on its own: if a small rough part carries the loss while the smooth
    part carries the error, the optimizer is spending itself on wiggles the error does not care about;
  * the loss and error after the best smooth correction, i.e. how far the smooth directions alone
    could take this network.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jump_replica3d as rp  # noqa: E402
from crop_vts import read_vts  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("files", nargs="+")
ap.add_argument("--nx", type=int, default=32)
ap.add_argument("--problem", choices=["sphere", "star"], default="sphere")
ap.add_argument("--ell", type=float, default=1.0e4)
ap.add_argument("--cut-floor", type=float, default=0.05)
a = ap.parse_args()

N = a.nx
pb = rp.make_problem("paper", radius=2 / 3) if a.problem == "sphere" else rp.make_star("jaxrobin")
s = rp.build(N, pb, True, alternate=True, bias="slow")
A, b = rp.scaled(s, "interface", ell=a.ell, cut_floor=a.cut_floor)
X, ue, inside = s["X"].reshape(-1, 3), s["u_exact"], s["side"] < 0
kind = np.where(s["wall"], "wall", np.where(s["AG"] > 0, "cut", np.where(s["A_ghost"] > 0, "ghost", "bulk")))
groups = [("bulk inside", (kind == "bulk") & inside), ("bulk outside", (kind == "bulk") & ~inside),
          ("ghost inside", (kind == "ghost") & inside), ("ghost outside", (kind == "ghost") & ~inside),
          ("cut", kind == "cut"), ("wall", kind == "wall")]
cols = rp.poly_columns(s)
AC = A @ cols
rms = lambda v: np.sqrt(np.mean(v ** 2))
loss = lambda r: 0.5 * np.mean(r ** 2)


def shares(r):
    tot = np.sum(r ** 2)
    return ", ".join(f"{g} {np.sum(r[m] ** 2) / tot:.0%}" for g, m in groups if m.any())


r_ex = A @ ue - b
print(f"{a.problem}, Nx = {N}, ell {a.ell:g}: loss of the exact solution {loss(r_ex):.3e}")
for path in a.files:
    dims, P, f = read_vts(path)
    if dims != (N, N, N):
        sys.exit(f"{path}: grid {dims}, expected {N}^3 -- write it with gridstates.Nx_eval = {N}")
    U = f["U"].astype(float).ravel()                         # C order = the replica's node order
    if np.abs(P.reshape(N, N, N, 3, order="F").reshape(-1, 3) - X).max() > 1e-5:
        sys.exit(f"{path}: node coordinates do not match the replica's grid")
    e = U - ue
    r = A @ U - b
    print(f"\n{path}")
    print(f"  node error: rmse {rms(e):.2e} (inside {rms(e[inside]):.2e}, outside {rms(e[~inside]):.2e}), max {np.abs(e).max():.2e}")
    print(f"  loss {loss(r):.3e}  ({loss(r) / loss(r_ex):.0f}x the exact solution's)")
    print(f"     by row kind: {shares(r)}")
    top = np.argsort(r ** 2)[::-1]
    print(f"     largest 10 / 100 rows hold {np.sum(r[top[:10]] ** 2) / np.sum(r ** 2):.0%} / {np.sum(r[top[:100]] ** 2) / np.sum(r ** 2):.0%};"
          f" largest row: {kind[top[0]]}, {'inside' if inside[top[0]] else 'outside'}, |phi|/h {abs(pb['phi'](X[top[0]])) / s['h']:.2f}")
    c, *_ = np.linalg.lstsq(cols, e, rcond=None)
    e_s = cols @ c
    e_r = e - e_s
    print(f"  error = smooth part (rmse {rms(e_s):.2e}) + rough part (rmse {rms(e_r):.2e})")
    print(f"     loss from the smooth part alone {loss(A @ e_s):.3e}, from the rough part alone {loss(A @ e_r):.3e}")
    print(f"     rough part by row kind: {shares(A @ e_r)}")
    d, *_ = np.linalg.lstsq(AC, -r, rcond=1e-13)
    U2 = U + cols @ d
    print(f"  after the best smooth correction: loss {loss(A @ U2 - b):.3e}, node rmse {rms(U2 - ue):.2e}")
