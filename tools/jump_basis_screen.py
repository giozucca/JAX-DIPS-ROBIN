"""
Offline screen of network-style bases for the jump problem: which activation, initial weight scale and
width can represent the rows' answer in float32, at every grid of a sweep, with one setting for all?

For each problem (sphere, star; JAX-Robin geometry), grid Nx and candidate basis, the jump rows are built
by the NumPy replica (tools/jump_replica3d.py, equal to the JAX rows: fixed scheme, Bias Slow, interface
weights, cut floor, alternating tetrahedra). One hidden layer of random weights is drawn per side, the
same weights at every Nx (one model for the whole sweep), and the last layer is solved as
Trainer.solve_last_layer does it in float32: features in float32, rows applied in float32, reduced QR +
SVD truncated at --rcond, then --refine steps of iterative refinement. The node RMSE is printed against
the replica's best fit (degree-10 Chebyshev, exact least squares in float64), with local orders.

Random hidden layers are not trained ones. This screens the property that failed in the trained 2 x 100
networks: whether a basis of a given kind keeps the rows' answer in float32 at every Nx.

    python3 tools/jump_basis_screen.py
    python3 tools/jump_basis_screen.py --problems star --nx 16 32 64 --activations sin --scales 1 2 --widths 400 800
    python3 tools/jump_basis_screen.py --chebyshev ...      # also run the Chebyshev basis through the float32 path

Memory: at Nx = 64 a width-W basis needs about 4 x 262144 x (2W + 2) x 4 bytes (W = 1600: about 13 GB).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jump_replica3d as rp  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--problems", nargs="+", default=["sphere", "star"], choices=["sphere", "star"])
ap.add_argument("--nx", type=int, nargs="+", default=[8, 16, 32, 64])
ap.add_argument("--activations", nargs="+", default=["tanh", "sin"], choices=["tanh", "sin"])
ap.add_argument("--scales", type=float, nargs="+", default=[0.3, 1.0, 3.0],
                help="standard deviation of the hidden layer's weights (the network's init_std)")
ap.add_argument("--widths", type=int, nargs="+", default=[400, 800, 1600], help="hidden units per side")
ap.add_argument("--rcond", type=float, default=1.0e-6)
ap.add_argument("--refine", type=int, default=1)
ap.add_argument("--ell", type=float, default=1.0e4)
ap.add_argument("--cut-floor", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chebyshev", action="store_true", help="also put the degree-10 Chebyshev basis through the float32 path")
a = ap.parse_args()

rms = lambda v: np.sqrt(np.mean(v ** 2))
f32 = np.float32


def solve_f32(M, A32, b32, ue):
    """The trainer's float32 last-layer solve from zero weights: QR + truncated SVD, then refinement."""
    G = (A32 @ M).astype(f32)
    Q, R = np.linalg.qr(G)
    del G
    Ur, s, Vt = np.linalg.svd(R, full_matrices=False)
    inv = np.zeros_like(s)
    keep = s > a.rcond * s[0]
    inv[keep] = 1 / s[keep]
    pinv = lambda r: Vt.T @ (inv * (Ur.T @ (Q.T @ r)))
    c = pinv(b32).astype(f32)
    for _ in range(a.refine):
        c = (c - pinv((A32 @ (M @ c) - b32).astype(f32))).astype(f32)
    return rms((M @ c).astype(float) - ue)


def orders(errs, Ns):
    h = 2.0 / (np.array(Ns) - 1)
    return " ".join(f"{np.log(errs[i] / errs[i + 1]) / np.log(h[i] / h[i + 1]):+.2f}" for i in range(len(errs) - 1))


for name in a.problems:
    pb = rp.make_problem("paper", radius=2 / 3) if name == "sphere" else rp.make_star("jaxrobin")
    systems, floors = {}, []
    for N in a.nx:
        s = rp.build(N, pb, True, alternate=True, bias="slow")
        A, b = rp.scaled(s, "interface", ell=a.ell, cut_floor=a.cut_floor)
        cols = rp.poly_columns(s, 10)
        c, *_ = np.linalg.lstsq(A @ cols, b, rcond=1e-13)
        floors.append(rms(cols @ c - s["u_exact"]))
        systems[N] = (s, A.astype(f32), b.astype(f32), cols if a.chebyshev else None)
    print(f"\n===== {name}: node RMSE at Nx = {a.nx}")
    print(f"  {'best fit (Chebyshev, float64)':38s} " + "  ".join(f"{e:.2e}" for e in floors) + f"   orders {orders(floors, a.nx)}")
    if a.chebyshev:
        errs = [solve_f32(systems[N][3].astype(f32), systems[N][1], systems[N][2], systems[N][0]["u_exact"]) for N in a.nx]
        print(f"  {'Chebyshev degree 10, float32 path':38s} " + "  ".join(f"{e:.2e}" for e in errs) + f"   orders {orders(errs, a.nx)}")
    for act in a.activations:
        fn = np.tanh if act == "tanh" else np.sin
        for sc in a.scales:
            for W in a.widths:
                rng = np.random.default_rng(a.seed)
                wts = []
                for _ in range(2):                        # u- then u+: the same draws at every Nx
                    Wt = rng.normal(0, sc, (3, W)).astype(f32)
                    bt = (rng.uniform(-np.pi, np.pi, W) if act == "sin" else rng.uniform(-sc, sc, W)).astype(f32)
                    wts.append((Wt, bt))
                t0, errs = time.time(), []
                for N in a.nx:
                    s, A32, b32, _ = systems[N]
                    X = s["X"].reshape(-1, 3).astype(f32)
                    inside = s["side"] < 0
                    M = np.hstack([np.hstack([fn(X @ Wt + bt), np.ones((len(X), 1), f32)]) * side[:, None]
                                   for (Wt, bt), side in zip(wts, (inside, ~inside))]).astype(f32)
                    errs.append(solve_f32(M, A32, b32, s["u_exact"]))
                    del M
                ratio = "  ".join(f"{e / f:4.1f}x" for e, f in zip(errs, floors))
                print(f"  {f'{act}, scale {sc:g}, width {W}':38s} " + "  ".join(f"{e:.2e}" for e in errs)
                      + f"   orders {orders(errs, a.nx)}   (vs best fit: {ratio}; {time.time() - t0:.0f} s)", flush=True)
