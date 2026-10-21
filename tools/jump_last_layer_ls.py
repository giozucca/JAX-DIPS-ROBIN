"""
Exact least-squares solve for the last layer of a trained jump network, offline, in NumPy.

The DoubleMLP output on each side is  u = tanh(tanh(x W0 + b0) W1 + b1) W2 + b2,  linear in (W2, b2):
101 numbers per side, 202 in all. Given the trained hidden layers, the best (W2, b2) for the loss is a
linear least-squares problem on the jump rows (rebuilt with tools/jump_replica3d.py, equal to the JAX
rows). This asks whether the hidden layers already contain the answer and only the last-layer
combination is stuck (error drops to the replica's best fit), or whether the hidden layers are the limit.

    python3 tools/jump_last_layer_ls.py params_lbfgs32.npz --nodes nodes_lbfgs32.vts

params_*.npz holds the checkpoint's haiku params as "module|w" / "module|b" arrays. --nodes, the .vts
written with the evaluation grid equal to the training grid, checks the forward pass against the
network's node values from the run.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jump_replica3d as rp  # noqa: E402
from crop_vts import read_vts  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("params", nargs="+")
ap.add_argument("--nodes", nargs="*", default=[], help="matching .vts files of node values, same order as params")
ap.add_argument("--nx", type=int, default=32)
ap.add_argument("--neval", type=int, default=128, help="whole-box RMSE on this grid, as the run's accuracy block")
ap.add_argument("--ells", type=float, nargs="+", default=[1.0e4, 1.0e2], help="weights the last layer is solved with")
a = ap.parse_args()

N = a.nx
pb = rp.make_problem("paper", radius=2 / 3)
s = rp.build(N, pb, True, alternate=True, bias="slow")
X, ue, inside = s["X"].reshape(-1, 3), s["u_exact"], s["side"] < 0
systems = {ell: rp.scaled(s, "interface", ell=ell, cut_floor=0.05) for ell in sorted(set(a.ells) | {1.0e4})}
rms = lambda v: np.sqrt(np.mean(v ** 2))
loss = lambda A, b, u: 0.5 * np.mean((A @ u - b) ** 2)


def layers(P, side):
    g = lambda i, k: P[f"double_mlp/~mlp_{side}_fn/linear{'' if i == 0 else '_' + str(i)}|{k}"].astype(float)
    return [(g(i, "w"), g(i, "b")) for i in range(3)]


def features(L, pts):
    """Last hidden layer plus a ones column, so that u = features @ [W2; b2]."""
    h = np.tanh(pts @ L[0][0] + L[0][1])
    h = np.tanh(h @ L[1][0] + L[1][1])
    return np.hstack([h, np.ones((len(pts), 1))])


def last(L):
    return np.concatenate([L[2][0].ravel(), L[2][1].ravel()])


def whole_box_rmse(Lm, Lp, cm, cp, n=128, chunk=1 << 18):
    """sqrt(mean(error^2)) over the n^3 box grid, sides by the sign of phi (as tests/test_poisson.py)."""
    x = np.linspace(-1, 1, n)
    G = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    se = 0.0
    for i in range(0, len(G), chunk):
        p = G[i:i + chunk]
        plus = pb["phi"](p) >= 0
        u = np.where(plus, features(Lp, p) @ cp, features(Lm, p) @ cm)
        se += np.sum((u - np.where(plus, pb["up"](p), pb["um"](p))) ** 2)
    return np.sqrt(se / len(G))


A4, b4 = systems[1.0e4]
print(f"sphere, Nx = {N}: replica best fit (degree-10 polynomials) for reference: node rmse 7.37e-05 at ell 1e4")
for k, path in enumerate(a.params):
    P = dict(np.load(path))
    Lm, Lp = layers(P, "m"), layers(P, "p")
    Fm, Fp = features(Lm, X), features(Lp, X)
    M = np.hstack([Fm * inside[:, None], Fp * ~inside[:, None]])          # node values = M @ [cm; cp]
    c0 = np.concatenate([last(Lm), last(Lp)])
    u0 = M @ c0
    print(f"\n{path}")
    if k < len(a.nodes):
        dims, _, f = read_vts(a.nodes[k])
        print(f"  forward pass vs the run's node values: max |difference| {np.abs(u0 - f['U'].astype(float).ravel()).max():.1e}")
    nm = len(c0) // 2
    print(f"  as trained:        loss (ell 1e4) {loss(A4, b4, u0):.3e}, node rmse {rms(u0 - ue):.2e} "
          f"(inside {rms((u0 - ue)[inside]):.2e}, outside {rms((u0 - ue)[~inside]):.2e}), "
          f"whole-box rmse {whole_box_rmse(Lm, Lp, c0[:nm], c0[nm:], a.neval):.2e}")
    for ell in a.ells:
        A, b = systems[ell]
        AM = A @ M
        sv = np.linalg.svd(AM, compute_uv=False)
        c, *_ = np.linalg.lstsq(AM, b, rcond=1e-13)
        u = M @ c
        print(f"  last layer solved with ell {ell:.0e}: loss (ell 1e4) {loss(A4, b4, u):.3e}, node rmse {rms(u - ue):.2e} "
              f"(inside {rms((u - ue)[inside]):.2e}, outside {rms((u - ue)[~inside]):.2e}), "
              f"whole-box rmse {whole_box_rmse(Lm, Lp, c[:nm], c[nm:], a.neval):.2e}; "
              f"condition number {sv[0] / sv[-1]:.1e}, |change| {np.linalg.norm(c - c0):.2e}")
