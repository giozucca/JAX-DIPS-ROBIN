"""
3D, NumPy-only replica of the Robin cut-cell rows for sphere_Robin. No JAX, no network,
no optimizer -- about 15 seconds on a laptop (a minute more with 128):

    python tools/robin_cutcell_replica3d.py [128]

It assembles the same rows as compute_Ax_and_b_preconditioned_fn_Robin (5-tetrahedra
linear geometry, face-centre fluxes, Bochkov-Gibou projection, every cell that overlaps
Omega-, rows divided by sqrt(V_m + interface area)) and solves them by EXACT least
squares over a smooth polynomial ansatz, so what it reports is the error of the discrete
problem itself.

It was checked against tests/check_robin_consistency.py on the runner at Nx=64 (exact
solution substituted, float32): 42872 rows, RMS residual 7.0499e-05, max 3.8617e-04,
loss 2.4850e-09. This file gives 42872, 7.0488e-05, 3.8577e-04, 2.4843e-09.

tets = "repo"         the same 5-tet split in every cell (the code before full-exploratory)
tets = "checkerboard" the split mirrored in x on cells with odd i+j+k, so two neighbouring
                      cells cut their shared face along the same diagonal
"""
import itertools, sys, time
import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as Cheb

R, MU, ALPHA, KK = 0.6666667, 2.0, 0.25, 0.5
MIN_VOL_FRAC = 1e-6

def phi_fn(P):
    v = np.sqrt((P ** 2).sum(-1)) - R
    return v + np.where(v > 0, 1.0, -1.0) * 1e-10          # level_set.perturb_level_set_fn

def uex(P): return np.cos(P[..., 0]) * np.sin(P[..., 1]) * np.cos(P[..., 2])
def f_fn(P): return (3 * MU + KK) * uex(P)
def g_fn(P):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    n = P / np.sqrt((P ** 2).sum(-1))[..., None]
    gu = np.stack([-np.sin(x) * np.sin(y) * np.cos(z), np.cos(x) * np.cos(y) * np.cos(z), -np.cos(x) * np.sin(y) * np.sin(z)], -1)
    return MU * (gu * n).sum(-1) + ALPHA * uex(P)

# corner (a,b,c) -> index ; tets exactly as geometric_integrations_per_point.py
C = {abc: i for i, abc in enumerate(itertools.product((0, 1), repeat=3))}
TETS = [[(0,0,0),(1,0,0),(0,1,0),(0,0,1)], [(1,1,0),(1,0,0),(0,1,0),(1,1,1)], [(1,0,1),(1,0,0),(1,1,1),(0,0,1)],
        [(0,1,1),(1,1,1),(0,1,0),(0,0,1)], [(1,1,1),(1,0,0),(0,1,0),(0,0,1)]]
TETS = [[C[v] for v in t] for t in TETS]
# the two triangles each face is split into by this decomposition (order: x-,x+,y-,y+,z-,z+)
FACE_TRIS = [[[(0,0,0),(0,1,0),(0,0,1)], [(0,1,1),(0,1,0),(0,0,1)]], [[(1,1,0),(1,0,0),(1,1,1)], [(1,0,1),(1,0,0),(1,1,1)]],
             [[(0,0,0),(1,0,0),(0,0,1)], [(1,0,1),(1,0,0),(0,0,1)]], [[(1,1,0),(0,1,0),(1,1,1)], [(0,1,1),(1,1,1),(0,1,0)]],
             [[(0,0,0),(1,0,0),(0,1,0)], [(1,1,0),(1,0,0),(0,1,0)]], [[(1,0,1),(1,1,1),(0,0,1)], [(0,1,1),(1,1,1),(0,0,1)]]]
FACE_TRIS = [[[C[v] for v in t] for t in f] for f in FACE_TRIS]
NBR = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
_inv = {i: abc for abc, i in C.items()}
_mir = lambda v: C[(1 - _inv[v][0], _inv[v][1], _inv[v][2])]
TETS_M = [[_mir(v) for v in t] for t in TETS]
_FM = [[[_mir(v) for v in t] for t in f] for f in FACE_TRIS]
FACE_TRIS_M = [_FM[1], _FM[0], _FM[2], _FM[3], _FM[4], _FM[5]]

def tri_wet(p):                 # fraction of a triangle where the linear interpolant of p is negative
    neg = p < 0; n = neg.sum()
    if n == 0: return 0.0
    if n == 3: return 1.0
    if n == 1:
        a = p[neg][0]; b, c = p[~neg]; return (a / (a - b)) * (a / (a - c))
    c = p[~neg][0]; a, b = p[neg]; return 1.0 - (c / (c - a)) * (c / (c - b))

def clip_tet(X, p):             # -> wetted volume, list of interface triangles
    o = np.argsort(p, kind="stable"); X, p = X[o], p[o]
    n = int((p < 0).sum()); vol = abs(np.linalg.det(X[1:] - X[0])) / 6.0
    cut = lambda i, j: (p[i] * X[j] - p[j] * X[i]) / (p[i] - p[j])
    if n == 0: return 0.0, []
    if n == 4: return vol, []
    if n == 1:
        t = [p[0] / (p[0] - p[j]) for j in (1, 2, 3)]
        return vol * t[0] * t[1] * t[2], [np.array([cut(0, 1), cut(0, 2), cut(0, 3)])]
    if n == 3:
        t = [p[3] / (p[3] - p[j]) for j in (0, 1, 2)]
        return vol * (1 - t[0] * t[1] * t[2]), [np.array([cut(3, 0), cut(3, 1), cut(3, 2)])]
    Q0, Q1, Q2, Q5 = cut(0, 2), cut(0, 3), cut(1, 3), cut(1, 2)
    tv = lambda A: abs(np.linalg.det(np.array(A[1:]) - A[0])) / 6.0
    v = tv([X[0], X[1], Q0, Q2]) + tv([Q5, X[1], Q0, Q2]) + tv([X[0], Q1, Q0, Q2])
    return v, [np.array([Q0, Q1, Q2]), np.array([Q0, Q5, Q2])]

def tri_area(T): return 0.5 * np.linalg.norm(np.cross(T[1] - T[0], T[2] - T[0]))

def build(N, tets="repo"):
    x = np.linspace(-1, 1, N); h = x[1] - x[0]
    xc = np.concatenate([x - h / 2, [x[-1] + h / 2]])
    pc = phi_fn(np.stack(np.meshgrid(xc, xc, xc, indexing="ij"), -1))
    negc = pc < 0
    nneg = sum(negc[a:N + a, b:N + b, c:N + c].astype(int) for a, b, c in C)
    offs = np.array(list(C.keys()), float) - 0.5
    idx = {}; cells = []
    for i, j, k in np.argwhere(nneg > 0):
        ctr = np.array([x[i], x[j], x[k]])
        p8 = np.array([pc[i + a, j + b, k + c] for a, b, c in C])
        if nneg[i, j, k] == 8:
            Vm, areaG, gint, A = h ** 3, 0.0, 0.0, np.full(6, h * h)
        else:
            X8 = ctr + offs * h
            Vm = areaG = gint = 0.0
            odd = tets == "checkerboard" and (i + j + k) % 2 == 1      # mirror every other cell
            TT, FT = (TETS_M, FACE_TRIS_M) if odd else (TETS, FACE_TRIS)
            for t in TT:
                v, tris = clip_tet(X8[t], p8[t]); Vm += v
                for T in tris:
                    a = tri_area(T); areaG += a; gint += a * g_fn(T).mean()
            A = np.array([sum(tri_wet(p8[t]) for t in FT[f]) for f in range(6)]) * h * h / 2
        if Vm <= MIN_VOL_FRAC * h ** 3: continue
        idx[(i, j, k)] = len(cells)
        cells.append(dict(ijk=(i, j, k), x=ctr, Vm=Vm, areaG=areaG, gint=gint, A=A, cut=nneg[i, j, k] < 8))
    # rows
    nodes = dict(idx); I, J, V = [], [], []; b = np.zeros(len(cells)); sumc = np.zeros(len(cells))
    def node(ijk):
        if ijk not in nodes: nodes[ijk] = len(nodes)
        return nodes[ijk]
    for r, c in enumerate(cells):
        xi = c["x"]; e = np.eye(3) * h
        grad = np.array([(phi_fn(xi + e[d]) - phi_fn(xi - e[d])) / (2 * h) for d in range(3)])
        gn = np.linalg.norm(grad); d = phi_fn(xi) / gn; xp = xi - d * grad / gn
        cf = MU * c["A"] / h; ael = ALPHA * c["areaG"]; coefR = MU * ael / (MU - ALPHA * d)
        s = 1.0 / np.sqrt(c["Vm"] + c["areaG"])
        I.append(r); J.append(r); V.append((KK * c["Vm"] + cf.sum() + coefR) * s)
        for fidx, dn in enumerate(NBR):
            if cf[fidx] != 0.0:
                I.append(r); J.append(node(tuple(np.add(c["ijk"], dn)))); V.append(-cf[fidx] * s)
        b[r] = (f_fn(xi) * c["Vm"] + c["gint"] + g_fn(xp) * d * ael / (MU - ALPHA * d)) * s
        sumc[r] = cf.sum()
    S = sp.csr_matrix((V, (I, J)), shape=(len(cells), len(nodes)))
    ijk = np.array(sorted(nodes, key=nodes.get)); Xn = np.stack([x[ijk[:, 0]], x[ijk[:, 1]], x[ijk[:, 2]]], 1)
    return dict(h=h, S=S, b=b, Xn=Xn, cells=cells, sumc=sumc, inside=phi_fn(Xn[:len(cells)]) <= 0)

def basis(X, p):                # tensor Chebyshev, total degree <= p
    Tx, Ty, Tz = (Cheb.chebvander(X[:, d], p) for d in range(3))
    return np.stack([Tx[:, a] * Ty[:, b] * Tz[:, c] for a in range(p + 1) for b in range(p + 1 - a) for c in range(p + 1 - a - b)], 1)

def solve_ls(pr, p=10):
    Phi = basis(pr["Xn"], p)
    coef, *_ = np.linalg.lstsq(pr["S"] @ Phi, pr["b"], rcond=1e-13)
    e = (Phi[:len(pr["cells"])] @ coef - uex(pr["Xn"][:len(pr["cells"])]))[pr["inside"]]
    return np.sqrt((e ** 2).mean()), np.abs(e).max()

if __name__ == "__main__":
    Ns = [8, 16, 32, 64] + ([128] if "128" in sys.argv else [])
    h = 2.0 / (np.array(Ns) - 1)
    print("exact least squares, polynomial ansatz of total degree 10; errors at the nodes inside Omega-")
    print(f"{'':50s}{'':5s}" + "".join(f"{'Nx=' + str(n):>10}" for n in Ns) + "   local orders")
    for name, tets in (("same 5-tet split in every cell", "repo"), ("5-tet split mirrored on a checkerboard", "checkerboard")):
        out = [solve_ls(build(N, tets=tets), 10) for N in Ns]
        for lab, v in (("RMSE", np.array([o[0] for o in out])), ("max", np.array([o[1] for o in out]))):
            orders = " ".join(f"{np.log(v[i] / v[i + 1]) / np.log(h[i] / h[i + 1]):+.2f}" for i in range(len(Ns) - 1))
            print(f"{name if lab == 'RMSE' else '':50s}{lab:5s}" + "".join(f"{x:>10.2e}" for x in v) + "   " + orders, flush=True)
    print("\nexact solution substituted into the rows (compare with tests/check_robin_consistency.py):")
    for tets in ("repo", "checkerboard"):
        pr = build(64, tets=tets); res = pr["S"] @ uex(pr["Xn"]) - pr["b"]
        print(f"  {tets:13s} Nx=64  rows {len(res)}  RMS res {np.sqrt((res ** 2).mean()):.4e}  "
              f"max res {np.abs(res).max():.4e}  loss {0.5 * (res ** 2).mean():.4e}")
