"""
3D NumPy replica of the JAX-DIPS jump-condition rows for the JAX-DIPS sphere, with the code's
extrapolation rules ("original") or corrected ones ("fixed"). No JAX, no network.

    python tools/jump_replica3d.py [--fixed] [--mu paper|swap|equal] [--nx 8 16 32 64] [--ls-max 32]
    python tools/jump_replica3d.py --check-rules [--mu swap]      # exact-input test of the four rules
    python tools/jump_replica3d.py --problem star --geometry jaxrobin --fixed --bias slow \
        --weights interface --ell 1e4 --cut-floor 0.05 --nx 16 32 48 64 96 --ls-max 96   # JAX-Robin shapes

Problem (tests/confs/experiment_configs.py::sphere; Guittet et al. 2015, sec. 4.6):

    -div(mu- grad u-) = f-  in Omega- = {|x| < 1/2},   -div(mu+ grad u+) = f+  in the rest of [-1, 1]^3
    [u] = u+ - u- = alpha,  [mu du/dn] = mu+ du+/dn - mu- du-/dn = beta  on Gamma,   u = u+ on the walls
    u- = exp(z),  u+ = cos(x) sin(y),  k = 0
    --mu paper: mu- = y^2 ln(x+2) + 4, mu+ = exp(-z)   (mu- > mu+ near Gamma: rule set 1)
    --mu swap:  the two swapped                       (mu+ > mu-: rule set 2)
    --mu equal: mu = 1 on both sides                  (the regression terms vanish)

beta here is the papers' [mu du/dn]. JAX-DIPS's configs store -beta; the code adds +integral of
that to each row, which is the correct -integral of beta, and passes the stored value to the rules.

Rows (compute_Ax_and_b_preconditioned_fn): one per node. A wall node says u = u+. Every other node
integrates both sides over its cell,

    sum_f [ cm_f (u-(node) - u-(nb_f)) + cp_f (u+(node) - u+(nb_f)) ] = f- V- + f+ V+ - integral of beta,

cm_f = mu-(face centre) A-_f / h, cp_f = mu+(face centre) A+_f / h, from the five-tetrahedra split.
The unknown at a node is u on its own side. The other side's value at a node whose cell is cut is
extrapolated across Gamma with one side's normal derivative from a least-squares fit over the
3x3x3 neighbours (Bochkov & Gibou 2020, sec. 2). Rule set 1 (used when mu- > mu+) fits on Omega-,
set 2 fits on Omega+. "original" is the code: the printed rules, fed -beta, delta = phi.
"fixed": beta with its defined sign, the [mu] term of both u- rules with the derived sign, and
delta = phi / |grad phi|. Every row stays linear in the unknowns.

Reported, as error against the exact solution at all nodes of the box:
  - classical solve of the square system A u = b: the discretization's own error;
  - exact least-squares minimizer of the diagonal-scaled rows over two polynomials of total degree
    <= 10, one per side: what the network minimizes, without training.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev as Cheb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robin_cutcell_replica3d import C, FACE_TRIS, FACE_TRIS_M, TETS, TETS_M, clip_tet, tri_area, tri_wet

R = 0.5
CORNERS = list(C)
# The 27-point neighbourhood in get_Xijk's order (x fastest, then y, then z); index 13 is the node.
OFFS27 = np.array([[a, b, c] for c in (-1, 0, 1) for b in (-1, 0, 1) for a in (-1, 0, 1)], float)
NBR = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]


def make_problem(mu="paper", radius=R):
    """The jump sphere. radius 0.5 is JAX-DIPS's; JAX-Robin's sphere has radius 2/3."""
    x, y, z = (lambda P: P[..., 0]), (lambda P: P[..., 1]), (lambda P: P[..., 2])
    zero = lambda P: 0 * x(P)

    def phi(P):
        v = np.sqrt((P ** 2).sum(-1)) - radius
        return v + np.where(v > 0, 1.0, -1.0) * 1e-10            # level_set.perturb_level_set_fn
    um, up = (lambda P: np.exp(z(P))), (lambda P: np.cos(x(P)) * np.sin(y(P)))
    gum = lambda P: np.stack([zero(P), zero(P), np.exp(z(P))], -1)
    gup = lambda P: np.stack([-np.sin(x(P)) * np.sin(y(P)), np.cos(x(P)) * np.cos(y(P)), zero(P)], -1)
    lapm, lapp = (lambda P: np.exp(z(P))), (lambda P: -2 * np.cos(x(P)) * np.sin(y(P)))
    one = lambda P: np.ones(np.shape(P)[:-1])
    mu_a = (lambda P: y(P) ** 2 * np.log(x(P) + 2) + 4,
            lambda P: np.stack([y(P) ** 2 / (x(P) + 2), 2 * y(P) * np.log(x(P) + 2), zero(P)], -1))
    mu_b = (lambda P: np.exp(-z(P)), lambda P: np.stack([zero(P), zero(P), -np.exp(-z(P))], -1))
    mu_1 = (one, lambda P: np.stack([zero(P)] * 3, -1))
    (mum, gmum), (mup, gmup) = {"paper": (mu_a, mu_b), "swap": (mu_b, mu_a), "equal": (mu_1, mu_1)}[mu]
    fm = lambda P: -((gmum(P) * gum(P)).sum(-1) + mum(P) * lapm(P))
    fp = lambda P: -((gmup(P) * gup(P)).sum(-1) + mup(P) * lapp(P))

    jump_flux = lambda P: mup(P)[..., None] * gup(P) - mum(P)[..., None] * gum(P)

    def beta(P):                                                    # [mu du/dn], n = P / |P|
        n = P / np.linalg.norm(P, axis=-1, keepdims=True)
        return (jump_flux(P) * n).sum(-1)
    return dict(phi=phi, um=um, up=up, mum=mum, mup=mup, fm=fm, fp=fp, alpha=lambda P: up(P) - um(P),
                beta=beta, beta_cfg=lambda P: -beta(P))             # the config stores -beta


def _phi_star_jaxdips(P):
    """JAX-DIPS sec. 4.3 star: r0 = 0.483, spiky (modes 3, 4, 7)."""
    r0, modes = 0.483, ((3, 0.1, 0.5), (4, -0.1, 1.8), (7, 0.15, 0.0))
    r2, rho2 = (P ** 2).sum(-1), P[..., 0] ** 2 + P[..., 1] ** 2
    ratio = np.divide(rho2, r2, out=np.zeros_like(r2), where=r2 > 0)
    th = np.arctan2(P[..., 1], P[..., 0])
    core = sum(b * np.cos(n * (th - t)) for n, b, t in modes)
    return np.sqrt(r2) - r0 * (1 + ratio ** 2 * core)


def _phi_star_jaxrobin(P):
    """JAX-Robin's star (experiment_configs.py::star_Robin), scaled by S to the half-extent 2/3 of
    the Robin geometries: milder bumps (modes 3, 4, 3)."""
    S = 0.5089048
    Q = P / S
    rho2, r2, az = Q[..., 0] ** 2 + Q[..., 1] ** 2, (Q ** 2).sum(-1), np.arctan2(Q[..., 1], Q[..., 0])
    t = rho2 / (r2 + 1e-8)
    pert = sum(b * t ** (n / 2) * np.cos(n * (az - 0.05)) for b, n in ((-0.05, 3), (0.05, 4), (-0.10, 3)))
    return S * (np.sqrt(r2) - 1.183 * (1 + (rho2 / 10) ** 2) + pert)


def make_star(geometry="jaxdips"):
    """The JAX-DIPS jump star's data (tests/confs/experiment_configs.py::star), with f- derived
    correctly (the config's f- has the sign of its mu_y u_y term flipped), on either the JAX-DIPS
    star or JAX-Robin's star. phi is not a signed distance."""
    x, y, z = (lambda P: P[..., 0]), (lambda P: P[..., 1]), (lambda P: P[..., 2])
    phi0 = {"jaxdips": _phi_star_jaxdips, "jaxrobin": _phi_star_jaxrobin}[geometry]

    def phi(P):
        v = phi0(P)
        return v + np.where(v > 0, 1.0, -1.0) * 1e-10

    def grad_phi(P, e=1e-6):
        E = np.eye(3) * e
        return np.stack([(phi0(P + E[d]) - phi0(P - E[d])) / (2 * e) for d in range(3)], -1)
    um = lambda P: np.sin(2 * x(P)) * np.cos(2 * y(P)) * np.exp(z(P))
    gum = lambda P: np.stack([2 * np.cos(2 * x(P)) * np.cos(2 * y(P)) * np.exp(z(P)),
                              -2 * np.sin(2 * x(P)) * np.sin(2 * y(P)) * np.exp(z(P)), um(P)], -1)
    t = lambda P: (y(P) - x(P)) / 3
    P5 = lambda P: 16 * t(P) ** 5 - 20 * t(P) ** 3 + 5 * t(P)
    dP5 = lambda P: 80 * t(P) ** 4 - 60 * t(P) ** 2 + 5
    ddP5 = lambda P: 320 * t(P) ** 3 - 120 * t(P)
    s = lambda P: x(P) + y(P) + 3
    up = lambda P: P5(P) * np.log(s(P)) * np.cos(z(P))
    gup = lambda P: np.stack([(-dP5(P) / 3) * np.log(s(P)) * np.cos(z(P)) + P5(P) * np.cos(z(P)) / s(P),
                              (dP5(P) / 3) * np.log(s(P)) * np.cos(z(P)) + P5(P) * np.cos(z(P)) / s(P),
                              -P5(P) * np.log(s(P)) * np.sin(z(P))], -1)
    lapp = lambda P: (2 * ddP5(P) / 9 - P5(P)) * np.log(s(P)) * np.cos(z(P)) - 2 * P5(P) * np.cos(z(P)) / s(P) ** 2
    mum = lambda P: 10 * (1 + 0.2 * np.cos(2 * np.pi * (x(P) + y(P))) * np.sin(2 * np.pi * (x(P) - y(P))) * np.cos(z(P)))
    gmum = lambda P: np.stack([4 * np.pi * np.cos(4 * np.pi * x(P)) * np.cos(z(P)),
                               -4 * np.pi * np.cos(4 * np.pi * y(P)) * np.cos(z(P)),
                               -(np.sin(4 * np.pi * x(P)) - np.sin(4 * np.pi * y(P))) * np.sin(z(P))], -1)
    mup = lambda P: np.ones(np.shape(P)[:-1])
    fm = lambda P: -(gmum(P) * gum(P)).sum(-1) + 7 * mum(P) * um(P)
    fp = lambda P: -lapp(P)
    jump_flux = lambda P: mup(P)[..., None] * gup(P) - mum(P)[..., None] * gum(P)

    def beta(P):
        g = grad_phi(P)
        return (jump_flux(P) * g).sum(-1) / np.linalg.norm(g, axis=-1)
    return dict(phi=phi, um=um, up=up, mum=mum, mup=mup, fm=fm, fp=fp, alpha=lambda P: up(P) - um(P),
                beta=beta, beta_cfg=lambda P: -(jump_flux(P) * grad_phi(P)).sum(-1))   # config: unnormalised grad phi


def extrapolation(pb, x0, h, fixed, bias="code"):
    """The other side's value at a node whose cell is cut, as (side, coef over the 27 values, const)."""
    X = OFFS27 * h
    E = np.eye(3) * h
    grad = np.array([pb["phi"](x0 + E[d]) - pb["phi"](x0 - E[d]) for d in range(3)]) / (2 * h)
    n = grad / np.linalg.norm(grad)
    d = pb["phi"](x0) / np.linalg.norm(grad) if fixed else pb["phi"](x0)
    mm, mp = pb["mum"](x0), pb["mup"](x0)
    rp = x0 - d * n                                                  # projection of the node onto Gamma
    a = pb["alpha"](rp)
    b = pb["beta"](rp) if fixed else pb["beta_cfg"](rp)              # the code passes the config's stored value
    e13 = np.zeros(27); e13[13] = 1.0
    pv = pb["phi"](x0 + X)
    # "code": set 1 (fit on Omega-) when mu- > mu+; "slow": fit in the slower-diffusing region instead
    set1 = (mm > mp) if bias == "code" else (mm <= mp)
    W = (pv < 0) if set1 else (pv > 0)                               # set 1 fits on Omega-, set 2 on Omega+
    D = np.linalg.pinv(X.T @ (W[:, None] * X)) @ (W[:, None] * X).T
    c = n @ D                                                        # du/dn ~ cc u_node_side + sum_q c_q u_q
    cc = -c.sum()
    if set1:
        K, jb = d * (mp - mm) / mp, a + d * b / pb["mup"](rp)     # beta/mu at the projection, as the code
        if d > 0:   # node in Omega+, u-:  u- = (u+ - alpha - d beta/mu+ + s K sum c_q u_q) / (1 - s K cc)
            s = 1.0 if fixed else -1.0                              # printed rule has the [mu] term with the wrong sign
            den = 1 - s * K * cc
            if not fixed:      # the code's form (eq. 2 / 4 of the JAX-DIPS paper), identical to the printed rule
                zq = K * c; z_ = K * cc; gq = zq / (1 - z_); g = -gq.sum()
                return "m", -gq + (1 - g) * e13, -(1 - g) * jb
            return "m", (e13 + s * K * c) / den, -jb / den
        return "p", e13 * (1 - K * cc) - K * c, jb                   # node in Omega-, u+ (correct as printed)
    L, jb = d * (mp - mm) / mm, a + d * b / pb["mum"](rp)
    if d > 0:       # node in Omega+, u-:  u- = u+ - alpha - d beta/mu- + s L (cc u+ + sum c_q u_q)
        s = 1.0 if fixed else -1.0
        return "m", e13 * (1 + s * L * cc) + s * L * c, -jb
    den = 1 + L * cc                                                 # node in Omega-, u+ (correct as printed)
    return "p", (e13 - L * c) / den, jb / den


def check_rules(pb, Ns=(16, 32, 64), bias="code"):
    """Largest error of each rule with the exact solution as input, original and fixed."""
    print(f"largest |extrapolated - exact| at cut nodes, exact inputs (fixed rules use bias {bias})")
    for fixed in (False, True):
        for N in Ns:
            x = np.linspace(-1, 1, N); h = x[1] - x[0]
            X3 = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
            xc = np.concatenate([x - h / 2, [x[-1] + h / 2]])
            pc = pb["phi"](np.stack(np.meshgrid(xc, xc, xc, indexing="ij"), -1))
            nneg = sum((pc[a:N + a, b:N + b, c:N + c] < 0).astype(int) for a, b, c in CORNERS)
            err = {}
            for i, j, k in np.argwhere((nneg > 0) & (nneg < 8)):
                x0 = X3[i, j, k]
                cube = x0 + OFFS27 * h
                u = np.where(pb["phi"](cube) < 0, pb["um"](cube), pb["up"](cube))
                bi = bias if fixed else "code"
                side, coef, const = extrapolation(pb, x0, h, fixed, bi)
                uses_set1 = (pb["mum"](x0) > pb["mup"](x0)) == (bi == "code")
                key = ("set 1" if uses_set1 else "set 2") + (", u-" if side == "m" else ", u+")
                exact = pb["um"](x0) if side == "m" else pb["up"](x0)
                err[key] = max(err.get(key, 0.0), abs(coef @ u + const - exact))
            print(f"  {'fixed   ' if fixed else 'original'} Nx={N:3d}  " + "   ".join(f"{k}: {v:.2e}" for k, v in sorted(err.items())))


def build(N, pb, fixed, alternate=True, bias="code"):
    """Sparse system A u = b (rows divided by their diagonal, as JAX-DIPS) and the grid."""
    x = np.linspace(-1, 1, N); h = x[1] - x[0]
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    xc = np.concatenate([x - h / 2, [x[-1] + h / 2]])
    pc = pb["phi"](np.stack(np.meshgrid(xc, xc, xc, indexing="ij"), -1))
    nneg = sum((pc[a:N + a, b:N + b, c:N + c] < 0).astype(int) for a, b, c in CORNERS)
    inside_full, cut = nneg == 8, (nneg > 0) & (nneg < 8)
    side = np.where(pb["phi"](X) < 0, -1, 1)                         # the side each node's unknown lives on
    idx = np.arange(N ** 3).reshape(N, N, N)

    # Geometry of every cell: volumes, face areas, integral of beta over Gamma.
    Vm = np.where(inside_full, h ** 3, 0.0)
    Am = np.repeat(np.where(inside_full, h * h, 0.0)[None], 6, 0)
    BI, AG = np.zeros((N, N, N)), np.zeros((N, N, N))                # integral of beta, area of Gamma
    offs = np.array(CORNERS, float) - 0.5
    for i, j, k in np.argwhere(cut):
        p8 = np.array([pc[i + a, j + b, k + c] for a, b, c in CORNERS])
        X8 = X[i, j, k] + offs * h
        TT, FT = (TETS_M, FACE_TRIS_M) if alternate and (i + j + k) % 2 else (TETS, FACE_TRIS)
        for t in TT:
            v, tris = clip_tet(X8[t], p8[t])
            Vm[i, j, k] += v
            for T in tris:
                # the row needs -integral of beta; the code adds +integral of the config's stored value
                bt = pb["beta"](T) if fixed else -pb["beta_cfg"](T)
                BI[i, j, k] += tri_area(T) * bt.mean()
                AG[i, j, k] += tri_area(T)
        Am[:, i, j, k] = [sum(tri_wet(p8[t]) for t in FT[f]) * h * h / 2 for f in range(6)]
    Vp = np.maximum(h ** 3 - Vm, 0.0)
    Ap = np.maximum(h * h - Am, 0.0)
    E = np.eye(3) * h
    faces = [X + s * E[ax] / 2 for ax in range(3) for s in (-1, 1)]
    cm = np.stack([pb["mum"](F) for F in faces]) * Am / h
    cp = np.stack([pb["mup"](F) for F in faces]) * Ap / h

    # u- and u+ at every node as linear maps of the unknowns: u-/+ = P u + q.
    rows = {"m": ([], [], []), "p": ([], [], [])}
    q = {"m": np.zeros(N ** 3), "p": np.zeros(N ** 3)}
    own = {-1: "m", 1: "p"}
    for i, j, k in np.ndindex(N, N, N):
        n0 = idx[i, j, k]
        if not cut[i, j, k]:               # bulk_point: the value on the cell's side, 0 on the other
            s_cell = "m" if inside_full[i, j, k] else "p"
            rows[s_cell][0].append(n0); rows[s_cell][1].append(n0); rows[s_cell][2].append(1.0)
            continue
        s = own[side[i, j, k]]
        rows[s][0].append(n0); rows[s][1].append(n0); rows[s][2].append(1.0)      # own side: the node's value
        o, coef, const = extrapolation(pb, X[i, j, k], h, fixed, bias)
        cube = [idx[i + a, j + b, k + c] for a, b, c in OFFS27.astype(int)]
        rows[o][0].extend([n0] * 27); rows[o][1].extend(cube); rows[o][2].extend(coef)
        q[o][n0] += const
    P = {s: sp.csr_matrix((v, (r, c)), shape=(N ** 3, N ** 3)) for s, (r, c, v) in rows.items()}

    def shift(dn):                                                     # (S u)[node] = u[node + dn], clipped at walls
        ii = np.clip(np.indices((N, N, N)) + np.array(dn)[:, None, None, None], 0, N - 1)
        return sp.csr_matrix((np.ones(N ** 3), (idx.ravel(), idx[ii[0], ii[1], ii[2]].ravel())), shape=(N ** 3,) * 2)
    S = [shift(dn) for dn in NBR]
    diag = cm.sum(0) + cp.sum(0)
    Lm = sp.diags(cm.sum(0).ravel()) - sum(sp.diags(cm[f].ravel()) @ S[f] for f in range(6))
    Lp = sp.diags(cp.sum(0).ravel()) - sum(sp.diags(cp[f].ravel()) @ S[f] for f in range(6))
    A = Lm @ P["m"] + Lp @ P["p"]
    b = (pb["fm"](X) * Vm + pb["fp"](X) * Vp - BI).ravel() - Lm @ q["m"] - Lp @ q["p"]
    wall = np.zeros((N, N, N), bool)
    wall[[0, -1]] = wall[:, [0, -1]] = wall[:, :, [0, -1]] = True
    # Faces of uncut cells whose neighbour's value comes from the other side's extrapolation.
    ng = sum((np.roll(side, -sh, axis=ax) != side).astype(float) for ax in range(3) for sh in (-1, 1))
    A_ghost = np.where(cut | wall, 0.0, ng) * h * h
    u_exact = np.where(side < 0, pb["um"](X), pb["up"](X)).ravel()
    return dict(A_raw=A.tocsr(), b_raw=b, diag=diag.ravel(), AG=AG.ravel(), A_ghost=A_ghost.ravel(),
                wall=wall.ravel(), u_wall=pb["up"](X).ravel(), X=X, h=h, side=side.ravel(), u_exact=u_exact,
                ncut=int(cut.sum()))


def scaled(sysm, weight="diag", ell=1.0, wall_w=1.0, cut_floor=0.0):
    """Rows divided by their weight: the diagonal (JAX-DIPS); sqrt(h^3 + ell A_Gamma); sqrt(h^3);
    or "interface", sqrt(h^3 + ell (A_Gamma + A_ghost)), which also counts the faces of uncut cells
    whose neighbour value is extrapolated. A wall row is u = u_wall, multiplied by wall_w (1 in the code)."""
    h3 = sysm["h"] ** 3
    # cut_floor: a cut cell counts as at least cut_floor * h^2 of interface. Without it a cell that
    # Gamma barely clips (a sliver) gets almost bulk weight although its row has O(1) truncation
    # error; on the JAX-Robin sphere such cells set the inner level badly at Nx = 64. 0.05 removes
    # that and keeps the star second order.
    AG = np.where(sysm["AG"] > 0, np.maximum(sysm["AG"], cut_floor * sysm["h"] ** 2), 0.0)
    w = {"diag": sysm["diag"], "vol_area": np.sqrt(h3 + ell * AG),
         "vol": np.full_like(sysm["diag"], np.sqrt(h3)),
         "interface": np.sqrt(h3 + ell * (AG + sysm["A_ghost"]))}[weight]
    wall = sysm["wall"]
    keep = sp.diags((~wall).astype(float) / np.where(wall, 1.0, w))
    A = (keep @ sysm["A_raw"] + sp.diags(wall_w * wall.astype(float))).tocsr()
    b = np.where(wall, wall_w * sysm["u_wall"], keep @ sysm["b_raw"])
    return A, b


def poly_columns(sysm, p=10):
    """u = (polynomial of total degree <= p) on each side, as a matrix over the coefficients."""
    T = [Cheb.chebvander(sysm["X"][..., d].ravel(), p) for d in range(3)]
    terms = [(a, b, c) for a in range(p + 1) for b in range(p + 1 - a) for c in range(p + 1 - a - b)]
    Phi = np.stack([T[0][:, a] * T[1][:, b] * T[2][:, c] for a, b, c in terms], 1)
    return np.hstack([Phi * (sysm["side"] < 0)[:, None], Phi * (sysm["side"] > 0)[:, None]])


def least_squares(A, b, cols):
    """Exact minimizer of |A u - b|^2 over u = cols @ coef."""
    coef, *_ = np.linalg.lstsq(A @ cols, b, rcond=1e-13)
    return cols @ coef


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--ls-max", type=int, default=32, help="largest Nx for the least-squares solve (memory)")
    ap.add_argument("--problem", choices=["sphere", "star"], default="sphere")
    ap.add_argument("--mu", choices=["paper", "swap", "equal"], default="paper", help="sphere only")
    ap.add_argument("--ell", type=float, default=1.0, help="length in the vol_area and interface weights")
    ap.add_argument("--cut-floor", type=float, default=0.0,
                    help="least interface area of a cut cell in the weights, as a fraction of h^2 (0.05 recommended)")
    ap.add_argument("--fixed", action="store_true", help="corrected extrapolation rules")
    ap.add_argument("--original-split", action="store_true", help="the same tetrahedral split in every cell")
    ap.add_argument("--check-rules", action="store_true")
    ap.add_argument("--bias", choices=["code", "slow"], default="code",
                    help="which side the gradient is fitted on: as the code, or the slower-diffusing side")
    ap.add_argument("--weights", nargs="+", default=["diag"], choices=["diag", "vol_area", "vol", "interface"],
                    help="row weights for the least-squares fit (the classical solve does not depend on them)")
    ap.add_argument("--no-classical", action="store_true", help="skip the square solve (slow at Nx=64)")
    ap.add_argument("--classical-max", type=int, default=64, help="largest Nx for the square solve (time, memory)")
    ap.add_argument("--geometry", choices=["jaxdips", "jaxrobin"], default="jaxdips",
                    help="jaxdips: sphere of radius 1/2, JAX-DIPS star; jaxrobin: sphere of radius 2/3, JAX-Robin star")
    a = ap.parse_args()
    if a.problem == "star":
        pb = make_star(a.geometry)
    else:
        pb = make_problem(a.mu, radius=R if a.geometry == "jaxdips" else 2 / 3)
    if a.check_rules:
        check_rules(pb, bias=a.bias)
        return
    h = 2.0 / (np.array(a.nx) - 1)
    orders = lambda v: " ".join(f"{np.log(v[i] / v[i + 1]) / np.log(h[i] / h[i + 1]):+.2f}"
                                if np.isfinite(v[i + 1]) else "  -  " for i in range(len(v) - 1))
    what = f"{a.geometry} " + ("star" if a.problem == "star" else f"sphere, mu {a.mu}")
    print(f"jump {what}, {'fixed' if a.fixed else 'original'} rules, {'original' if a.original_split else 'alternating'}"
          f" split, bias {a.bias}, ell {a.ell:g}: errors over all nodes of the box")
    print(f"{'':28s}" + "".join(f"{'Nx=' + str(n):>11}" for n in a.nx) + "   local orders")
    rows = [] if a.no_classical else [("classical solve", None)]
    rows += [(f"least squares, {w}", w) for w in a.weights]
    res = {(name, k): [] for name, _ in rows for k in ("rmse", "max")}
    for N in a.nx:
        t0 = time.time()
        sysm = build(N, pb, a.fixed, alternate=not a.original_split, bias=a.bias)
        cols = poly_columns(sysm) if N <= a.ls_max else None
        for name, w in rows:
            if w is None and N <= a.classical_max:
                A, b = scaled(sysm, "diag")
                e = spla.spsolve(A.tocsc(), b) - sysm["u_exact"]
            elif w is not None and cols is not None:
                e = least_squares(*scaled(sysm, w, ell=a.ell, cut_floor=a.cut_floor), cols) - sysm["u_exact"]
            else:
                e = np.full(1, np.nan)
            res[(name, "rmse")].append(np.sqrt((e ** 2).mean())); res[(name, "max")].append(np.abs(e).max())
        print(f"  (Nx={N}: {sysm['ncut']} cut cells, {time.time() - t0:.0f} s)", flush=True)
    for name, _ in rows:
        for k in ("rmse", "max"):
            v = np.array(res[(name, k)])
            label = f"{name:24s}{k:5s}" if k == "rmse" else f"{'':24s}{k:5s}"
            print(f"{label:28s}" + "".join(f"{e:11.3e}" for e in v) + "   " + orders(v))


if __name__ == "__main__":
    main()
