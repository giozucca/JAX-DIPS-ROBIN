"""
Jump-condition sweep, original against alternating tetrahedral split: errors against
effective resolution, over the whole box.

    ./visualization_env/bin/python tools/plot_jump.py [folder]

Looks in [folder] (default: the repository root) for the .vts files the jump sweep writes,
    jump_<geometry>_tets_<true|false>/Nx<N>/<geometry>/<geometry>.vts
or the same under jump/<geometry>_tets_<true|false>/, and writes to <folder>/GRAPHS_JUMP/:
    <geometry>_jump_convergence.png   L2 (left) and L_inf (right) for both splits
    jump_convergence_data.txt         the errors, local orders and fitted orders
Errors are over the whole box, since a jump problem is solved on both sides of the
interface: L2 = sqrt(sum(e^2) dx dy dz) and L_inf = max |e|, as tests/test_poisson.py logs
them. Missing runs are skipped, so it also works while the sweep is still running.
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_vts import read_vts

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "GRAPHS_JUMP")
NXS = [8, 16, 32, 64]
LBOX = 2.0                                     # the domain is [-1, 1] in every direction
GEOMS = [("sphere", "Sphere"), ("star", "Star")]
# (alternate_tets value, legend, Okabe-Ito colour, linestyle, marker)
ARMS = [("false", "original split", "#D55E00", "--", "s"),
        ("true", "alternating split", "#0072B2", "-", "o")]
INK, GRID = "#1a1a1a", "#c9c9c9"

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.7, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "savefig.dpi": 300, "figure.dpi": 300,
})


def find(geom, tets, nx):
    for sub in (f"jump_{geom}_tets_{tets}", os.path.join("jump", f"{geom}_tets_{tets}")):
        path = os.path.join(BASE, sub, f"Nx{nx}", geom, f"{geom}.vts")
        if os.path.exists(path):
            return path
    return None


def norms(path):
    """Whole-box L2 (volume weighted) and L_inf of the error stored in one .vts."""
    _, pts, d = read_vts(path)
    P = np.asarray(pts)
    h = [np.diff(np.unique(P[:, k]))[0] for k in range(3)]
    e = np.asarray(d["U-U_exact"], dtype=float)
    return float(np.sqrt((e ** 2).sum() * h[0] * h[1] * h[2])), float(np.abs(e).max())


def series(geom, tets, k):
    """Effective resolutions and errors (k = 0: L2, 1: L_inf) of the runs that exist."""
    nx = [n for n in NXS if (geom, tets, n) in RES]
    return nx, LBOX / (np.array(nx, float) - 1), np.array([RES[(geom, tets, n)][k] for n in nx])


def figure(geom, label, path):
    h = LBOX / (np.array(NXS, float) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for ax, k, name in zip(axes, (0, 1), (r"$L_2$", r"$L_\infty$")):
        pts = []
        for tets, arm, col, ls, mk in ARMS:
            nx, x, y = series(geom, tets, k)
            if nx:
                ax.loglog(x, y, color=col, ls=ls, lw=1.7, marker=mk, ms=6, mfc="white", mew=1.5, label=arm)
                pts += list(zip(x, y))
        # O(h) and O(h^2) guides fanning out from just above the largest error at Nx = 16
        # (the coarsest run present if Nx = 16 is missing), so neither swamps the data.
        x0 = h[1] if any(np.isclose(xi, h[1]) for xi, _ in pts) else max(xi for xi, _ in pts)
        y0 = 1.25 * max(yi for xi, yi in pts if np.isclose(xi, x0))
        for order, gls, glab in ((1, ":", r"$O(h)$"), (2, "-.", r"$O(h^2)$")):
            ax.loglog(h, y0 * (h / x0) ** order, color=INK, ls=gls, lw=1.1, label=glab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(h); ax.set_xticklabels([f"{LBOX:g}/{n - 1}" for n in NXS])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.invert_xaxis()                      # finest grid on the right
        ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=12)
        ax.grid(True, which="major", color=GRID, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Effective resolution", fontsize=14)
        ax.set_title(name, fontsize=14)
    axes[0].set_ylabel("Error over the box", fontsize=14)
    axes[0].legend(frameon=False, fontsize=11, loc="lower left")    # coarse h, small error: always empty
    fig.suptitle(label, fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


RES = {(g, t, n): norms(p) for g, _ in GEOMS for t, *_ in ARMS for n in NXS if (p := find(g, t, n))}
if not RES:
    sys.exit(f"no jump runs found in {BASE}; expected jump_<geometry>_tets_<true|false>/Nx<N>/<geometry>/<geometry>.vts")

os.makedirs(OUT, exist_ok=True)
lines = ["GRAPHS_JUMP -- jump-condition sweep, original against alternating tetrahedral split",
         "errors over the whole box: L2 = sqrt(sum(e^2) dx dy dz), L_inf = max |e|", ""]
for geom, label in GEOMS:
    if not any(g == geom for g, _, _ in RES):
        continue
    for tets, arm, *_ in ARMS:
        nx, h, l2 = series(geom, tets, 0)
        if not nx:
            continue
        li = series(geom, tets, 1)[2]
        lines += [f"{label}, {arm} (alternate_tets={tets})",
                  f"  {'Nx':>4}{'h':>8}{'L2':>14}{'L_inf':>14}{'order L2':>11}{'order Linf':>12}"]
        for i, n in enumerate(nx):
            o2 = "" if i == 0 else f"{np.log(l2[i - 1] / l2[i]) / np.log(h[i - 1] / h[i]):+.2f}"
            oi = "" if i == 0 else f"{np.log(li[i - 1] / li[i]) / np.log(h[i - 1] / h[i]):+.2f}"
            lines.append(f"  {n:>4}{f'2/{n - 1}':>8}{l2[i]:14.4e}{li[i]:14.4e}{o2:>11}{oi:>12}")
        sel = [i for i, n in enumerate(nx) if n >= 16]
        if len(sel) >= 2:
            f2 = np.polyfit(np.log(h[sel]), np.log(l2[sel]), 1)[0]
            fi = np.polyfit(np.log(h[sel]), np.log(li[sel]), 1)[0]
            lines.append(f"  fit over Nx = {nx[sel[0]]}..{nx[sel[-1]]}:   L2 {f2:+.2f}   L_inf {fi:+.2f}")
        lines.append("")
    figure(geom, label, os.path.join(OUT, f"{geom}_jump_convergence.png"))

with open(os.path.join(OUT, "jump_convergence_data.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"wrote {OUT}/")
