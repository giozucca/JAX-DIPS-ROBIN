"""
Paper figure: solution and error at the finest resolution, all three geometries.

3 rows (Sphere / Star / Star3) x 2 columns (U, U - U_exact) at Nx=64, sliced
through the interior.

The colour scales are IDENTICAL to those of the per-geometry figures produced by
make_field_figures.py, so this figure can be read directly against them. Deriving a
tighter scale from the Nx=64 data alone would make the same errors render darker here
than in the per-geometry figures, implying a difference that does not exist.

This is the companion to the convergence plots: those give the RATE, this shows
the error has no structure -- smooth, global, no pile-up at the interface, which
is the obvious worry for a boundary condition imposed by a first-order Taylor
projection in cut cells.

Usage:  python3 tools/make_summary_figure.py
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_vts import read_vts

RAW, OUT, NX = "JAX-DIPS-RobinRAW", "figures/summary", 64
GEOMS = [("sphere_Robin", "Sphere"), ("star_Robin", "Star"), ("star_Robin3", "Star3")]

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 10,
    "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def best_slice(phi):
    inside = phi <= 0.0
    best = (None, None, -1)
    for ax in range(3):
        c = inside.sum(axis=tuple(j for j in range(3) if j != ax))
        i = int(c.argmax())
        if c[i] > best[2]:
            best = (ax, i, int(c[i]))
    return best[0], best[1]


def take(a, ax, i):
    return [a[i, :, :], a[:, i, :], a[:, :, i]][ax]


data = {}
for key, title in GEOMS:
    p = os.path.join(RAW, f"{key}_{NX}", f"{key}_{NX}.vts")
    _, _, a = read_vts(p)
    data[key] = a

# Scales must match make_field_figures.py exactly:
#   U     -- largest |U_exact| over every geometry and resolution
#   error -- largest error at the reference resolution Nx=16, over every geometry
#            (Nx=16 is the coarsest resolution passing the r/dx >= 3 resolved criterion)
_U, _E = 0.0, 0.0
for _k, _ in GEOMS:
    for _n in (8, 16, 32, 64):
        _p = os.path.join(RAW, f"{_k}_{_n}", f"{_k}_{_n}.vts")
        if not os.path.isfile(_p):
            continue
        _, _, _a = read_vts(_p)
        _m = _a["phi"] <= 0
        _U = max(_U, float(np.abs(_a["U_exact"][_m]).max()))
        if _n == 16:
            _E = max(_E, float(np.abs(_a["U-U_exact"][_m]).max()))
ULIM, ELIM = _U, _E
print(f"shared scales (matching the per-geometry figures):")
print(f"    U     +-{ULIM:.4f}")
print(f"    error +-{ELIM:.4e}")
for _k, _t in GEOMS:
    _v = float(np.abs(data[_k]["U-U_exact"][data[_k]["phi"] <= 0]).max())
    print(f"      {_t:7s} Nx=64 error {_v:.3e} = {100*_v/ELIM:5.1f}% of the scale")

fig, axes = plt.subplots(3, 2, figsize=(6.4, 9.2))
for r, (key, title) in enumerate(GEOMS):
    a = data[key]
    ax_, idx = best_slice(a["phi"])
    phi_s = take(a["phi"], ax_, idx)
    u_s = np.where(phi_s <= 0, take(a["U"], ax_, idx), np.nan)
    e_s = np.where(phi_s <= 0, take(a["U-U_exact"], ax_, idx), np.nan)

    im_u = axes[r, 0].imshow(u_s.T, origin="lower", cmap="RdBu_r",
                             vmin=-ULIM, vmax=ULIM, interpolation="nearest")
    im_e = axes[r, 1].imshow(e_s.T, origin="lower", cmap="RdBu_r",
                             vmin=-ELIM, vmax=ELIM, interpolation="nearest")
    for c, lab in ((0, "$U$"), (1, "$U - U_{exact}$")):
        ax = axes[r, c]
        ax.contour(phi_s.T, levels=[0.0], colors="k", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if c == 0:
            ax.set_ylabel(title, fontsize=11, labelpad=8)
        if r == 0:
            ax.set_title(lab, fontsize=11, pad=8)
    # L_inf over the full 3D interior, not the displayed slice -- this is the
    # quantity reported in the convergence tables, so the figure corroborates them.
    linf = float(np.abs(a["U-U_exact"][a["phi"] <= 0]).max())
    axes[r, 1].text(0.97, 0.03, f"$L_\\infty$ = {linf:.2e}", transform=axes[r, 1].transAxes,
                    ha="right", va="bottom", fontsize=8)

fig.suptitle(f"Robin solution and error at $N_x = {NX}$", y=0.985, fontsize=12)
fig.tight_layout(rect=(0, 0.065, 1, 0.975))
b0 = axes[-1, 0].get_position(); b1 = axes[-1, 1].get_position()
cb_u = fig.colorbar(im_u, cax=fig.add_axes([b0.x0, 0.034, b0.width, 0.010]),
                    orientation="horizontal")
cb_u.set_label("$U$", fontsize=9)
cb_e = fig.colorbar(im_e, cax=fig.add_axes([b1.x0, 0.034, b1.width, 0.010]),
                    orientation="horizontal")
cb_e.set_label("$U - U_{exact}$", fontsize=9)
for cb in (cb_u, cb_e):
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    cb.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"fields_Nx{NX}")
for ext in ("png",):
    fig.savefig(f"{stem}.{ext}", dpi=220, bbox_inches="tight")
print(f"wrote {stem}.png")
