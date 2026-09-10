"""
Solution / error field figures from the JAX-DIPS .vts outputs.

One figure per geometry: rows are training resolutions, columns are U and
U - U_exact, each a slice through the volume masked to the interior (phi <= 0).
The exterior is left blank because the Omega+ network is untrained for a Robin
problem, so values there are meaningless.

Slice choice: the axis-aligned plane containing the most interior points, so a
disconnected geometry like star_Robin3 is cut where there is something to see.

Colour: U is signed and centred on zero, so both columns use a diverging map with
symmetric limits and a neutral midpoint. (The reference figures use a rainbow;
diverging is the correct encoding for signed data and keeps zero readable.)
BOTH columns use one global LINEAR scale over all 12 panels, shared across every
resolution AND every geometry. The error scale is set by the worst case in the
sweep (star_Robin3 at its reference resolution), so the sphere and star figures
render correspondingly paler -- which is correct, their errors really are smaller.
A per-geometry error scale was rejected: it makes three genuinely different
accuracy levels look alike, which is the opposite of what the figure is for.

Two rejected alternatives, for the record. Per-panel scaling normalises the
convergence away and makes a coarse panel look as clean as a fine one. A symmetric
LOG scale keeps every row visible but inflates small values, so a 4e-3 error still
renders saturated -- again hiding the convergence.

Each geometry's error scale is set by its Nx=16 field, the coarsest resolution that
passes the r/dx >= 3 resolved-range criterion used for the fitted orders. Nx=8 is
under-resolved and clips; its true max is printed in the panel title.

Usage:  python3 tools/make_field_figures.py
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_vts import read_vts

RAW = "JAX-DIPS-RobinRAW"
OUT = "figures"
GEOMS = [("sphere_Robin", "Sphere"), ("star_Robin", "Star"), ("star_Robin3", "Star3")]
NXS = [8, 16, 32, 64]

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def best_slice(phi):
    """axis and index of the axis-aligned plane holding the most interior points"""
    inside = phi <= 0.0
    best = (None, None, -1)
    for ax in range(3):
        counts = inside.sum(axis=tuple(j for j in range(3) if j != ax))
        i = int(counts.argmax())
        if counts[i] > best[2]:
            best = (ax, i, int(counts[i]))
    return best[0], best[1]


def take(arr, ax, i):
    return [arr[i, :, :], arr[:, i, :], arr[:, :, i]][ax]


# ---- pass 1: global colour limits over every geometry and resolution
_U, _E = 0.0, 0.0
_emins = []
for _k, _ in GEOMS:
    for _n in NXS:
        _p = os.path.join(RAW, f"{_k}_{_n}", f"{_k}_{_n}.vts")
        if not os.path.isfile(_p):
            continue
        _, _, _a = read_vts(_p)
        _m = _a["phi"] <= 0
        _U = max(_U, float(np.abs(_a["U_exact"][_m]).max()))
        _e = float(np.abs(_a["U-U_exact"][_m]).max())
        _E = max(_E, _e)
        _emins.append(_e)
ULIM = _U
# Global error scale: the largest reference-resolution error over all geometries.
# The reference is the coarsest RESOLVED resolution (Nx=16), matching the fit range;
# Nx=8 exceeds it everywhere and is marked clipped.
_ELIMS = {}
for _k, _t in GEOMS:
    _p = os.path.join(RAW, f"{_k}_16", f"{_k}_16.vts")
    if os.path.isfile(_p):
        _, _, _a = read_vts(_p)
        _ELIMS[_t] = float(np.abs(_a["U-U_exact"][_a["phi"] <= 0]).max())
ELIM = max(_ELIMS.values())
_worst = max(_ELIMS, key=_ELIMS.get)
print(f"global U scale:     +-{ULIM:.4f}")
print(f"global error scale: +-{ELIM:.4e}   (set by {_worst}, the worst case)")
for _t, _v in _ELIMS.items():
    print(f"    {_t:7s} Nx=16 error {_v:.3e}  = {100*_v/ELIM:5.1f}% of the shared scale")

for key, title in GEOMS:
    data = {}
    for nx in NXS:
        p = os.path.join(RAW, f"{key}_{nx}", f"{key}_{nx}.vts")
        if not os.path.isfile(p):
            print(f"  missing {p}"); continue
        _, _, a = read_vts(p)
        data[nx] = a
    if not data:
        continue

    ax_, idx = best_slice(data[NXS[0]]["phi"])
    axis_name = "yz xz xy".split()[ax_]

    n = len(data)
    fig, axes = plt.subplots(n, 2, figsize=(7.4, 3.05 * n))
    if n == 1:
        axes = axes[None, :]

    for r, nx in enumerate([k for k in NXS if k in data]):
        a = data[nx]
        phi_s = take(a["phi"], ax_, idx)
        u_s = np.where(phi_s <= 0, take(a["U"], ax_, idx), np.nan)
        e_s = np.where(phi_s <= 0, take(a["U-U_exact"], ax_, idx), np.nan)
        # L_inf over the full 3D interior (matches the convergence tables);
        # the slice max is kept only to decide whether the panel visibly clips.
        elim = float(np.abs(a["U-U_exact"][a["phi"] <= 0]).max())
        eslice = float(np.nanmax(np.abs(e_s)))

        for c, (fld, lab) in enumerate(((u_s, "$U$"), (e_s, "$U - U_{exact}$"))):
            ax = axes[r, c]
            if c == 0:
                im_u = ax.imshow(fld.T, origin="lower", cmap="RdBu_r",
                                 vmin=-ULIM, vmax=ULIM, interpolation="nearest")
            else:
                im_e = ax.imshow(fld.T, origin="lower", cmap="RdBu_r",
                                 vmin=-ELIM, vmax=ELIM, interpolation="nearest")
            ax.contour(phi_s.T, levels=[0.0], colors="k", linewidths=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(lab, fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(f"$N_x = {nx}$", fontsize=10, labelpad=8)
            else:
                clipped = "  (clipped)" if eslice > ELIM * 1.01 else ""
                ax.text(0.97, 0.03, f"$L_\\infty$ = {elim:.2e}{clipped}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    fig.tight_layout(rect=(0, 0.055, 1, 1.0))
    # One horizontal colourbar UNDER each column. They were previously stacked
    # vertically at the right, which read as "top rows use one scale, bottom rows
    # another" -- they apply to columns, so they belong under columns.
    b0 = axes[-1, 0].get_position()
    b1 = axes[-1, 1].get_position()
    cb_u = fig.colorbar(im_u, cax=fig.add_axes([b0.x0, 0.028, b0.width, 0.009]),
                        orientation="horizontal")
    cb_u.set_label("$U$", fontsize=9)
    cb_e = fig.colorbar(im_e, cax=fig.add_axes([b1.x0, 0.028, b1.width, 0.009]),
                        orientation="horizontal")
    cb_e.set_label("$U - U_{exact}$", fontsize=9)
    for cb in (cb_u, cb_e):
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)
        cb.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    d = os.path.join(OUT, key.replace("_Robin3", "3").replace("_Robin", "").lower() or "sphere")
    d = os.path.join(OUT, {"sphere_Robin": "sphere", "star_Robin": "star",
                           "star_Robin3": "star3"}[key])
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{key}_fields")
    for ext in ("png",):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png   ({axis_name} slice, index {idx})")
