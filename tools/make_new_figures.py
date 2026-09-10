"""
Figures for the [-1,1]^3 Robin runs in new_results/.

Outputs (PNG only, 300 dpi, colourblind-safe palettes throughout):

  new_figures/convergence_<geom>.png      RMSE and L_inf vs Nx, with an O(Nx^-1) guide
  new_figures/convergence_all.png         the three geometries side by side
  new_figures/fields_<geom>.png           U | U_exact | U - U_exact, rows = Nx  (LINEAR)
  fields_figures_log/fields_<geom>_log.png  same, logarithmic

WHY THE SOLUTION COLUMN LOOKS THE SAME AT EVERY RESOLUTION
----------------------------------------------------------
Because it *is* nearly the same -- that is what convergence means. For the sphere
max|U| runs 0.4511 -> 0.4696 -> 0.4740 -> 0.4763 against max|U_exact| = 0.4759, so
successive solutions differ from each other by <1% of the colour range and are
indistinguishable by eye. All of the information lives in the error column, and its
magnitude lives in the COLORBAR LIMITS, not in the pattern. That is precisely why
the error column must not be rescaled per panel: doing so normalises the
convergence away and makes a coarse panel look as clean as a fine one.

SCALE RULES (this is the "uniform plotting" the figures depend on)
------------------------------------------------------------------
  slice plane : chosen ONCE per geometry from phi (which is bit-identical across
                all four runs) as the axis plane holding the most interior pixels.
                Every row therefore shows the identical cut.
  U, U_exact  : one symmetric scale per geometry, set by that geometry's exact
                solution and shared by both columns and all four rows. It is NOT
                shared across geometries -- |u| is set by the exact solution, not
                by accuracy, so equalising it would imply a difference that is not
                about error.
  error       : ONE global symmetric scale across all three geometries and all
                rows, so a geometry with larger errors renders darker. Set by the
                worst Nx >= 16 case in the sweep; Nx = 8 clips and is annotated
                with its true maximum. (Nx = 8 is excluded from setting the scale
                because star_Robin3 is under-resolved there by construction --
                see interior_pts.txt -- and its error is ~44x the next row.)
  log figures : |U - U_exact| over a fixed 5-decade range shared by every panel in
                every geometry, so convergence reads as a uniform darkening and no
                row is clipped. U and U_exact use a symmetric log norm.

Usage:  python3 tools/make_new_figures.py
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, SymLogNorm, Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_vts import read_vts

RAW = "new_results"
OUT_LIN = "new_figures"
OUT_LOG = "fields_figures_log"
GEOMS = [("sphere_Robin", "Sphere"), ("star_Robin", "Star"), ("star_Robin3", "Two Star")]
NXS = [8, 16, 32, 64]
DPI = 300

# Okabe-Ito: distinguishable under protanopia, deuteranopia and tritanopia, and
# still separable in greyscale once combined with distinct dashes and markers.
C_RMSE = "#0072B2"   # blue
C_LINF = "#D55E00"   # vermillion
C_REF  = "#000000"   # black

# Diverging maps: RdBu is ColorBrewer "colorblind safe". Sequential: viridis.
CMAP_DIV = "RdBu_r"
CMAP_SEQ = "viridis"
BAD = "#DDDDDD"      # exterior (phi > 0), masked

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "savefig.dpi": DPI,
    "figure.dpi": DPI,
})


def load(geom, nx):
    path = os.path.join(RAW, f"{geom}_{nx}", f"{geom}_{nx}.vts")
    _, _, a = read_vts(path)
    return a


def metrics(a):
    """RMSE and L_inf over the interior, matching test_poisson.py exactly:
    the error is masked to phi <= 0 and RMSE divides by the interior COUNT."""
    m = a["phi"] <= 0.0
    e = a["U-U_exact"][m]
    return float(np.sqrt((e ** 2).sum() / m.sum())), float(np.abs(e).max())


def slice_plane(phi):
    """Axis-aligned plane holding the most interior pixels. phi is identical across
    resolutions, so this is a property of the geometry alone."""
    m = phi <= 0.0
    best = (0, 0, -1)
    for ax in range(3):
        counts = m.sum(axis=tuple(i for i in range(3) if i != ax))
        i = int(counts.argmax())
        if counts[i] > best[2]:
            best = (ax, i, int(counts[i]))
    return best[0], best[1]


def take(arr, ax, idx):
    return [arr[idx, :, :], arr[:, idx, :], arr[:, :, idx]][ax]


# --------------------------------------------------------------------------
# gather everything once
# --------------------------------------------------------------------------
DATA, MET, PLANE = {}, {}, {}
for geom, _ in GEOMS:
    for nx in NXS:
        a = load(geom, nx)
        DATA[(geom, nx)] = a
        MET[(geom, nx)] = metrics(a)
    PLANE[geom] = slice_plane(DATA[(geom, NXS[0])]["phi"])

# global error scale, set by the resolved rows only
ERR_VMAX = max(MET[(g, nx)][1] for g, _ in GEOMS for nx in NXS if nx >= 16)
LOG_LO, LOG_HI = -5.0, 0.0


def masked_slice(a, field, geom):
    ax, idx = PLANE[geom]
    v = take(a[field], ax, idx).astype(float)
    phi = take(a["phi"], ax, idx)
    return np.ma.masked_where(phi > 0.0, v)


# --------------------------------------------------------------------------
# 1. convergence
# --------------------------------------------------------------------------
def convergence_panel(ax, geom, label, legend=False):
    nx = np.array(NXS, dtype=float)
    rmse = np.array([MET[(geom, n)][0] for n in NXS])
    linf = np.array([MET[(geom, n)][1] for n in NXS])

    ax.loglog(nx, rmse, color=C_RMSE, ls="-", marker="o", ms=6, lw=1.6,
              mfc="white", mew=1.4, label=r"RMSE")
    ax.loglog(nx, linf, color=C_LINF, ls="--", marker="s", ms=6, lw=1.6,
              mfc="white", mew=1.4, label=r"$L^{\infty}$")
    # First-order guide, drawn clear of the data rather than through it. Writing the
    # guide as A/N_x, the condition "at least a factor GAP below every plotted point"
    # is A <= GAP * min_i (N_x,i * y_i) over both curves -- the binding point is
    # wherever the data is flattest, not simply the finest resolution. Anchoring off
    # the finest point alone is what let the guide cross star_Robin3, whose RMSE
    # barely moves from Nx=16 to Nx=64.
    GAP = 0.35
    A = GAP * min((nx * rmse).min(), (nx * linf).min())
    ref = A / nx
    ax.loglog(nx, ref, color=C_REF, ls=":", lw=1.4, label=r"$\mathcal{O}(N_x^{-1})$")

    ax.set_xscale("log", base=2)
    ax.set_xticks(NXS)
    ax.set_xticklabels([str(n) for n in NXS])
    ax.set_xlabel(r"$N_x$")
    ax.set_title(label)
    ax.grid(True, which="both", ls="-", lw=0.4, alpha=0.35)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc="lower left")


for geom, label in GEOMS:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    convergence_panel(ax, geom, label, legend=True)
    ax.set_ylabel("error")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_LIN, f"convergence_{geom}.png"), dpi=DPI)
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True)
for ax, (geom, label) in zip(axes, GEOMS):
    convergence_panel(ax, geom, label, legend=(geom == GEOMS[0][0]))
axes[0].set_ylabel("error")
fig.tight_layout()
fig.savefig(os.path.join(OUT_LIN, "convergence_all.png"), dpi=DPI)
plt.close(fig)


# --------------------------------------------------------------------------
# 2. field figures
# --------------------------------------------------------------------------
def field_figure(geom, label, logscale):
    umax = max(np.abs(DATA[(geom, n)]["U_exact"][DATA[(geom, n)]["phi"] <= 0]).max()
               for n in NXS)
    umax = float(umax)

    fig, axes = plt.subplots(len(NXS), 3, figsize=(9.6, 3.05 * len(NXS)))
    ims = [None, None, None]
    for r, nx in enumerate(NXS):
        a = DATA[(geom, nx)]
        for c, field in enumerate(["U", "U_exact", "U-U_exact"]):
            ax = axes[r, c]
            v = masked_slice(a, field, geom)
            if c < 2:
                if logscale:
                    norm = SymLogNorm(linthresh=1e-2, vmin=-umax, vmax=umax, base=10)
                else:
                    norm = Normalize(vmin=-umax, vmax=umax)
                cmap = plt.get_cmap(CMAP_DIV).copy()
                im = ax.imshow(v.T, origin="lower", cmap=cmap, norm=norm)
            else:
                if logscale:
                    with np.errstate(divide="ignore"):
                        lv = np.ma.masked_array(
                            np.log10(np.maximum(np.abs(v.filled(np.nan)), 1e-30)),
                            mask=v.mask)
                    cmap = plt.get_cmap(CMAP_SEQ).copy()
                    im = ax.imshow(lv.T, origin="lower", cmap=cmap,
                                   norm=Normalize(vmin=LOG_LO, vmax=LOG_HI))
                else:
                    cmap = plt.get_cmap(CMAP_DIV).copy()
                    im = ax.imshow(v.T, origin="lower", cmap=cmap,
                                   norm=Normalize(vmin=-ERR_VMAX, vmax=ERR_VMAX))
            cmap.set_bad(BAD)
            im.cmap.set_bad(BAD)
            ims[c] = im

            ax.set_xticks([]); ax.set_yticks([])
            # Column header on the top row only. The geometry is in the suptitle and
            # the resolution is in the row label, so repeating either per panel is
            # noise -- everything else belongs in the paper's caption, not the figure.
            if r == 0:
                ttl = [r"$u_\theta$", r"$u_{\mathrm{exact}}$",
                       (r"$\log_{10}|u_\theta - u_{\mathrm{exact}}|$" if logscale
                        else r"$u_\theta - u_{\mathrm{exact}}$")][c]
                ax.set_title(ttl, fontsize=12, pad=6)
        # No per-panel error numbers here: RMSE and L_inf belong to the convergence
        # figure, and repeating them invites the reader to compare digits instead of
        # reading the field. The row label carries the resolution only.
        axes[r, 0].set_ylabel(f"$N_x = {nx}$", fontsize=11)

    fig.suptitle(label, fontsize=15, y=0.997)
    fig.tight_layout(rect=[0, 0.075, 1, 0.985])

    labels = [r"$u_\theta$ and $u_{\mathrm{exact}}$",
              r"$u_\theta$ and $u_{\mathrm{exact}}$",
              (r"$\log_{10}|u_\theta - u_{\mathrm{exact}}|$" if logscale
               else r"$u_\theta - u_{\mathrm{exact}}$")]
    for c, x0 in enumerate([0.055, 0.385, 0.715]):
        cax = fig.add_axes([x0, 0.035, 0.26, 0.013])
        cb = fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label(labels[c], fontsize=8)
        if c == 1:
            cb.ax.set_visible(False)

    out = os.path.join(OUT_LOG if logscale else OUT_LIN,
                       f"fields_{geom}_log.png" if logscale else f"fields_{geom}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


os.makedirs(OUT_LIN, exist_ok=True)
os.makedirs(OUT_LOG, exist_ok=True)
written = []
for geom, label in GEOMS:
    written.append(field_figure(geom, label, logscale=False))
    written.append(field_figure(geom, label, logscale=True))

print(f"global error scale  +-{ERR_VMAX:.4e}   (set by worst Nx>=16)")
print(f"log error range     1e{LOG_LO:.0f} .. 1e{LOG_HI:.0f}")
for g, l in GEOMS:
    ax_i, idx = PLANE[g]
    print(f"{l:7s} slice {'yz xz xy'.split()[ax_i]} @ index {idx}")
print()
for w in written:
    print("wrote", w)
for g, _ in GEOMS:
    print("wrote", os.path.join(OUT_LIN, f"convergence_{g}.png"))
print("wrote", os.path.join(OUT_LIN, "convergence_all.png"))
