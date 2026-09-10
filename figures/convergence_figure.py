"""
Robin-BC convergence figures for JAX-DIPS-ROBIN.

One single-panel figure per geometry, drawn in the visual language of the JAX-DIPS
paper's convergence figure: RMSE as a blue solid line with filled diamonds, L^inf
as a red dash-dot line with filled circles, and a single black dashed O(N_x^-1)
guide. Serif type, in-panel legend, ticks inward on all four sides.

NB: the reference figure labels its series "regress d_n" because the original
jump-condition solver recovers u+ and u- at the interface by regression
(algorithm=0 -> get_u_mp_by_regression_at_point_fn). That qualifier does NOT apply
here. The Robin assembly path, compute_Ax_and_b_preconditioned_fn_Robin, never
calls self.u_mp_fn -- it reads the network directly via solution_at_point_fn with
phi forced negative. A Robin problem has no jump to regress across, so the series
are labelled plainly.

Writes, relative to this file:
    sphere/sphere_convergence.pdf | .png
    star/star_convergence.pdf     | .png
    star3/star3_convergence.pdf   | .png

Regenerate with:  python3 convergence_figure.py
"""
import math
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Convergence orders are fitted over the RESOLVED range: resolutions at which the
# interface is covered by at least this many cells per feature radius. The rule is
# evaluated on the geometry alone -- it never looks at the errors -- and it excludes
# exactly Nx=8, identically for all three shapes (r/dx there is 2.00 / 2.63 / 1.52,
# against 4.00 / 5.26 / 3.05 at Nx=16). Below it the discrete system is
# underdetermined: Star3 at Nx=8 has only 18 grid points inside Omega- against 501+
# network parameters, so training reaches machine-zero residual while the solution
# is still far off. Every data point is plotted; only the fit is restricted.
MIN_CELLS_PER_FEATURE = 3.0

# (folder, title, L, feature_radius, {Nx: (L_inf, RMSE)})
# L is the edge length of the cubic training domain, so dx = L / Nx.
# All runs used an identical 2x100 tanh MLP and training configuration.
SHAPES = [
    ("sphere", "Sphere", 2.0, 0.500, {
        8:  (0.03249034285545349,   0.013483742251992226),
        16: (0.010589778423309326,  0.004905113484710455),
        32: (0.0036998987197875977, 0.001731810043565929),
        64: (0.002419576048851013,  0.001104802475310862)}),
    ("star", "Star", 3.6, 1.183, {
        8:  (0.18977278470993042,   0.04672547057271004),
        16: (0.02495896816253662,   0.008460689336061478),
        32: (0.011859238147735596,  0.004652589559555054),
        64: (0.00495070219039917,   0.0020633782260119915)}),
    ("star3", "Star3", 4.2, 0.800, {
        8:  (1.321732521057129,     0.6940196752548218),
        16: (0.05271536111831665,   0.016389884054660797),
        32: (0.013866782188415527,  0.005005416925996542),
        64: (0.007100999355316162,  0.003192652715370059)}),
]

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 13,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": 0.9,
    "xtick.color": "black", "ytick.color": "black",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.fontsize": 10, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_RMSE, C_LINF = "#0000ff", "#ff0000"

def fitted_order(ns, vs, L):
    """order = slope of log(err) vs log(dx); positive when converging"""
    x = np.log([L / n for n in ns]); y = np.log(vs)
    return float(np.polyfit(x, y, 1)[0])

here = os.path.dirname(os.path.abspath(__file__))

for folder, title, L, rad, d in SHAPES:
    nxs = sorted(d)
    linf = [d[n][0] for n in nxs]
    rmse = [d[n][1] for n in nxs]

    fig, ax = plt.subplots(figsize=(5.0, 4.8))

    # O(N_x^-1) guide as a loose upper envelope, as in the reference figure.
    # Only its gradient is meaningful; the offset is arbitrary.
    xr = np.array([float(nxs[0]), float(nxs[-1])])
    ax.plot(xr, 2.4 * max(linf) * (nxs[0] / xr), ls=(0, (6, 4)), lw=1.2,
            color="black", label=r"$\mathcal{O}(N_x^{-1})$", zorder=1)

    ax.plot(nxs, rmse, "-D", color=C_RMSE, lw=1.6, ms=6,
            mfc=C_RMSE, mec="black", mew=0.8,
            label="RMSE", zorder=3)
    ax.plot(nxs, linf, ls=(0, (7, 2, 1.5, 2)), marker="o", color=C_LINF,
            lw=1.6, ms=5.5, mfc=C_LINF, mec="black", mew=0.8,
            label=r"$L^{\infty}$", zorder=3)

    fit = [n for n in nxs if rad / (L / n) >= MIN_CELLS_PER_FEATURE]
    o_i = fitted_order(fit, [d[n][0] for n in fit], L)
    o_r = fitted_order(fit, [d[n][1] for n in fit], L)

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(nxs); ax.set_xticklabels([str(n) for n in nxs])
    ax.minorticks_off()
    ax.set_xlim(nxs[0] / 1.22, nxs[-1] * 1.22)
    lo, hi = min(min(rmse), min(linf)), max(max(rmse), max(linf))
    ax.set_ylim(10 ** (math.floor(math.log10(lo)) - 0.15),
                10 ** (math.ceil(math.log10(hi * 3.2))))
    ax.set_xlabel(r"$\mathrm{N_x}$")
    ax.set_ylabel("error")
    ax.set_title(title, pad=7)

    # data series before the guide, matching the reference figure's legend order
    h, l = ax.get_legend_handles_labels()
    idx = [1, 2, 0]
    ax.legend([h[i] for i in idx], [l[i] for i in idx],
              loc="upper right", handlelength=2.6, borderaxespad=0.7)

    fig.tight_layout()
    out = os.path.join(here, folder)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"{folder}_convergence")
    for ext in ("png",):
        fig.savefig(f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"{title:7s} order  L_inf {o_i:.2f}  RMSE {o_r:.2f}   ->  {folder}/{folder}_convergence.pdf|png")
