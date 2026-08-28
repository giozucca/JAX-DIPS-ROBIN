"""
Discretization consistency check for the Robin solver.

This script answers one question, with the neural network and the optimizer removed
from the picture entirely:

    If we substitute the EXACT solution into the discrete Robin operator, does the
    residual ||A u_exact - b|| go to zero as the grid is refined?

If it does, the discretization is consistent and any error plateau/blow-up in a
training run is an optimization or loss-weighting problem.
If it does NOT shrink with h, the discrete equations themselves have the wrong
solution, and no amount of training or model capacity can fix it.

Usage (from the repo root):

    python tests/check_robin_consistency.py --exp sphere_Robin --nx 16 32 64
    python tests/check_robin_consistency.py --exp sphere_Robin --nx 16 32 --x64
    python tests/check_robin_consistency.py --exp star_Robin  --nx 16 32

--x64 reruns in double precision. Comparing float32 vs float64 separates genuine
truncation error from float32 round-off in the geometric integration.

The "zoom" column mirrors TrainData.alternate_res_sequentially, which shrinks the
finite-volume stencil to dx/2**zoom during training (zoom reaches 3 in the last
quarter of a run). zoom=0 is the honest finite-volume cell; zoom=3 is what the
network is actually being trained against at the end.
"""

import argparse
import os
import sys

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:
    sys.path.append(rootDir)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--exp", default="sphere_Robin", choices=["sphere_Robin", "star_Robin", "star_Robin3"])
parser.add_argument("--nx", type=int, nargs="+", default=[16, 32], help="training-grid resolutions to test")
parser.add_argument("--zoom", type=int, nargs="+", default=[0, 3], help="stencil zoom levels (dx -> dx/2**zoom)")
parser.add_argument("--x64", action="store_true", help="run in float64 instead of float32")
parser.add_argument(
    "--cpu",
    action="store_true",
    help="force the CPU backend; this check is small and does not need the GPU, "
    "so use this if a training run is holding the card",
)
args = parser.parse_args()

# Must be set BEFORE jax is imported. Without the platform allocator, jax
# preallocates most of the VRAM on import and cuDNN can then fail to initialize
# ("DNN library initialization failed"). Every other script in this repo does this.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
if args.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"

try:
    from jax.config import config
except ImportError:
    from jax import config

config.update("jax_enable_x64", args.x64)

from jax import numpy as jnp  # noqa: E402
from jax import vmap  # noqa: E402

from jax_dips._jaxmd_modules.util import f32  # noqa: E402
from jax_dips.domain import mesh  # noqa: E402
from jax_dips.solvers.poisson.discretization import Discretization  # noqa: E402
from jax_dips.solvers.simulation_states import PoissonSimStateFn  # noqa: E402
from tests.confs import experiment_configs  # noqa: E402


class _StandaloneDiscretization(Discretization):
    """`Discretization` is effectively abstract: its __init__ binds four
    post-processing hooks (`*_neural_network`) that only `Trainer` defines, so the
    base class cannot be instantiated on its own. Those hooks compute solution
    gradients for visualization and are never reached by `compute_Ax_and_b_*`,
    so stubbing them is enough to build the operator standalone.
    """

    def compute_normal_gradient_solution_mp_on_interface_neural_network(self, *a, **kw):
        raise NotImplementedError("not needed for the consistency check")

    def compute_gradient_solution_mp_neural_network(self, *a, **kw):
        raise NotImplementedError("not needed for the consistency check")

    def compute_normal_gradient_solution_on_interface_neural_network(self, *a, **kw):
        raise NotImplementedError("not needed for the consistency check")

    def compute_gradient_solution_neural_network(self, *a, **kw):
        raise NotImplementedError("not needed for the consistency check")


def domain_for(exp_name):
    """Mirrors the domain selection in tests/test_poisson.py.

    Plain Python floats on purpose: wrapping these in f32() makes them device
    arrays, which forces GPU work before we actually need any.
    """
    if exp_name == "star_Robin3":
        return -2.1, 2.1
    if exp_name == "star_Robin":
        return -1.8, 1.8
    return -1.0, 1.0


def build_discretization(exp_name, Nx, lo, hi):
    (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alphaRobin_fn,
        _exact_sol_m_fn,
        _exact_sol_p_fn,
        evaluate_exact_solution_fn,
        g_m_fn,
        g_p_fn,
        beta_fn,
    ) = getattr(experiment_configs, exp_name)()

    # Plain int, not i32(3): mesh.construct does `if dimension == 3`, and a device
    # array there turns a Python branch into a GPU op.
    init_mesh_fn, _ = mesh.construct(3)
    xc = jnp.linspace(lo, hi, Nx, dtype=f32)
    gstate = init_mesh_fn(xc, xc, xc)

    def zero_op(u):
        return 0.0

    # Field order must match trainer.setup_Robin / PoissonSimStateFn.
    sim_state_fn = PoissonSimStateFn(
        vmap(initial_value_fn),
        vmap(dirichlet_bc_fn),
        vmap(phi_fn),
        vmap(mu_m_fn),
        vmap(mu_p_fn),
        vmap(k_m_fn),
        vmap(k_p_fn),
        vmap(f_m_fn),
        vmap(f_p_fn),
        vmap(g_m_fn),
        vmap(g_p_fn),
        vmap(alphaRobin_fn),
        vmap(beta_fn),
        zero_op,
        zero_op,
    )

    disc = _StandaloneDiscretization(
        lvl_gstate=gstate,
        sim_state=None,
        sim_state_fn=sim_state_fn,
        precondition=1,
        algorithm=0,
    )

    # Replace the neural network by the exact solution, and disable the learned
    # preconditioner, so the residual measures the discretization and nothing else.
    disc.solution_at_point_fn = lambda params, r_point, phi_point: evaluate_exact_solution_fn(r_point)
    disc.precond_fn = lambda params, coeffs: 1.0

    return disc, gstate, evaluate_exact_solution_fn


def residual_stats(disc, gstate, dx):
    points = gstate.R
    lhs_rhs = vmap(disc.compute_Ax_and_b_fn, (None, 0, None, None, None))(None, points, dx, dx, dx)
    lhs_rhs = lhs_rhs.reshape(points.shape[0], 2)
    res = lhs_rhs[:, 0] - lhs_rhs[:, 1]

    phi = disc.phi_interp_fn(points).reshape(-1)
    in_minus = phi <= 0.0
    crossed = vmap(disc.is_cell_crossed_by_interface, (0, None, None, None))(points, dx, dx, dx).reshape(-1) == 0

    def rms(mask):
        n = jnp.sum(mask)
        return jnp.sqrt(jnp.sum(jnp.where(mask, res**2, 0.0)) / jnp.maximum(n, 1)), n

    rms_minus, n_minus = rms(in_minus)
    rms_iface, n_iface = rms(in_minus & crossed)
    max_minus = jnp.max(jnp.where(in_minus, jnp.abs(res), 0.0))
    return {
        "rms_minus": float(rms_minus),
        "max_minus": float(max_minus),
        "rms_iface": float(rms_iface),
        "n_minus": int(n_minus),
        "n_iface": int(n_iface),
        "n_total": int(points.shape[0]),
    }


def main():
    lo, hi = domain_for(args.exp)
    precision = "float64" if args.x64 else "float32"
    print(f"\nRobin discretization consistency check")
    print(f"  experiment : {args.exp}")
    print(f"  domain     : [{float(lo)}, {float(hi)}]^3")
    print(f"  precision  : {precision}")
    print(f"  u          : EXACT solution substituted (no neural network)\n")

    header = (
        f"{'Nx':>5} {'zoom':>5} {'dx':>10} "
        f"{'RMS res (O-)':>14} {'max res (O-)':>14} {'RMS res (iface)':>16} "
        f"{'#O-':>8} {'#iface':>8} {'BC share':>9}"
    )
    print(header)
    print("-" * len(header))

    for Nx in args.nx:
        disc, gstate, _ = build_discretization(args.exp, Nx, lo, hi)
        for zoom in args.zoom:
            dx = gstate.dx * 0.5**zoom
            s = residual_stats(disc, gstate, dx)
            bc_share = 100.0 * s["n_iface"] / max(s["n_total"], 1)
            print(
                f"{Nx:>5} {zoom:>5} {float(dx):>10.5f} "
                f"{s['rms_minus']:>14.4e} {s['max_minus']:>14.4e} {s['rms_iface']:>16.4e} "
                f"{s['n_minus']:>8} {s['n_iface']:>8} {bc_share:>8.2f}%"
            )
        print()

    print("How to read this:")
    print("  * RMS res (O-) should DROP as Nx doubles at fixed zoom. If it is flat, the")
    print("    discrete equations do not converge to the PDE and training cannot fix it.")
    print("  * Compare zoom=0 vs zoom=3 at the same Nx: zoom=3 is the stencil the network")
    print("    is actually trained against during the final quarter of a run.")
    print("  * 'BC share' is the fraction of training points whose cell is crossed by the")
    print("    interface, i.e. the only points that carry any Robin boundary information.")
    print("  * Rerun with --x64: if the float32 numbers are much worse, the limit is")
    print("    round-off in the geometric integration, not the scheme itself.\n")


if __name__ == "__main__":
    main()
