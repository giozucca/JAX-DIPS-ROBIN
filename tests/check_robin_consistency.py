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
    "--checkpoint",
    default=None,
    help="path to a checkpoint file or a checkpoints/ dir. If given, the TRAINED network is "
    "substituted instead of the exact solution, so the printed residual is what training "
    "actually achieved. Compare it against the exact-solution run to see how much of the "
    "error is unconverged optimization vs the discretization floor.",
)
parser.add_argument("--hidden-layers-m", type=int, default=None, help="override; else read from the run's .hydra/config.yaml")
parser.add_argument("--hidden-dim-m", type=int, default=None, help="override; else read from the run's .hydra/config.yaml")
parser.add_argument("--hidden-layers-p", type=int, default=None, help="override; else read from the run's .hydra/config.yaml")
parser.add_argument("--hidden-dim-p", type=int, default=None, help="override; else read from the run's .hydra/config.yaml")
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

# jax_enable_x64 alone does NOT make this codebase run in float64: util.py hardcodes
# `f32 = jnp.float32`, and mesh.py / geometric_integrations_per_point.py /
# discretization.py all pass `dtype=f32` explicitly, so every array is cast straight
# back down to single precision. Patch the shared alias BEFORE importing those
# modules -- they all bind it at import time via `from ...util import f32` or
# `f32 = util.f32`. (jax_dips/__init__.py is empty, so nothing binds it earlier.)
from jax_dips._jaxmd_modules import util as _jaxdips_util  # noqa: E402

if args.x64:
    _jaxdips_util.f32 = jnp.float64

f32 = _jaxdips_util.f32

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

    # Disable the learned preconditioner so the residual measures the
    # discretization (and, with --checkpoint, the trained network) and nothing else.
    disc.precond_fn = lambda params, coeffs: 1.0

    if args.checkpoint is None:
        # Substitute the EXACT solution: this is the discretization floor.
        disc.solution_at_point_fn = lambda params, r_point, phi_point: evaluate_exact_solution_fn(r_point)
    else:
        trained = _load_trained_solution_fn(args.checkpoint)
        disc.solution_at_point_fn = lambda params, r_point, phi_point: trained(r_point, phi_point)

    return disc, gstate, evaluate_exact_solution_fn


def _find_run_config(start):
    """Walk up from the checkpoint looking for the Hydra config Hydra saved for that run."""
    d = os.path.abspath(start if os.path.isdir(start) else os.path.dirname(start))
    for _ in range(6):
        cand = os.path.join(d, ".hydra", "config.yaml")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _load_trained_solution_fn(path):
    """Rebuild the MLP from a checkpoint and return f(r, phi) -> scalar."""
    import pickle

    import jax

    from jax_dips.nn.configure import get_model

    ckpt_root = path
    if os.path.isdir(path):
        cands = [p for p in os.listdir(path) if "checkpoint_" in p]
        if not cands:
            raise FileNotFoundError(f"no checkpoint_* files in {path}")
        path = os.path.join(path, max(cands, key=lambda p: int(p.rsplit("_", 1)[-1])))
    print(f"  checkpoint : {path}")
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(f"  ckpt epoch : {state.get('epoch')}   ckpt resolution: {state.get('resolution')}")
    params = state["params"]

    # The saved params carry a "preconditioner" entry that the bare model does not.
    try:
        import flax

        params = flax.core.unfreeze(params)
    except Exception:
        pass
    if isinstance(params, dict) and "preconditioner" in params:
        params = {k: v for k, v in params.items() if k != "preconditioner"}

    # Model dims must match the trained network exactly. Prefer the config Hydra
    # saved next to the run over anything typed on the command line.
    mlp_cfg = {"hidden_layers_m": 1, "hidden_dim_m": 200, "hidden_layers_p": 0, "hidden_dim_p": 0}
    cfg_path = _find_run_config(ckpt_root)
    if cfg_path is not None:
        try:
            from omegaconf import OmegaConf

            _c = OmegaConf.load(cfg_path)
            mlp_cfg = {k: int(_c.model.mlp[k]) for k in mlp_cfg}
            print(f"  model dims : {mlp_cfg}  (read from {cfg_path})")
        except Exception as e:  # noqa: BLE001
            print(f"  model dims : could not read {cfg_path} ({e}); using defaults {mlp_cfg}")
    else:
        print(f"  model dims : no .hydra/config.yaml found near the checkpoint; using {mlp_cfg}")

    _overrides = {
        "hidden_layers_m": args.hidden_layers_m,
        "hidden_dim_m": args.hidden_dim_m,
        "hidden_layers_p": args.hidden_layers_p,
        "hidden_dim_p": args.hidden_dim_p,
    }
    for k, v in _overrides.items():
        if v is not None:
            mlp_cfg[k] = v
            print(f"  model dims : {k} overridden from the command line -> {v}")

    model_dict = {
        "name": None,
        "model_type": "mlp",
        "mlp": {
            "hidden_layers_m": mlp_cfg["hidden_layers_m"],
            "hidden_dim_m": mlp_cfg["hidden_dim_m"],
            "activation_m": "jnp.tanh",
            "hidden_layers_p": mlp_cfg["hidden_layers_p"],
            "hidden_dim_p": mlp_cfg["hidden_dim_p"],
            "activation_p": "jnp.tanh",
        },
        "resnet": {
            "res_blocks_m": 3,
            "res_dim_m": 40,
            "activation_m": "jnp.tanh",
            "res_blocks_p": 0,
            "res_dim_p": 0,
            "activation_p": "jnp.tanh",
        },
    }
    forward, _framework = get_model(model_dict, model_type="mlp")

    # Fail loudly and specifically if the declared dims do not match the checkpoint.
    probe = forward.init(jax.random.PRNGKey(0), x=jnp.zeros(3, dtype=f32), phi_x=f32(0.1))

    def _shapes(tree):
        return {
            f"{mod}/{name}": tuple(arr.shape)
            for mod, sub in dict(tree).items()
            for name, arr in dict(sub).items()
        }

    want, got = _shapes(probe), _shapes(params)
    if want != got:
        only_want = {k: v for k, v in want.items() if got.get(k) != v}
        only_got = {k: v for k, v in got.items() if want.get(k) != v}
        raise ValueError(
            "Checkpoint does not match the declared model dims.\n"
            f"  model built from : {mlp_cfg}\n"
            f"  expected shapes  : {only_want}\n"
            f"  checkpoint has   : {only_got}\n"
            "Pass the right --hidden-layers-m / --hidden-dim-m (etc.) for this run."
        )

    def solution_fn(r_point, phi_point):
        return forward.apply(params, None, r_point, phi_point).reshape()

    return solution_fn


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

    # --- small-cell (sliver) test -------------------------------------------
    # coeffs_[:12] are the mu*A/dx face coefficients, coeffs_[12] is V_m.
    # If the worst residuals sit on cells whose volume/face-areas have collapsed,
    # this is the classic cut-cell small-cell problem rather than plain truncation.
    coeffs_ = vmap(disc.compute_face_centroids_values_plus_minus_at_point, (0, None, None, None))(
        points, dx, dx, dx
    )
    V_m = coeffs_[:, 12]
    sum_coeffs = sum(coeffs_[:, k] for k in (0, 2, 4, 6, 8, 10))
    vol_frac = V_m / (dx**3)

    order = jnp.argsort(jnp.where(in_minus, jnp.abs(res), -1.0))[::-1][:5]
    worst = [
        {
            "res": float(jnp.abs(res)[i]),
            "vol_frac": float(vol_frac[i]),
            "sum_coeffs": float(sum_coeffs[i]),
        }
        for i in order
    ]
    # median volume fraction among interface cells, for scale
    _ifc = in_minus & crossed
    med_vol_frac_iface = float(jnp.median(jnp.where(_ifc, vol_frac, jnp.nan)[~jnp.isnan(jnp.where(_ifc, vol_frac, jnp.nan))])) if int(jnp.sum(_ifc)) else float("nan")
    min_vol_frac_iface = float(jnp.min(jnp.where(_ifc, vol_frac, jnp.inf))) if int(jnp.sum(_ifc)) else float("nan")

    return {
        "dtype": str(res.dtype),
        "worst": worst,
        "med_vol_frac_iface": med_vol_frac_iface,
        "min_vol_frac_iface": min_vol_frac_iface,
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
    if args.checkpoint is None:
        print(f"  u          : EXACT solution substituted (no neural network)\n")
    else:
        print(f"  u          : TRAINED network from checkpoint "
              f"(mlp_m {args.hidden_layers_m}x{args.hidden_dim_m})\n")

    header = (
        f"{'Nx':>5} {'zoom':>5} {'dx':>10} "
        f"{'RMS res (O-)':>14} {'max res (O-)':>14} {'RMS res (iface)':>16} "
        f"{'#O-':>8} {'#iface':>8} {'BC share':>9} {'dtype':>9}"
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
                f"{s['n_minus']:>8} {s['n_iface']:>8} {bc_share:>8.2f}% {s['dtype']:>9}"
            )
            if args.x64 and s["dtype"] != "float64":
                raise RuntimeError(
                    f"--x64 requested but the residual came back as {s['dtype']}. "
                    "Something is still forcing single precision; the numbers below are NOT float64."
                )
            print(
                f"        interface cell volume fraction V_m/dx^3:  "
                f"median={s['med_vol_frac_iface']:.4f}  min={s['min_vol_frac_iface']:.2e}"
            )
            print(f"        5 worst Omega- cells (|res|, V_m/dx^3, sum of face coeffs):")
            for w in s["worst"]:
                print(
                    f"            |res|={w['res']:.4e}   V_m/dx^3={w['vol_frac']:.4e}   "
                    f"sum_coeffs={w['sum_coeffs']:.4e}"
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
