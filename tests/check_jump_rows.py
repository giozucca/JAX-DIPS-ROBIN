"""
Checks the ported jump rows (jump_scheme "fixed", jump_eval_once) against the NumPy replica, with no
network and no training. Run it on the runner before training:

    python tests/check_jump_rows.py                     # sphere_jaxrobin and star_jaxrobin, Nx = 16
    python tests/check_jump_rows.py --nx 8 16 --cpu

For each problem and grid it fills every node with a field u (the exact solution, then random values)
and compares three sets of rows, i.e. (lhs - rhs) / weight at every node:

  1. the per-point path, compute_Ax_and_b_preconditioned_fn, with the network replaced by a lookup
     of u at the nodes (each extrapolation reads its 27 values itself, as in training without
     jump_eval_once);
  2. the node-values path, compute_Ax_and_b_from_values_fn, reading u once per node (jump_eval_once);
  3. the replica, tools/jump_replica3d.py (fixed rules, Bias Slow, interface weights, same ell and
     floor), in float64.

1 = 2 shows the restructure changes nothing (agreement to float32 round-off, ~1e-6 relative). 2 = 3
shows the JAX rows are the replica's rows, which were shown second order with the exact least-squares
fit (agreement to float32 round-off, ~1e-5 relative). For the exact solution it also prints the
weighted residual of the rows split by kind (bulk, ghost, cut, wall), which should be small in the
bulk and O(1) per unit weight at cut and ghost rows, as in the replica.
"""
import argparse
import os
import sys

rootDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
for d in (rootDir, os.path.join(rootDir, "tools")):
    if d not in sys.path:
        sys.path.append(d)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nx", type=int, nargs="+", default=[16])
parser.add_argument("--problems", nargs="+", default=["sphere_jaxrobin", "star_jaxrobin"],
                    choices=["sphere_jaxrobin", "star_jaxrobin"])
parser.add_argument("--ell", type=float, default=1.0e4)
parser.add_argument("--cut-floor", type=float, default=0.05)
parser.add_argument("--cpu", action="store_true", help="force the CPU backend")
parser.add_argument("--diag", action="store_true",
                    help="at the worst row (exact solution), print JAX and replica geometry and extrapolation side by side")
args = parser.parse_args()

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
if args.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax import vmap  # noqa: E402

import jump_replica3d as replica  # noqa: E402
from jax_dips._jaxmd_modules.util import f32  # noqa: E402
from jax_dips.domain import mesh  # noqa: E402
from jax_dips.solvers.poisson.discretization import Discretization  # noqa: E402
from jax_dips.solvers.simulation_states import PoissonSimStateFn  # noqa: E402
from tests.confs import experiment_configs  # noqa: E402


class _StandaloneDiscretization(Discretization):
    """Stubs the Trainer-only hooks, as in tests/check_jump_extrapolation.py."""

    jump_scheme = "fixed"

    def compute_normal_gradient_solution_mp_on_interface_neural_network(self, *a, **kw):
        raise NotImplementedError

    def compute_gradient_solution_mp_neural_network(self, *a, **kw):
        raise NotImplementedError

    def compute_normal_gradient_solution_on_interface_neural_network(self, *a, **kw):
        raise NotImplementedError

    def compute_gradient_solution_neural_network(self, *a, **kw):
        raise NotImplementedError

    def precond_fn(self, params, coeffs):
        return 1.0


def replica_problem(name):
    if name == "sphere_jaxrobin":
        return replica.make_problem("paper", radius=2 / 3)
    return replica.make_star("jaxrobin")


def diagnose(disc, sysm, pb, N, row, h):
    """JAX and replica, side by side, at a row and its 6 neighbours (cut cells only for the extrapolation)."""
    X = sysm["X"].reshape(-1, 3)
    i, j, k = np.unravel_index(row, (N, N, N))
    print(f"   --- diagnosis at row {row} = node ({i},{j},{k}), x = {X[row].round(4)}")
    for off in [(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        a, b, c = i + off[0], j + off[1], k + off[2]
        n = np.ravel_multi_index((a, b, c), (N, N, N))
        p = jnp.asarray(X[n], dtype=f32)
        co = np.asarray(disc.compute_face_centroids_values_plus_minus_at_point(p, h, h, h))
        bint = float(np.asarray(disc.beta_integrate_over_interface_at_point(p, h, h, h)).reshape(()))
        area = float(np.asarray(disc.interface_area_at_point(p, h, h, h)).reshape(()))
        cut_jax = int(np.asarray(disc.is_cell_crossed_by_interface(p, h, h, h)).reshape(())) == 0
        line = (f"   node {off}: cut JAX {cut_jax} replica {bool(sysm['cut'][n])} | V- {co[12]:.5e} vs {sysm['Vm'][n]:.5e}"
                f" | face coeffs max diff {np.abs(co[0:12:2] - sysm['cm'][:, n]).max():.1e} (minus), "
                f"{np.abs(co[1:12:2] - sysm['cp'][:, n]).max():.1e} (plus)"
                f" | int beta: JAX stored {bint:+.5e}, replica -int beta {-sysm['BI'][n]:+.5e} | area {area:.4e} vs {sysm['AG'][n]:.4e}")
        print(line)
        if sysm["cut"][n]:
            node_plus, coef, const = disc._jump_fixed_extrapolation(p, h, h, h)
            side, rcoef, rconst = replica.extrapolation(pb, X[n], h, True, "slow")
            x0 = X[n]
            E = np.eye(3) * h
            g = np.array([pb["phi"](x0 + E[d]) - pb["phi"](x0 - E[d]) for d in range(3)]) / (2 * h)
            nrm = g / np.linalg.norm(g); dd = pb["phi"](x0) / np.linalg.norm(g); rp = x0 - dd * nrm
            rpj = jnp.asarray(rp[None], dtype=f32)
            print(f"      extrapolation: JAX node in Omega+ {bool(node_plus)}, replica extrapolates {'u-' if side == 'm' else 'u+'}"
                  f" | max coef diff {np.abs(np.asarray(coef) - rcoef).max():.2e} | const JAX {float(const):+.5e} replica {rconst:+.5e}")
            print(f"      at the projection {rp.round(5)}: alpha JAX {float(disc.alpha_interp_fn(rpj)[0]):+.5e} replica {pb['alpha'](rp):+.5e}"
                  f" | beta (JAX = -stored) {-float(disc.beta_interp_fn(rpj)[0]):+.5e} replica {pb['beta'](rp):+.5e}"
                  f" | mu- {float(disc.mu_m_interp_fn(rpj)[0]):.5e} vs {pb['mum'](rp):.5e} | mu+ {float(disc.mu_p_interp_fn(rpj)[0]):.5e} vs {pb['mup'](rp):.5e}")


def main():
    _StandaloneDiscretization.jump_ell = args.ell
    _StandaloneDiscretization.jump_cut_floor = args.cut_floor
    init_mesh_fn, _ = mesh.construct(3)
    for name in args.problems:
        (initial_value_fn, dirichlet_bc_fn, phi_fn, mu_m_fn, mu_p_fn, k_m_fn, k_p_fn, f_m_fn, f_p_fn,
         alpha_fn, beta_fn, exact_m, exact_p, exact) = getattr(experiment_configs, name)()
        zero_op = lambda u: 0.0
        sim_state_fn = PoissonSimStateFn(
            vmap(initial_value_fn), vmap(dirichlet_bc_fn), vmap(phi_fn), vmap(mu_m_fn), vmap(mu_p_fn),
            vmap(k_m_fn), vmap(k_p_fn), vmap(f_m_fn), vmap(f_p_fn),
            alpha_fn=vmap(alpha_fn), beta_fn=vmap(beta_fn), nonlinear_op_m=zero_op, nonlinear_op_p=zero_op)
        for N in args.nx:
            xc = jnp.linspace(-1.0, 1.0, N, dtype=f32)
            gstate = init_mesh_fn(xc, xc, xc)
            disc = _StandaloneDiscretization(lvl_gstate=gstate, sim_state=None, sim_state_fn=sim_state_fn,
                                             precondition=1, algorithm=0)
            disc.eval_once_origin = jnp.array([-1.0, -1.0, -1.0], dtype=f32)
            dx = gstate.dx
            R = gstate.R
            sysm = replica.build(N, replica_problem(name), True, alternate=True, bias="slow")
            A, b = replica.scaled(sysm, "interface", ell=args.ell, cut_floor=args.cut_floor)
            kind = np.where(sysm["wall"], "wall", np.where(sysm["AG"] > 0, "cut", np.where(sysm["A_ghost"] > 0, "ghost", "bulk")))

            fields = {"exact solution": np.asarray(vmap(exact)(R), dtype=np.float32),
                      "random field": np.random.default_rng(0).normal(size=N ** 3).astype(np.float32)}
            for label, u in fields.items():
                U_pad = jnp.pad(jnp.asarray(u).reshape(N, N, N), 2)

                def lookup(r):
                    ijk = jnp.round((r - disc.eval_once_origin) / dx).astype(jnp.int32) + 2
                    return U_pad[ijk[0], ijk[1], ijk[2]]

                disc.solution_at_point_fn = lambda params, r, phi: lookup(r)
                disc.evaluate_solution_fn = lambda params, pts: vmap(lookup)(pts)
                rows_old = np.asarray(vmap(disc.compute_Ax_and_b_preconditioned_fn, (None, 0, None, None, None))(None, R, dx, dx, dx))
                rows_new = np.asarray(vmap(disc.compute_Ax_and_b_from_values_fn, (None, 0, None, None, None))(U_pad, R, dx, dx, dx))
                r_old = (rows_old[:, 0] - rows_old[:, 1]).ravel().astype(np.float64)
                r_new = (rows_new[:, 0] - rows_new[:, 1]).ravel().astype(np.float64)
                r_rep = A @ u.astype(np.float64) - b
                scale = np.abs(r_rep).max()
                print(f"{name}, Nx = {N}, {label}: largest |row| {scale:.3e}")
                print(f"   per-point path vs node-values path: max |difference| {np.abs(r_old - r_new).max():.2e} "
                      f"({np.abs(r_old - r_new).max() / scale:.1e} of the largest row)")
                diff = np.abs(r_new - r_rep)
                worst = int(np.argmax(diff))
                print(f"   node-values path vs replica:        max |difference| {diff.max():.2e} "
                      f"({diff.max() / scale:.1e} of the largest row), worst at a {kind[worst]} row")
                if label == "exact solution" and args.diag:
                    diagnose(disc, sysm, replica_problem(name), N, worst, float(dx))
                if label == "exact solution":
                    h3 = float(dx) ** 3
                    parts = []
                    for k in ("bulk", "ghost", "cut", "wall"):
                        sel = kind == k
                        if sel.any():
                            parts.append(f"{k} {np.abs(r_new[sel]).max():.2e} (replica {np.abs(r_rep[sel]).max():.2e}, {int(sel.sum())} rows)")
                    print("   exact solution's weighted residual, largest by row kind: " + "; ".join(parts))


if __name__ == "__main__":
    main()
