"""
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2022 Pouria Mistani and Samira Pakravan. All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: mistani

"""

from functools import partial
from typing import Callable

import jax
from jax import config, grad, jit
from jax import numpy as jnp
from jax import vmap

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.domain import interpolate
from jax_dips.domain.mesh import GridState
from jax_dips.geometry import geometric_integrations_per_point
from jax_dips.solvers.simulation_states import PoissonSimState, PoissonSimStateFn


class Discretization:
    """
    This is a completely local point-based Poisson solver.
    """

    # Robin: cells with a smaller Omega- volume fraction count as outside (rounding slivers).
    min_vol_frac = 1e-6
    # Robin: length l in the row weight sqrt(V + l*A); "h" uses the cell size instead of a fixed length.
    robin_weight_length = 1.0
    # Mirror the 5-tet split on odd i+j+k cells so neighbours share face diagonals; False is the original split.
    alternate_tets = True
    # Jump problems. "original": the extrapolation rules as published (Bochkov & Gibou 2020, JAX-DIPS
    # eqs. 2-6), fed the configs' stored -beta, with delta = phi, the gradient fitted in the faster
    # region, and every row divided by its diagonal. "fixed": the rules derived and checked in
    # tools/jump_replica3d.py (beta with its own sign, the [mu] term of both u- rules with the derived
    # sign, delta = phi / |grad phi|), the gradient fitted in the slower region (Bochkov-Gibou's Bias
    # Slow), and each row divided by sqrt(h^3 + jump_ell * (max(A_Gamma, jump_cut_floor h^2) + A_ghost)),
    # A_ghost being h^2 per face of an uncut cell whose neighbour lies across Gamma.
    jump_scheme = "original"
    jump_ell = 1.0e4
    jump_cut_floor = 0.05
    # Jump problems: evaluate the network once per grid node per loss evaluation and build every row
    # from those values (compute_Ax_and_b_from_values_fn), instead of 28 evaluations per extrapolation,
    # 7 per row. The rows are the same; the memory is about 200 times smaller.
    jump_eval_once = False

    def __init__(
        self,
        lvl_gstate: GridState,
        sim_state: PoissonSimState,
        sim_state_fn: PoissonSimStateFn,
        precondition: int = 1,
        algorithm: int = 1,
    ) -> None:
        r"""
        algorithm = 0: use regression to evaluate u^\pm
        algorithm = 1: use neural network to evaluate u^\pm
        """
        self.algorithm = algorithm
        self.lvl_gstate = lvl_gstate
        self.sim_state_fn = sim_state_fn
        self.sim_state = sim_state

        """ Grid Info """
        # self.bandwidth_squared = (2.0 * self.dx)*(2.0 * self.dx)
        self.xmin = lvl_gstate.xmin()
        self.xmax = lvl_gstate.xmax()
        self.ymin = lvl_gstate.ymin()
        self.ymax = lvl_gstate.ymax()
        self.zmin = lvl_gstate.zmin()
        self.zmax = lvl_gstate.zmax()

        """ functions for the method """
        self.dir_bc_fn = self.sim_state_fn.dir_bc_fn
        self.f_m_interp_fn = self.sim_state_fn.f_m_fn
        self.f_p_interp_fn = self.sim_state_fn.f_p_fn
        self.k_m_interp_fn = self.sim_state_fn.k_m_fn
        self.k_p_interp_fn = self.sim_state_fn.k_p_fn
        self.mu_m_interp_fn = self.sim_state_fn.mu_m_fn
        self.mu_p_interp_fn = self.sim_state_fn.mu_p_fn
        self.alpha_interp_fn = self.sim_state_fn.alpha_fn
        self.beta_interp_fn = self.sim_state_fn.beta_fn
        self.g_interp_fn = getattr(
            self.sim_state_fn, "g_p_fn", lambda r: 0.0
        )  # For Robin boundary condition
        self.alphaRobin_interp_fn = (
            self.sim_state_fn.alpha_fn
        )  # For Robin boundary condition
        self.nonlinear_op_m = self.sim_state_fn.nonlinear_op_m
        self.nonlinear_op_p = self.sim_state_fn.nonlinear_op_p

        self.mu_m_over_mu_p_interp_fn = lambda r: self.mu_m_interp_fn(
            r
        ) / self.mu_p_interp_fn(r)
        self.beta_over_mu_m_interp_fn = lambda r: self.beta_interp_fn(
            r
        ) / self.mu_m_interp_fn(r)
        self.beta_over_mu_p_interp_fn = lambda r: self.beta_interp_fn(
            r
        ) / self.mu_p_interp_fn(r)

        """ The level set function or its interpolant (if is free boundary) """
        # self.phi_cube_ = sim_state.phi.reshape(self.grid_shape)
        # x, y, z, phi_cube = interpolate.add_ghost_layer_3d(xo, yo, zo, self.phi_cube_)
        # x, y, z, self.phi_cube = interpolate.add_ghost_layer_3d(x, y, z, phi_cube)
        # self.phi_flat = self.phi_cube_.reshape(-1)
        # self.phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(self.sim_state.phi, self.lvl_gstate)
        self.phi_interp_fn = self.sim_state_fn.phi_fn

        """ Geometric operations per point """
        (
            self.get_vertices_of_cell_intersection_with_interface_at_point,
            self.is_cell_crossed_by_interface,
        ) = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(
            self.phi_interp_fn, origin=jnp.array([self.xmin, self.ymin, self.zmin]) if self.alternate_tets else None
        )
        (
            self.beta_integrate_over_interface_at_point,
            self.beta_integrate_in_negative_domain,
        ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
            self.get_vertices_of_cell_intersection_with_interface_at_point,
            self.is_cell_crossed_by_interface,
            self.beta_interp_fn,
        )
        if hasattr(self.sim_state_fn, "g_p_fn"):
            (
                self.g_integrate_over_interface_at_point,
                self.g_integrate_in_negative_domain,
            ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
                self.get_vertices_of_cell_intersection_with_interface_at_point,
                self.is_cell_crossed_by_interface,
                self.g_interp_fn,
            )
            (
                self.alphaRobin_integrate_over_interface_at_point,
                self.alphaRobin_integrate_in_negative_domain,
            ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
                self.get_vertices_of_cell_intersection_with_interface_at_point,
                self.is_cell_crossed_by_interface,
                self.alphaRobin_interp_fn,
            )
            # Integrating 1 over the interface gives its area inside the cell.
            (
                self.area_integrate_over_interface_at_point,
                _,
            ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
                self.get_vertices_of_cell_intersection_with_interface_at_point,
                self.is_cell_crossed_by_interface,
                lambda r: jnp.ones(r.shape[0]),
            )
        if self.jump_scheme not in ("original", "fixed"):
            raise ValueError(f"jump_scheme must be 'original' or 'fixed', not {self.jump_scheme!r}")
        # Area of Gamma inside a cell (the integral of 1 over it); the "fixed" jump rows' weights use it.
        (
            self.interface_area_at_point,
            _,
        ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
            self.get_vertices_of_cell_intersection_with_interface_at_point,
            self.is_cell_crossed_by_interface,
            lambda r: jnp.ones(r.shape[0]),
        )
        self.compute_face_centroids_values_plus_minus_at_point = (
            geometric_integrations_per_point.compute_cell_faces_areas_values(
                self.get_vertices_of_cell_intersection_with_interface_at_point,
                self.is_cell_crossed_by_interface,
                self.mu_m_interp_fn,
                self.mu_p_interp_fn,
            )
        )

        # self.ngbs = jnp.array([ [-1, -1, -1],
        #                         [0, -1, -1],
        #                         [1, -1, -1],
        #                         [-1,  0, -1],
        #                         [0,  0, -1],
        #                         [1,  0, -1],
        #                         [-1,  1, -1],
        #                         [0,  1, -1],
        #                         [1,  1, -1],
        #                         [-1, -1,  0],
        #                         [0, -1,  0],
        #                         [1, -1,  0],
        #                         [-1,  0,  0],
        #                         [0,  0,  0],
        #                         [1,  0,  0],
        #                         [-1,  1,  0],
        #                         [0,  1,  0],
        #                         [1,  1,  0],
        #                         [-1, -1,  1],
        #                         [0, -1,  1],
        #                         [1, -1,  1],
        #                         [-1,  0,  1],
        #                         [0,  0,  1],
        #                         [1,  0,  1],
        #                         [-1,  1,  1],
        #                         [0,  1,  1],
        #                         [1,  1,  1]], dtype=i32)

        """ initialize configurated solver """
        if self.algorithm == 0:
            self.u_mp_fn = self.get_u_mp_by_regression_at_point_fn

        elif (
            self.algorithm == 1
        ):  # TODO: implement neural network based extrapolation function
            self.initialize_neural_based_algorithm()
            self.u_mp_fn = NotImplemented  # self.get_u_mp_by_neural_network_at_node_fn

        self.compute_normal_gradient_solution_mp_on_interface = (
            self.compute_normal_gradient_solution_mp_on_interface_neural_network
        )
        self.compute_gradient_solution_mp = (
            self.compute_gradient_solution_mp_neural_network
        )
        self.compute_normal_gradient_solution_on_interface = (
            self.compute_normal_gradient_solution_on_interface_neural_network
        )
        self.compute_gradient_solution = self.compute_gradient_solution_neural_network

        self.robin = self.sim_state_fn.g_p_fn is not None

        if precondition == 1:
            if self.robin:
                self.compute_Ax_and_b_fn = self.compute_Ax_and_b_preconditioned_fn_Robin
            else:
                self.compute_Ax_and_b_fn = self.compute_Ax_and_b_preconditioned_fn
        elif precondition == 0:
            self.compute_Ax_and_b_fn = self.compute_Ax_and_b_vanilla_fn

    # This function gives the 27 neighboring nodes around node Xijk.
    def get_Xijk(self, cell_dx, cell_dy, cell_dz):
        Xijk = jnp.array(
            [
                [-cell_dx, -cell_dy, -cell_dz],
                [0.0, -cell_dy, -cell_dz],
                [cell_dx, -cell_dy, -cell_dz],
                [-cell_dx, 0.0, -cell_dz],
                [0.0, 0.0, -cell_dz],
                [cell_dx, 0.0, -cell_dz],
                [-cell_dx, cell_dy, -cell_dz],
                [0.0, cell_dy, -cell_dz],
                [cell_dx, cell_dy, -cell_dz],
                [-cell_dx, -cell_dy, 0.0],
                [0.0, -cell_dy, 0.0],
                [cell_dx, -cell_dy, 0.0],
                [-cell_dx, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [cell_dx, 0.0, 0.0],
                [-cell_dx, cell_dy, 0.0],
                [0.0, cell_dy, 0.0],
                [cell_dx, cell_dy, 0.0],
                [-cell_dx, -cell_dy, cell_dz],
                [0.0, -cell_dy, cell_dz],
                [cell_dx, -cell_dy, cell_dz],
                [-cell_dx, 0.0, cell_dz],
                [0.0, 0.0, cell_dz],
                [cell_dx, 0.0, cell_dz],
                [-cell_dx, cell_dy, cell_dz],
                [0.0, cell_dy, cell_dz],
                [cell_dx, cell_dy, cell_dz],
            ],
            dtype=f32,
        )
        return Xijk

    def normal_point_fn(self, point, dx, dy, dz):
        """
        Evaluate normal vector at a given point based on interpolated values
        of the level set function at the face-centers of a 3D cell centered at the
        point with each side length given by dx, dy, dz.
        """
        point_ip1_j_k = jnp.array([[point[0] + dx, point[1], point[2]]])
        point_im1_j_k = jnp.array([[point[0] - dx, point[1], point[2]]])
        phi_x = (
            self.phi_interp_fn(point_ip1_j_k) - self.phi_interp_fn(point_im1_j_k)
        ) / (2 * dx)

        point_i_jp1_k = jnp.array([[point[0], point[1] + dy, point[2]]])
        point_i_jm1_k = jnp.array([[point[0], point[1] - dy, point[2]]])
        phi_y = (
            self.phi_interp_fn(point_i_jp1_k) - self.phi_interp_fn(point_i_jm1_k)
        ) / (2 * dy)

        point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + dz]])
        point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - dz]])
        phi_z = (
            self.phi_interp_fn(point_i_j_kp1) - self.phi_interp_fn(point_i_j_km1)
        ) / (2 * dz)

        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm], dtype=f32)

    def grad_phi_r(self, point, dx, dy, dz):
        point_ip1_j_k = jnp.array([[point[0] + dx, point[1], point[2]]])
        point_im1_j_k = jnp.array([[point[0] - dx, point[1], point[2]]])
        phi_x = (
            self.phi_interp_fn(point_ip1_j_k) - self.phi_interp_fn(point_im1_j_k)
        ) / (2 * dx)

        point_i_jp1_k = jnp.array([[point[0], point[1] + dy, point[2]]])
        point_i_jm1_k = jnp.array([[point[0], point[1] - dy, point[2]]])
        phi_y = (
            self.phi_interp_fn(point_i_jp1_k) - self.phi_interp_fn(point_i_jm1_k)
        ) / (2 * dy)

        point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + dz]])
        point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - dz]])
        phi_z = (
            self.phi_interp_fn(point_i_j_kp1) - self.phi_interp_fn(point_i_j_km1)
        ) / (2 * dz)

        norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
        return norm  # returns grad(phi_r) (copy of above but just return after norm)

    def initialize_neural_based_algorithm(self):
        """Initialize masks needed for neural network based extrapolation approach"""

        def sign_p_fn(a):
            # returns 1 only if a>0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.floor(0.5 * sgn + 0.75)

        def sign_m_fn(a):
            # returns 1 only if a<0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.ceil(0.5 * sgn - 0.75) * (-1.0)

        self.mask_region_m = sign_m_fn(self.phi_flat)
        self.mask_region_p = sign_p_fn(self.phi_flat)
        self.mask_interface_bandwidth = sign_m_fn(
            self.phi_flat**2 - self.bandwidth_squared
        )
        self.mask_non_interface_bandwidth = sign_p_fn(
            self.phi_flat**2 - self.bandwidth_squared
        )

    def get_regression_coeffs_at_point(self, point, dx, dy, dz):
        def sign_p_fn(a):
            # returns 1 only if a>0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.floor(0.5 * sgn + 0.75)

        def sign_m_fn(a):  # This is to define equation (1) in JAX-dips, i.e. the w_ijk
            # returns 1 only if a<0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.ceil(0.5 * sgn - 0.75) * (-1.0)

        x, y, z = point
        Xijk = self.get_Xijk(dx, dy, dz)  # Local neighborhood as     -dx   0   dx
        curr_vertices = jnp.add(
            jnp.array([x, y, z]), Xijk
        )  # Local neighborhood as     x-dx   x   x+dx
        phi_vertices = self.phi_interp_fn(
            curr_vertices
        )  # Value of the level-set function at the neighboring points

        Wijk_p = jnp.diag(
            vmap(sign_p_fn)(phi_vertices)
        )  # Get the diagonal elements of Wijk in + region (page 8 of JAX dips)
        Wijk_m = jnp.diag(
            vmap(sign_m_fn)(phi_vertices)
        )  # Get the diagonal elements of Wijk in - region (page 8 of JAX dips)

        Dp = (
            jnp.linalg.pinv(Xijk.T @ Wijk_p @ Xijk) @ (Wijk_p @ Xijk).T
        )  # Get D_ijk = ()^-1 ()^T in + region (page 8 of JAX dips)
        Dm = (
            jnp.linalg.pinv(Xijk.T @ Wijk_m @ Xijk) @ (Wijk_m @ Xijk).T
        )  # Get D_ijk = ()^-1 ()^T in + region (page 8 of JAX dips)
        D_m_mat = jnp.nan_to_num(Dm)  # Not sure - maybe a sanity check.
        D_p_mat = jnp.nan_to_num(Dp)  # Not sure - maybe a sanity check.

        normal_vec = self.normal_point_fn(
            point, dx, dy, dz
        ).T  # Get the normal    at current point.
        phi_point = self.phi_interp_fn(
            point[jnp.newaxis]
        )  # Get the level-set at current point.

        Cm_ijk_pqm = normal_vec @ D_m_mat  # Equation C in JAX
        Cp_ijk_pqm = normal_vec @ D_p_mat  # Equation C in JAX

        # Define equation Zeta in JAX:
        zeta_p_ijk_pqm = (
            (
                self.mu_p_interp_fn(point[jnp.newaxis])
                - self.mu_m_interp_fn(point[jnp.newaxis])
            )
            / self.mu_m_interp_fn(point[jnp.newaxis])
        ) * phi_point
        zeta_p_ijk_pqm = zeta_p_ijk_pqm[..., jnp.newaxis] * Cp_ijk_pqm
        zeta_m_ijk_pqm = (
            (
                self.mu_p_interp_fn(point[jnp.newaxis])
                - self.mu_m_interp_fn(point[jnp.newaxis])
            )
            / self.mu_p_interp_fn(point[jnp.newaxis])
        ) * phi_point
        zeta_m_ijk_pqm = zeta_m_ijk_pqm[..., jnp.newaxis] * Cm_ijk_pqm
        zeta_p_ijk = (zeta_p_ijk_pqm.sum(axis=1) - zeta_p_ijk_pqm[:, 13]) * f32(-1.0)
        zeta_m_ijk = (zeta_m_ijk_pqm.sum(axis=1) - zeta_m_ijk_pqm[:, 13]) * f32(-1.0)

        # Define equation Gamma in JAX:
        gamma_p_ijk_pqm = zeta_p_ijk_pqm / (1.0 + zeta_p_ijk[:, jnp.newaxis])
        gamma_m_ijk_pqm = zeta_m_ijk_pqm / (1.0 - zeta_m_ijk[:, jnp.newaxis])
        gamma_p_ijk = (gamma_p_ijk_pqm.sum(axis=1) - gamma_p_ijk_pqm[:, 13]) * f32(-1.0)
        gamma_m_ijk = (gamma_m_ijk_pqm.sum(axis=1) - gamma_m_ijk_pqm[:, 13]) * f32(-1.0)

        return (
            normal_vec,
            gamma_m_ijk,
            gamma_m_ijk_pqm,
            gamma_p_ijk,
            gamma_p_ijk_pqm,
            zeta_m_ijk,
            zeta_m_ijk_pqm,
            zeta_p_ijk,
            zeta_p_ijk_pqm,
        )

    # @partial(jit, static_argnums=(0))
    def compute_Ax_and_b_preconditioned_fn(self, params, point, dx, dy, dz):
        """
        This function calculates  A @ u for a given vector of unknowns u.
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || A u^k - b ||^2

        using autodiff we can compute gradients w.r.t u^k values, and optimize for the solution field.

        * PROCEDURE:
            first compute u = B:u + r for each node
            then use the actual cell geometries (face areas and mu coeffs) to
            compute the rhs of the linear system given currently passed-in u vector
            for solution estimate.

        The network is evaluated inside the row: 28 times for each of the 7 nodes of the stencil
        (get_u_mp_by_regression_at_point_fn). compute_Ax_and_b_from_values_fn builds the same row
        from values computed once per node.
        """

        u_mp_at_point = partial(
            self.u_mp_fn, params, dx, dy, dz
        )  # Get the value of u at the current grid point using the BIAS SLOW algorithm (coded at 426 get_u_mp_by_regression_at_point_fn, which calls get_regression_coeffs_at_point where all the computations are done).
        steps = jnp.array([dx, dy, dz], dtype=f32)

        def u_mp_at(offset):
            return u_mp_at_point(point + jnp.array(offset, dtype=f32) * steps)

        def u_at_node():
            return self.solution_at_point_fn(params, point, self.phi_interp_fn(point[jnp.newaxis]))

        def precond_fn(coeffs_):
            return self.precond_fn(params, coeffs_)  # TODO learning voxel-level preconditioner (Onur did this!)

        return self._jump_row(point, dx, dy, dz, u_mp_at, u_at_node, precond_fn)

    def compute_Ax_and_b_from_values_fn(self, U_pad, point, dx, dy, dz):
        """
        The same row as compute_Ax_and_b_preconditioned_fn, built from the network's values at the
        grid nodes instead of evaluating the network inside the row.

        U_pad: the solution at every node of the training grid (each node's value on its own side of
        Gamma, as evaluate_solution_fn gives it), shape (Nx, Ny, Nz), padded with 2 zeros on every
        side. The row of a node needs, for each of its 7 stencil nodes, the 27 values of that node's
        3x3x3 neighbourhood (the extrapolation's least-squares fit), so it reads the 5x5x5 block of
        nodes around it. All of them are grid nodes (get_Xijk steps by dx, dy, dz), so the values are
        exactly those the network would give; padded values are read only by rows that do not use
        them (wall rows, and cells away from Gamma near the walls). eval_once_origin is the grid's
        lowest corner (set by the Trainer).

        The learned preconditioner is not applied here (it needs params); with the default dummy
        preconditioner it is 1 anyway.
        """
        steps = jnp.array([dx, dy, dz], dtype=f32)
        ijk = jnp.round((point - self.eval_once_origin) / steps).astype(jnp.int32)
        # The node is at ijk + 2 in U_pad, so its 5x5x5 block starts at ijk.
        block = jax.lax.dynamic_slice(U_pad, (ijk[0], ijk[1], ijk[2]), (5, 5, 5))

        def cube_at(offset):
            # The 27 values around the stencil node at `offset`, in get_Xijk's order (x fastest).
            a, b, c = 1 + offset[0], 1 + offset[1], 1 + offset[2]
            return jnp.transpose(block[a:a + 3, b:b + 3, c:c + 3], (2, 1, 0)).reshape(27)

        def u_mp_at(offset):
            cube = cube_at(offset)
            node = point + jnp.array(offset, dtype=f32) * steps
            return self.u_mp_from_values(node, cube[13], cube, dx, dy, dz)

        def u_at_node():
            return block[2, 2, 2]

        return self._jump_row(point, dx, dy, dz, u_mp_at, u_at_node, lambda coeffs_: 1.0)

    def _jump_row(self, point, dx, dy, dz, u_mp_at, u_at_node, precond_fn):
        """
        One row of the jump discretization (Bochkov & Gibou 2020, JAX-DIPS eq. Standard), divided by
        its weight: the diagonal for jump_scheme "original", the interface weight for "fixed".

        u_mp_at(offset): (u-, u+) at the stencil node point + offset * (dx, dy, dz), offset one of
        (0,0,0) and the 6 neighbours; u_at_node(): the solution at the node itself (wall rows).
        Returns [lhs / w, rhs / w], shape (2, 1).
        """
        steps = jnp.array([dx, dy, dz], dtype=f32)

        def is_box_boundary_point(point):
            """
            Check if current node is on the boundary of box (i.e. the sides of the computational domain \Omega
            """
            x, y, z = point
            boundary = jnp.where(abs(x - self.xmin) < 1e-6 * dx, 0, 1) * jnp.where(
                abs(x - self.xmax) < 1e-6 * dx, 0, 1
            )
            boundary *= jnp.where(abs(y - self.ymin) < 1e-6 * dy, 0, 1) * jnp.where(
                abs(y - self.ymax) < 1e-6 * dy, 0, 1
            )
            boundary *= jnp.where(abs(z - self.zmin) < 1e-6 * dz, 0, 1) * jnp.where(
                abs(z - self.zmax) < 1e-6 * dz, 0, 1
            )
            return jnp.where(boundary == 0, True, False)

        # --- LHS
        coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(point, dx, dy, dz)
        coeffs = coeffs_[:12]
        precond = precond_fn(coeffs_)

        vols = coeffs_[12:14]
        V_m_ijk = vols[0]  # Volume of the partial cell in the minus region.
        V_p_ijk = vols[1]  # Volume of the partial cell in the plus  region.
        Vol_cell_nominal = dx * dy * dz  # Elementary volume

        def get_lhs_at_interior_point(point):
            k_m_ijk = self.k_m_interp_fn(point[jnp.newaxis])  # Reaction coefficient in the minus region.
            k_p_ijk = self.k_p_interp_fn(point[jnp.newaxis])  # Reaction coefficient in the plus  region.

            u_m_ijk, u_p_ijk = u_mp_at((0, 0, 0))  # Current point
            u_m_imjk, u_p_imjk = u_mp_at((-1, 0, 0))  # u value at the grid point to the left.
            u_m_ipjk, u_p_ipjk = u_mp_at((1, 0, 0))  # u value at the grid point to the right.
            u_m_ijmk, u_p_ijmk = u_mp_at((0, -1, 0))  # u value at the grid point to the bottom
            u_m_ijpk, u_p_ijpk = u_mp_at((0, 1, 0))  # u value at the grid point to the top.
            u_m_ijkm, u_p_ijkm = u_mp_at((0, 0, -1))  # u value at the grid point to the back.
            u_m_ijkp, u_p_ijkp = u_mp_at((0, 0, 1))  # u value at the grid point to the front.

            # \sum_{\pm} k^s_{i,j} u^s_{i,j} u^s_{i,j}, i.e. the first term in (Equation Standard of JAX)
            lhs = k_m_ijk * u_m_ijk * V_m_ijk
            lhs += k_p_ijk * u_p_ijk * V_p_ijk

            # Treating the case of a nonlinear \mu (not done here, i.e. setting it to zero). To do later if needed.
            lhs += self.nonlinear_op_m(u_m_ijk) * V_m_ijk + self.nonlinear_op_p(u_p_ijk) * V_p_ijk

            # We assume that the coeffs array gives the \mu_^s_{i-\frac12, j} A^s_{i-\frac12, j} / dx, etc. So the following gives
            # all the coefficients of the matrix in front of u_{ijk} in the minus and plus regions:
            lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) * u_m_ijk + (
                coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11]
            ) * u_p_ijk
            # Extra diagonal coefficients of the linear system, i.e. the matrix A
            lhs += -1.0 * coeffs[0] * u_m_imjk - coeffs[1] * u_p_imjk
            lhs += -1.0 * coeffs[2] * u_m_ipjk - coeffs[3] * u_p_ipjk
            lhs += -1.0 * coeffs[4] * u_m_ijmk - coeffs[5] * u_p_ijmk
            lhs += -1.0 * coeffs[6] * u_m_ijpk - coeffs[7] * u_p_ijpk
            lhs += -1.0 * coeffs[8] * u_m_ijkm - coeffs[9] * u_p_ijkm
            lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp

            # At this point, the matrix A is defined.
            # Compute the diagonal coefficient of the assembled matrix, which will serve as the (Jacobi) preconditioner.
            diag_coeff = (
                k_p_ijk * V_p_ijk
                + k_m_ijk * V_m_ijk
                + (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10])
                + (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11])
            )
            return jnp.array([lhs.reshape(), diag_coeff.reshape()])

        def get_lhs_on_box_boundary(point):  # Handle the boundary condition at the domain's wall
            u_boundary = u_at_node()
            lhs = u_boundary * Vol_cell_nominal
            return jnp.array([lhs.reshape(), Vol_cell_nominal])

        on_wall = is_box_boundary_point(point)
        lhs_diagcoeff = jnp.where(on_wall, get_lhs_on_box_boundary(point), get_lhs_at_interior_point(point))
        lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)

        # --- RHS
        def get_rhs_at_interior_point(point):  # Implementation of the right-hand side of (Equation Standard)
            rhs = (
                self.f_m_interp_fn(point[jnp.newaxis]) * V_m_ijk
                + self.f_p_interp_fn(point[jnp.newaxis]) * V_p_ijk
            )
            # The configs store beta = -[mu du/dn], so adding its integral is the -integral of [mu du/dn]
            # that the divergence theorem gives.
            rhs += self.beta_integrate_over_interface_at_point(point, dx, dy, dz)
            return rhs

        def get_rhs_on_box_boundary(point):  # Impose the boundary condition at the walls of the computational domain.
            return self.dir_bc_fn(point[jnp.newaxis]).reshape() * Vol_cell_nominal

        rhs = jnp.where(on_wall, get_rhs_on_box_boundary(point), get_rhs_at_interior_point(point))

        if self.jump_scheme == "fixed":
            # Interface weight: a row with an O(1) local truncation error (a cut cell, or an uncut cell
            # that reads a value extrapolated across Gamma) counts by its interface area times jump_ell,
            # every other row by its volume. Wall rows (u = g) are divided by their volume, i.e. weight 1.
            h3 = Vol_cell_nominal
            h2 = h3 ** (2.0 / 3.0)
            cut = self.is_cell_crossed_by_interface(point, dx, dy, dz) == 0
            area = self.interface_area_at_point(point, dx, dy, dz).reshape()
            area = jnp.where(cut, jnp.maximum(area, self.jump_cut_floor * h2), 0.0)
            minus = self.phi_interp_fn(point[jnp.newaxis])[0] < 0
            nbrs = point + jnp.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=f32) * steps
            across = (self.phi_interp_fn(nbrs) < 0) != minus
            face_areas = jnp.array([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy], dtype=f32)
            a_ghost = jnp.where(cut | on_wall, 0.0, (across * face_areas).sum())
            weight = jnp.where(on_wall, Vol_cell_nominal, jnp.sqrt(h3 + self.jump_ell * (area + a_ghost)))
        else:
            weight = diagcoeff  # Jacobi: "precond" is short for "we need to do better but we have not done it yet).

        lhs_over_w = jnp.nan_to_num(lhs / weight) * precond
        rhs_over_w = jnp.nan_to_num(rhs / weight) * precond
        return jnp.array([lhs_over_w.reshape(1), rhs_over_w.reshape(1)])

    def compute_Ax_and_b_preconditioned_fn_Robin(self, params, point, dx, dy, dz):
        """
        This function calculates  A @ u for a given vector of unknowns u in the case of a Robin boundary condition
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || A u^k - b ||^2

        using autodiff we can compute gradients w.r.t u^k values, and optimize for the solution field.

        * PROCEDURE:
            first compute u = B:u + r for each node
            then use the actual cell geometries (face areas and mu coeffs) to
            compute the rhs of the linear system given currently passed-in u vector
            for solution estimate.

        """

        def u_m_at_point(pt):
            # Pass a negative value to force evaluation of the Omega- MLP (mlp_m_fn)
            return self.solution_at_point_fn(params, pt, -1.0)

        def is_box_boundary_point(point):
            """
            Check if current node is on the boundary of box (i.e. the sides of the computational domain \Omega
            """
            x, y, z = point
            boundary = jnp.where(abs(x - self.xmin) < 1e-6 * dx, 0, 1) * jnp.where(
                abs(x - self.xmax) < 1e-6 * dx, 0, 1
            )
            boundary *= jnp.where(abs(y - self.ymin) < 1e-6 * dy, 0, 1) * jnp.where(
                abs(y - self.ymax) < 1e-6 * dy, 0, 1
            )
            boundary *= jnp.where(abs(z - self.zmin) < 1e-6 * dz, 0, 1) * jnp.where(
                abs(z - self.zmax) < 1e-6 * dz, 0, 1
            )
            return jnp.where(boundary == 0, True, False)

        def evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz):
            # --- LHS
            # coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(point, dx, dy, dz)
            coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(
                point, dx, dy, dz
            )
            # UPDATED: used to be minus at point but now its plusminus at
            coeffs = coeffs_[:12]
            precond = self.precond_fn(
                params, coeffs_
            )  # TODO learning voxel-level preconditioner (Onur did this!)

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]  # Volume of the partial cell in the minus region.
            # V_p_ijk = vols[1]    (not used) Volume of the partial cell in the plus  region.
            Vol_cell_nominal = dx * dy * dz  # Elementary volume

            def get_lhs_at_interior_point(point):
                point_ijk = point  # Current point
                point_imjk = jnp.array(
                    [point[0] - dx, point[1], point[2]], dtype=f32
                )  # Coordinate of the grid point to the left.
                point_ipjk = jnp.array(
                    [point[0] + dx, point[1], point[2]], dtype=f32
                )  # Coordinate of the grid point to the right.
                point_ijmk = jnp.array(
                    [point[0], point[1] - dy, point[2]], dtype=f32
                )  # Coordinate of the grid point to the bottom.
                point_ijpk = jnp.array(
                    [point[0], point[1] + dy, point[2]], dtype=f32
                )  # Coordinate of the grid point to the top.
                point_ijkm = jnp.array(
                    [point[0], point[1], point[2] - dz], dtype=f32
                )  # Coordinate of the grid point to the back.
                point_ijkp = jnp.array(
                    [point[0], point[1], point[2] + dz], dtype=f32
                )  # Coordinate of the grid point to the front.

                k_m_ijk = self.k_m_interp_fn(
                    point[jnp.newaxis]
                ).squeeze()  # Reaction coefficient in the minus region.
                # k_p_ijk = self.k_p_interp_fn(point[jnp.newaxis])    # (Likely dont use)Reaction coefficient in the plus  region.

                # Only need u_m for Robin (no u_p)
                u_m_ijk = u_m_at_point(point_ijk)  # Current point
                u_m_imjk = u_m_at_point(
                    point_imjk
                )  # u value at the grid point to the left.
                u_m_ipjk = u_m_at_point(
                    point_ipjk
                )  # u value at the grid point to the right.
                u_m_ijmk = u_m_at_point(
                    point_ijmk
                )  # u value at the grid point to the bottom
                u_m_ijpk = u_m_at_point(
                    point_ijpk
                )  # u value at the grid point to the top.
                u_m_ijkm = u_m_at_point(
                    point_ijkm
                )  # u value at the grid point to the back.
                u_m_ijkp = u_m_at_point(
                    point_ijkp
                )  # u value at the grid point to the front.

                # \sum_{\pm} k^s_{i,j} u^s_{i,j} u^s_{i,j}, i.e. the first term in (Equation Standard of JAX)

                lhs = k_m_ijk * u_m_ijk * V_m_ijk
                # (dont use) lhs += k_p_ijk * u_p_ijk * V_p_ijk

                # Treating the case of a nonlinear \mu (not done here, i.e. setting it to zero). To do later if needed.
                lhs += (
                    self.nonlinear_op_m(u_m_ijk) * V_m_ijk
                )  # + self.nonlinear_op_p(u_p_ijk) * V_p_ijk

                # We assume that the coeffs array gives the \mu_^s_{i-\frac12, j} A^s_{i-\frac12, j} / dx, etc. So the following gives
                # all the coefficients of the matrix in front of u_{ijk} in the minus and plus regions:
                # lhs += (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10]) * u_m_ijk + ( # can take out +()upijk
                #         coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11]
                # ) * u_p_ijk
                lhs += (
                    coeffs[0]
                    + coeffs[2]
                    + coeffs[4]
                    + coeffs[6]
                    + coeffs[8]
                    + coeffs[10]
                ) * u_m_ijk

                # Extra diagonal coefficients of the linear system, i.e. the matrix A

                # Take out the upimjk

                # lhs = -1.0 + coeffs[i] will be 0 1 2 3 not 0 2 4 6, no plus
                # lhs += -1.0 * coeffs[0] * u_m_imjk #- coeffs[1] * u_p_imjk
                # lhs += -1.0 * coeffs[2] * u_m_ipjk #- coeffs[3] * u_p_ipjk
                # lhs += -1.0 * coeffs[4] * u_m_ijmk #- coeffs[5] * u_p_ijmk
                # lhs += -1.0 * coeffs[6] * u_m_ijpk #- coeffs[7] * u_p_ijpk
                # lhs += -1.0 * coeffs[8] * u_m_ijkm #- coeffs[9] * u_p_ijkm
                # lhs += -1.0 * coeffs[10] * u_m_ijkp #- coeffs[11] * u_p_ijkp

                lhs += -1.0 * coeffs[0] * u_m_imjk
                lhs += -1.0 * coeffs[2] * u_m_ipjk
                lhs += -1.0 * coeffs[4] * u_m_ijmk
                lhs += -1.0 * coeffs[6] * u_m_ijpk
                lhs += -1.0 * coeffs[8] * u_m_ijkm
                lhs += -1.0 * coeffs[10] * u_m_ijkp

                # Impose the Robin boundary condition (Eq 12, 13, 14):
                
                # d_ijk = self.phi_interp_fn(point) / self.grad_phi_r(point)

                # d_ijk = jnp.abs(self.phi_interp_fn(point[jnp.newaxis])).reshape()[0] / self.grad_phi_r(point, dx, dy, dz)
                phi_val = self.phi_interp_fn(point[jnp.newaxis]).reshape(-1)[0]
                grad_val = self.grad_phi_r(point, dx, dy, dz).squeeze()
                
                d_ijk = phi_val / grad_val
            
                n = self.normal_point_fn(point, dx, dy, dz).squeeze()

                point_projected = point - d_ijk * n
                
                alpha_ell = self.alphaRobin_integrate_over_interface_at_point(point, dx, dy, dz)
                mu_r = self.mu_m_interp_fn(point_projected[jnp.newaxis]).squeeze()
                alpha_r = self.alphaRobin_interp_fn(
                    point_projected[jnp.newaxis]
                ).squeeze()

               

                # Bochkov, Gibou paper Equation 14
                coeff_integral_au = (mu_r * alpha_ell) / (mu_r - (alpha_r * d_ijk))

                lhs += coeff_integral_au * u_m_ijk
                
                # Compute the diagonal coefficient of the assembled matrix, which will serve as the (Jacobi) preconditioner.

                # take out KpVp, add alpha L(interface and cell, rewrite coeffs 012345)
                # diag_coeff = (
                #         k_p_ijk * V_p_ijk
                #         + k_m_ijk * V_m_ijk
                #         + (coeffs[0] + coeffs[2] + coeffs[4] + coeffs[6] + coeffs[8] + coeffs[10])
                #         + (coeffs[1] + coeffs[3] + coeffs[5] + coeffs[7] + coeffs[9] + coeffs[11])
                # )
                diag_coeff = (
                    k_m_ijk * V_m_ijk
                    + (
                        coeffs[0]
                        + coeffs[2]
                        + coeffs[4]
                        + coeffs[6]
                        + coeffs[8]
                        + coeffs[10]
                    )
                    + coeff_integral_au
                )
                # Row scale sqrt(V_m + l * interface area) instead of diag_coeff: the squared loss then
                # sums the PDE residual over the volume and the Robin residual over the interface.
                gamma_ell = self.area_integrate_over_interface_at_point(point, dx, dy, dz)
                ell = dx if self.robin_weight_length == "h" else self.robin_weight_length
                row_scale = jnp.sqrt(V_m_ijk + ell * gamma_ell)
                return jnp.array([lhs.squeeze(), row_scale.squeeze()])

            def get_lhs_on_box_boundary(
                point,
            ):  # Handle the boundary condition at the domain's wall
                phi_boundary = self.phi_interp_fn(point[jnp.newaxis])
                u_boundary = self.solution_at_point_fn(params, point, phi_boundary)
                lhs = u_boundary * Vol_cell_nominal
                return jnp.array([lhs, Vol_cell_nominal])

            def get_lhs_in_omega_plus(point):
                # In Omega+ we are not solving the PDE — enforce u = dirichlet_bc
                phi_pt = self.phi_interp_fn(point[jnp.newaxis])
                u_pt = self.solution_at_point_fn(params, point, phi_pt)
                lhs = u_pt * Vol_cell_nominal
                return jnp.array([lhs, Vol_cell_nominal])

            def is_in_omega_plus(point):
                # True if no part of the cell lies in Omega-
                return V_m_ijk <= self.min_vol_frac * Vol_cell_nominal

            # Three-way dispatch: box boundary → Omega+ interior → Omega- interior
            # OLD: 
            lhs_diagcoeff = jnp.where(
                is_box_boundary_point(point),
                get_lhs_on_box_boundary(point),
                jnp.where(
                    is_in_omega_plus(point),
                    get_lhs_in_omega_plus(point),
                    get_lhs_at_interior_point(point),
                ),
            )
            # lhs_diagcoeff = jnp.where(
            #     is_in_omega_plus(point),
            #     get_lhs_in_omega_plus(point),
            #     jnp.where(
            #         is_box_boundary_point(point),
            #         get_lhs_on_box_boundary(point),
            #         get_lhs_at_interior_point(point),
            #     ),
            # )
            lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)

            # --- RHS
            def get_rhs_at_interior_point(
                point,
            ):  # Implementation of the right-hand side of (Equation Standard)
                rhs = (
                    # self.f_m_interp_fn(point[jnp.newaxis]) * V_m_ijk + self.f_p_interp_fn(point[jnp.newaxis]) * V_p_ijk ## remove the V_p_ijk?
                    self.f_m_interp_fn(point[jnp.newaxis]).squeeze()
                    * V_m_ijk
                )
                rhs += self.g_integrate_over_interface_at_point(point, dx, dy, dz)

                # d_ijk = jnp.abs(self.phi_interp_fn(point[jnp.newaxis])).reshape()[0] / self.grad_phi_r(point, dx, dy, dz)

                phi_val = self.phi_interp_fn(point[jnp.newaxis]).reshape(-1)[0]
                grad_val = self.grad_phi_r(point, dx, dy, dz).squeeze()
            
                d_ijk = phi_val / grad_val
                n = self.normal_point_fn(point, dx, dy, dz).squeeze()
            
                point_projected = point - d_ijk * n
                alpha_ell = self.alphaRobin_integrate_over_interface_at_point(
                                    point, dx, dy, dz
                                )
                

                # g_r = self.g_interp_fn(point)
                g_r = self.g_interp_fn(point_projected[jnp.newaxis]).squeeze()
                # mu_r = self.mu_m_interp_fn(point)
                mu_r = self.mu_m_interp_fn(point_projected[jnp.newaxis]).squeeze()
                # alpha_r = self.alphaRobin_interp_fn(point)
                alpha_r = self.alphaRobin_interp_fn(point_projected[jnp.newaxis]).squeeze()
                # Signed d (Bochkov-Gibou eq. 14): holds for cell centres on either side of Gamma.
                rhs += (g_r * d_ijk * alpha_ell) / (mu_r - (alpha_r * d_ijk))
                return rhs.squeeze()

            def get_rhs_on_box_boundary(
                point,
            ):  # Impose the boundary condition at the walls of the computational domain.
                return self.dir_bc_fn(point[jnp.newaxis]).squeeze() * Vol_cell_nominal

            def get_rhs_in_omega_plus(point):
                # In Omega+ enforce u = dirichlet_bc
                return self.dir_bc_fn(point[jnp.newaxis]).squeeze() * Vol_cell_nominal
                #return jnp.array(0.0)

            # Three-way dispatch matching LHS
            rhs = jnp.where(
                is_box_boundary_point(point),
                get_rhs_on_box_boundary(point),
                jnp.where(
                    is_in_omega_plus(point),
                    get_rhs_in_omega_plus(point),
                    get_rhs_at_interior_point(point),
                ),
            )

            # Apply the preconditioning of the linear system:
            lhs_over_diag = (
                jnp.nan_to_num(lhs / diagcoeff) * precond
            )  # "precond" is short for "we need to do better but we have not done it yet).
            rhs_over_diag = jnp.nan_to_num(rhs / diagcoeff) * precond
            return jnp.array([lhs_over_diag, rhs_over_diag])

        lhs_rhs = evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz)
        return lhs_rhs

    # @partial(jit, static_argnums=(0))
    def get_u_mp_by_regression_at_point_fn(self, params, dx, dy, dz, point):
        """
        This function evaluates pairs of u^+ and u^- at each grid point
        in the domain, given the neural network models.

        BIAS SLOW:
            This function evaluates
                u_m = B_m : u + r_m
            and
                u_p = B_p : u + r_p
        """

        delta_ijk = self.phi_interp_fn(point[jnp.newaxis])
        u_ijk = self.solution_at_point_fn(params, point, delta_ijk)
        Xijk = self.get_Xijk(dx, dy, dz)

        curr_vertices = jnp.add(point, Xijk)
        u_cube_ijk = self.evaluate_solution_fn(params, curr_vertices)
        return self.u_mp_from_values(point, u_ijk, u_cube_ijk.reshape(27), dx, dy, dz)

    def u_mp_from_values(self, point, u_ijk, u_cube_ijk, dx, dy, dz):
        """(u-, u+) at a grid node from the solution's values at the node (u_ijk) and at its 27-point
        neighbourhood (u_cube_ijk, get_Xijk's order), each on its own side of Gamma. In an uncut cell
        the value on the cell's side and 0 on the other; in a cut cell the node's own value and the
        other side's value extrapolated across Gamma, by jump_scheme's rules."""
        if self.jump_scheme == "fixed":
            return self._u_mp_fixed(point, u_ijk, u_cube_ijk, dx, dy, dz)
        return self._u_mp_original(point, u_ijk, u_cube_ijk, dx, dy, dz)

    def _jump_fixed_extrapolation(self, point, dx, dy, dz):
        """The other side's value at a node whose cell is cut, as coef . u_cube + const, with the
        corrected rules (tools/jump_replica3d.py::extrapolation, fixed=True, bias="slow"):

          Taylor step to the projection of the node on Gamma, distance d = phi / |grad phi|:
              u+ - u- = alpha + d (du+/dn - du-/dn)
          the flux jump mu+ du+/dn - mu- du-/dn = beta eliminates one normal derivative, and the other
          comes from a least-squares fit of the gradient over the node's 3x3x3 neighbours in the
          slower-diffusing region (Bias Slow): du/dn = cc u_node + sum_q c_q u_q.
        Fit in Omega- (mu- <= mu+), K = d (mu+ - mu-)/mu+:
              node in Omega+: u- = (u+ - alpha - d beta/mu+ + K sum c_q u_q) / (1 - K cc)
              node in Omega-: u+ = (1 - K cc) u- - K sum c_q u_q + alpha + d beta/mu+
        Fit in Omega+ (mu+ < mu-), L = d (mu+ - mu-)/mu-:
              node in Omega+: u- = (1 + L cc) u+ + L sum c_q u_q - alpha - d beta/mu-
              node in Omega-: u+ = (u- - L sum c_q u_q + alpha + d beta/mu-) / (1 + L cc)
        beta is [mu du/dn] (the configs store -beta); beta/mu and alpha are taken at the projection,
        the mu in K and L at the node. Returns (node in Omega+, coef (27,), const)."""
        X = self.get_Xijk(dx, dy, dz)
        e = jnp.diag(jnp.array([dx, dy, dz], dtype=f32))
        phi_at = lambda p: self.phi_interp_fn(p[jnp.newaxis])[0]
        g = jnp.array([(phi_at(point + e[i]) - phi_at(point - e[i])) / (2.0 * e[i, i]) for i in range(3)])
        g_norm = jnp.linalg.norm(g)
        n = g / g_norm
        d = phi_at(point) / g_norm  # signed distance to Gamma, phi need not be one
        mu_m = self.mu_m_interp_fn(point[jnp.newaxis])[0]
        mu_p = self.mu_p_interp_fn(point[jnp.newaxis])[0]
        r_proj = (point - d * n)[jnp.newaxis]
        alpha = self.alpha_interp_fn(r_proj)[0]
        beta = -1.0 * self.beta_interp_fn(r_proj)[0]  # the configs store -[mu du/dn]
        mu_m_proj = self.mu_m_interp_fn(r_proj)[0]
        mu_p_proj = self.mu_p_interp_fn(r_proj)[0]

        fit_minus = mu_m <= mu_p  # Bias Slow: fit the gradient in the slower region
        phi_cube = self.phi_interp_fn(point + X)
        W = jnp.where(fit_minus, phi_cube < 0, phi_cube > 0).astype(X.dtype)
        WX = W[:, jnp.newaxis] * X
        D = jnp.nan_to_num(jnp.linalg.pinv(X.T @ WX) @ WX.T)
        c = n @ D
        cc = -1.0 * c.sum()
        e13 = jnp.zeros(27, dtype=X.dtype).at[13].set(1.0)
        node_plus = d > 0

        K = d * (mu_p - mu_m) / mu_p
        jb1 = alpha + d * beta / mu_p_proj
        den1 = 1.0 - K * cc
        coef_1m, const_1m = (e13 + K * c) / den1, -1.0 * jb1 / den1
        coef_1p, const_1p = e13 * (1.0 - K * cc) - K * c, jb1
        L = d * (mu_p - mu_m) / mu_m
        jb2 = alpha + d * beta / mu_m_proj
        den2 = 1.0 + L * cc
        coef_2m, const_2m = e13 * (1.0 + L * cc) + L * c, -1.0 * jb2
        coef_2p, const_2p = (e13 - L * c) / den2, jb2 / den2

        coef = jnp.where(fit_minus, jnp.where(node_plus, coef_1m, coef_1p), jnp.where(node_plus, coef_2m, coef_2p))
        const = jnp.where(fit_minus, jnp.where(node_plus, const_1m, const_1p), jnp.where(node_plus, const_2m, const_2p))
        # Away from Gamma these are computed and then discarded; keep them finite so that the
        # discarded branch cannot turn gradients into NaN.
        coef = jnp.where(jnp.isfinite(coef), coef, 0.0)
        const = jnp.where(jnp.isfinite(const), const, 0.0)
        return node_plus, coef, const

    def _u_mp_fixed(self, point, u_ijk, u_cube_ijk, dx, dy, dz):
        node_plus, coef, const = self._jump_fixed_extrapolation(point, dx, dy, dz)
        u_ijk = jnp.reshape(u_ijk, ())
        u_other = jnp.dot(coef, u_cube_ijk) + const
        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = self.is_cell_crossed_by_interface(point, dx, dy, dz)
        u_m = jnp.where(is_interface == 0, jnp.where(node_plus, u_other, u_ijk), jnp.where(is_interface == -1, u_ijk, 0.0))
        u_p = jnp.where(is_interface == 0, jnp.where(node_plus, u_ijk, u_other), jnp.where(is_interface == 1, u_ijk, 0.0))
        return jnp.array([u_m, u_p])

    def _u_mp_original(self, point, u_ijk, u_cube_ijk, dx, dy, dz):
        """The published rules, as the code had them (see jump_scheme)."""
        delta_ijk = self.phi_interp_fn(point[jnp.newaxis])

        (
            normal_ijk,
            gamma_m_ijk,
            gamma_m_ijk_pqm,
            gamma_p_ijk,
            gamma_p_ijk_pqm,
            zeta_m_ijk,
            zeta_m_ijk_pqm,
            zeta_p_ijk,
            zeta_p_ijk_pqm,
        ) = self.get_regression_coeffs_at_point(point, dx, dy, dz)

        def bulk_point(is_interface_, u_ijk_):
            return jnp.array(
                [
                    jnp.where(is_interface_ == -1, u_ijk_, 0.0),
                    jnp.where(is_interface_ == 1, u_ijk_, 0.0),
                ]
            )

        def interface_point(point):
            def mu_minus_bigger_fn(point):
                def extrapolate_u_m_from_negative_domain(r_ijk):
                    r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_m = -1.0 * jnp.dot(gamma_m_ijk_pqm, u_cube_ijk)
                    u_m += (1.0 - gamma_m_ijk + gamma_m_ijk_pqm[:, 13]) * u_ijk
                    u_m += (
                        -1.0
                        * (1.0 - gamma_m_ijk)
                        * (
                            self.alpha_interp_fn(r_m_proj)
                            + delta_ijk * self.beta_over_mu_p_interp_fn(r_m_proj)
                        )
                    )
                    return u_m.reshape()

                def extrapolate_u_p_from_positive_domain(r_ijk):
                    r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk[0]
                    u_p = -1.0 * jnp.dot(zeta_m_ijk_pqm, u_cube_ijk)
                    u_p += (1.0 - zeta_m_ijk + zeta_m_ijk_pqm[:, 13]) * u_ijk
                    u_p += self.alpha_interp_fn(
                        r_p_proj
                    ) + delta_ijk * self.beta_over_mu_p_interp_fn(r_p_proj)
                    return u_p.reshape()

                u_m = jnp.where(
                    delta_ijk > 0, extrapolate_u_m_from_negative_domain(point), u_ijk
                )[0]
                u_p = jnp.where(
                    delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain(point)
                )[0]
                return jnp.array([u_m, u_p])

            def mu_plus_bigger_fn(point):
                def extrapolate_u_m_from_negative_domain_(r_ijk):
                    r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_m = -1.0 * jnp.dot(zeta_p_ijk_pqm, u_cube_ijk)
                    u_m += (1.0 - zeta_p_ijk + zeta_p_ijk_pqm[:, 13]) * u_ijk
                    u_m += (-1.0) * (
                        self.alpha_interp_fn(r_m_proj)
                        + delta_ijk * self.beta_over_mu_m_interp_fn(r_m_proj)
                    )
                    return u_m.reshape()

                def extrapolate_u_p_from_positive_domain_(r_ijk):
                    r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                    u_p = -1.0 * jnp.dot(gamma_p_ijk_pqm, u_cube_ijk)
                    u_p += (1.0 - gamma_p_ijk + gamma_p_ijk_pqm[:, 13]) * u_ijk
                    u_p += (1.0 - gamma_p_ijk) * (
                        self.alpha_interp_fn(r_p_proj)
                        + delta_ijk * self.beta_over_mu_m_interp_fn(r_p_proj)
                    )
                    return u_p.reshape()

                u_m = jnp.where(
                    delta_ijk > 0, extrapolate_u_m_from_negative_domain_(point), u_ijk
                )[0]
                u_p = jnp.where(
                    delta_ijk > 0, u_ijk, extrapolate_u_p_from_positive_domain_(point)
                )[0]
                return jnp.array([u_m, u_p])

            mu_m_ijk = self.mu_m_interp_fn(point[jnp.newaxis])
            mu_p_ijk = self.mu_p_interp_fn(point[jnp.newaxis])
            return jnp.where(
                mu_m_ijk > mu_p_ijk, mu_minus_bigger_fn(point), mu_plus_bigger_fn(point)
            )

        # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
        is_interface = self.is_cell_crossed_by_interface(point, dx, dy, dz)
        # is_interface = jnp.where( delta_ijk*delta_ijk <= self.bandwidth_squared,  0, jnp.sign(delta_ijk))
        u_mp = jnp.where(
            is_interface == 0, interface_point(point), bulk_point(is_interface, u_ijk)
        )
        return u_mp

    # ------------------- traditional
    def compute_Ax_and_b_discrete_fn(self, eval_gstate, u, point, dx, dy, dz):
        """
        WARNING: Assumes lvl_gstate == tr_gstate and structured mesh

        This function calculates  A @ u for a given vector of unknowns u.
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || A u^k - b ||^2

        using autodiff we can compute gradients w.r.t u^k values, and optimize for the solution field.

        * PROCEDURE:
            first compute u = B:u + r for each node
            then use the actual cell geometries (face areas and mu coeffs) to
            compute the rhs of the linear system given currently passed-in u vector
            for solution estimate.

        """
        u_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(u, eval_gstate)
        u_mp_at_point = partial(
            self.get_u_mp_by_regression_at_point_discrete_fn, u_interp_fn, dx, dy, dz
        )

        def is_box_boundary_point(point):
            """
            Check if current node is on the boundary of box
            """
            x, y, z = point
            boundary = jnp.where(abs(x - self.xmin) < 1e-6 * dx, 0, 1) * jnp.where(
                abs(x - self.xmax) < 1e-6 * dx, 0, 1
            )
            boundary *= jnp.where(abs(y - self.ymin) < 1e-6 * dy, 0, 1) * jnp.where(
                abs(y - self.ymax) < 1e-6 * dy, 0, 1
            )
            boundary *= jnp.where(abs(z - self.zmin) < 1e-6 * dz, 0, 1) * jnp.where(
                abs(z - self.zmax) < 1e-6 * dz, 0, 1
            )
            return jnp.where(boundary == 0, True, False)

        def evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz):
            # --- LHS
            coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(
                point, dx, dy, dz
            )
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            V_p_ijk = vols[1]
            Vol_cell_nominal = dx * dy * dz

            def get_lhs_at_interior_point(point):
                point_ijk = point
                point_imjk = jnp.array([point[0] - dx, point[1], point[2]], dtype=f32)
                point_ipjk = jnp.array([point[0] + dx, point[1], point[2]], dtype=f32)
                point_ijmk = jnp.array([point[0], point[1] - dy, point[2]], dtype=f32)
                point_ijpk = jnp.array([point[0], point[1] + dy, point[2]], dtype=f32)
                point_ijkm = jnp.array([point[0], point[1], point[2] - dz], dtype=f32)
                point_ijkp = jnp.array([point[0], point[1], point[2] + dz], dtype=f32)

                k_m_ijk = self.k_m_interp_fn(point[jnp.newaxis])
                k_p_ijk = self.k_p_interp_fn(point[jnp.newaxis])

                u_m_ijk, u_p_ijk = u_mp_at_point(point_ijk)
                u_m_imjk, u_p_imjk = u_mp_at_point(point_imjk)
                u_m_ipjk, u_p_ipjk = u_mp_at_point(point_ipjk)
                u_m_ijmk, u_p_ijmk = u_mp_at_point(point_ijmk)
                u_m_ijpk, u_p_ijpk = u_mp_at_point(point_ijpk)
                u_m_ijkm, u_p_ijkm = u_mp_at_point(point_ijkm)
                u_m_ijkp, u_p_ijkp = u_mp_at_point(point_ijkp)

                lhs = k_m_ijk * V_m_ijk * u_m_ijk
                lhs += k_p_ijk * V_p_ijk * u_p_ijk

                lhs += (
                    self.nonlinear_op_m(u_m_ijk) * V_m_ijk
                    + self.nonlinear_op_p(u_p_ijk) * V_p_ijk
                )

                lhs += (
                    coeffs[0]
                    + coeffs[2]
                    + coeffs[4]
                    + coeffs[6]
                    + coeffs[8]
                    + coeffs[10]
                ) * u_m_ijk + (
                    coeffs[1]
                    + coeffs[3]
                    + coeffs[5]
                    + coeffs[7]
                    + coeffs[9]
                    + coeffs[11]
                ) * u_p_ijk
                lhs += -1.0 * coeffs[0] * u_m_imjk - coeffs[1] * u_p_imjk
                lhs += -1.0 * coeffs[2] * u_m_ipjk - coeffs[3] * u_p_ipjk
                lhs += -1.0 * coeffs[4] * u_m_ijmk - coeffs[5] * u_p_ijmk
                lhs += -1.0 * coeffs[6] * u_m_ijpk - coeffs[7] * u_p_ijpk
                lhs += -1.0 * coeffs[8] * u_m_ijkm - coeffs[9] * u_p_ijkm
                lhs += -1.0 * coeffs[10] * u_m_ijkp - coeffs[11] * u_p_ijkp

                diag_coeff = (
                    k_p_ijk * V_p_ijk
                    + k_m_ijk * V_m_ijk
                    + (
                        coeffs[0]
                        + coeffs[2]
                        + coeffs[4]
                        + coeffs[6]
                        + coeffs[8]
                        + coeffs[10]
                    )
                    + (
                        coeffs[1]
                        + coeffs[3]
                        + coeffs[5]
                        + coeffs[7]
                        + coeffs[9]
                        + coeffs[11]
                    )
                )
                return jnp.array([lhs.reshape(), diag_coeff.reshape()])

            def get_lhs_on_box_boundary(point):
                phi_boundary = self.phi_interp_fn(point[jnp.newaxis])
                u_boundary = u_interp_fn(point[jnp.newaxis]).squeeze()
                lhs = u_boundary * Vol_cell_nominal
                return jnp.array([lhs, Vol_cell_nominal])

            lhs_diagcoeff = jnp.where(
                is_box_boundary_point(point),
                get_lhs_on_box_boundary(point),
                get_lhs_at_interior_point(point),
            )
            lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)

            # --- RHS
            def get_rhs_at_interior_point(point):
                rhs = (
                    self.f_m_interp_fn(point[jnp.newaxis]) * V_m_ijk
                    + self.f_p_interp_fn(point[jnp.newaxis]) * V_p_ijk
                )
                rhs += self.beta_integrate_over_interface_at_point(point, dx, dy, dz)
                return rhs

            def get_rhs_on_box_boundary(point):
                return self.dir_bc_fn(point[jnp.newaxis]).reshape() * Vol_cell_nominal

            rhs = jnp.where(
                is_box_boundary_point(point),
                get_rhs_on_box_boundary(point),
                get_rhs_at_interior_point(point),
            )
            lhs_over_diag = jnp.nan_to_num(lhs / diagcoeff)
            rhs_over_diag = jnp.nan_to_num(rhs / diagcoeff)
            return jnp.array([lhs_over_diag, rhs_over_diag])

        lhs_rhs = evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz)
        return lhs_rhs

    def compute_Ax_and_b_discrete_Robin_fn(self, eval_gstate, u, point, dx, dy, dz):
        """
        WARNING: Assumes lvl_gstate == tr_gstate and structured mesh

        This function calculates  A @ u for a given vector of unknowns u.
        This evaluates the rhs in Au^k=b given estimate u^k.
        The purpose would be to define an optimization problem with:

        min || A u^k - b ||^2

        using autodiff we can compute gradients w.r.t u^k values, and optimize for the solution field.

        """
        u_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(u, eval_gstate)
        # u_mp_at_point = partial(self.get_u_mp_by_regression_at_point_discrete_fn, u_interp_fn, dx, dy, dz)

        x0 = eval_gstate.x
        y0 = eval_gstate.y
        z0 = eval_gstate.z
        u_cube = u.reshape((x0.shape[0], y0.shape[0], z0.shape[0]))

        def is_box_boundary_point(point):
            """
            Check if current node is on the boundary of box
            """
            x, y, z = point
            boundary = jnp.where(abs(x - self.xmin) < 1e-6 * dx, 0, 1) * jnp.where(
                abs(x - self.xmax) < 1e-6 * dx, 0, 1
            )
            boundary *= jnp.where(abs(y - self.ymin) < 1e-6 * dy, 0, 1) * jnp.where(
                abs(y - self.ymax) < 1e-6 * dy, 0, 1
            )
            boundary *= jnp.where(abs(z - self.zmin) < 1e-6 * dz, 0, 1) * jnp.where(
                abs(z - self.zmax) < 1e-6 * dz, 0, 1
            )
            return jnp.where(boundary == 0, True, False)

        def evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz):
            # --- LHS
            coeffs_ = self.compute_face_centroids_values_plus_minus_at_point(
                point, dx, dy, dz
            )
            coeffs = coeffs_[:12]

            vols = coeffs_[12:14]
            V_m_ijk = vols[0]
            Vol_cell_nominal = dx * dy * dz

            def get_lhs_at_interior_point(point):
                k_m_ijk = self.k_m_interp_fn(point[jnp.newaxis])

                i, j, k = self.find_lower_left_cell_idx(point)
                u_m_ijk = u_cube[i, j, k]
                u_m_imjk = u_cube[i - 1, j, k]
                u_m_ipjk = u_cube[i + 1, j, k]
                u_m_ijmk = u_cube[i, j - 1, k]
                u_m_ijpk = u_cube[i, j + 1, k]
                u_m_ijkm = u_cube[i, j, k - 1]
                u_m_ijkp = u_cube[i, j, k + 1]

                lhs = k_m_ijk * V_m_ijk * u_m_ijk

                lhs += self.nonlinear_op_m(u_m_ijk) * V_m_ijk

                lhs += (
                    -1.0 * coeffs[0] * u_m_imjk
                )  # should be L(i - 0.5, j      , k      )
                lhs += (
                    -1.0 * coeffs[2] * u_m_ipjk
                )  # should be L(i + 0.5, j      , k      )
                lhs += (
                    -1.0 * coeffs[4] * u_m_ijmk
                )  # should be L(i - 0.5, j - 0.5, k      )
                lhs += (
                    -1.0 * coeffs[6] * u_m_ijpk
                )  # should be L(i - 0.5, j + 0.5, k      )
                lhs += (
                    -1.0 * coeffs[8] * u_m_ijkm
                )  # should be L(i - 0.5, j      , k - 0.5)
                lhs += (
                    -1.0 * coeffs[10] * u_m_ijkp
                )  # should be L(i - 0.5, j      , k + 0.5)
                lhs += (
                    coeffs[0]
                    + coeffs[2]
                    + coeffs[4]
                    + coeffs[6]
                    + coeffs[8]
                    + coeffs[10]
                ) * u_m_ijk

                # Impose the Robin boundary condition (Eq 12, 13, 14):
                alpha_ell = self.alphaRobin_integrate_over_interface_at_point(
                    point, dx, dy, dz
                )

                d_ijk = jnp.abs(self.phi_interp_fn(point))
                mu_r = self.mu_m_interp_fn(point)
                alpha_r = self.alphaRobin_interp_fn(point)
                g_r = self.g_interp_fn(point)

                u_interface = (mu_r * u_m_ijk + g_r * d_ijk) / (mu_r + alpha_r * d_ijk)
                lhs += alpha_ell * u_interface  # Note, evaluated at interface

                # do we need to add alpha ell as part of the diag_coeff?
                # if so, we need to separate out alphaell * u_m_ijk and add them separately to lhs and then add
                # alpha ell to the diag_coeff element.

                # IS THIS DIAGONAL ELEMENT TO SCALE THE LINEAR SYSTEM OR IS USED INTO THE LHS?
                diag_coeff = (
                    k_m_ijk * V_m_ijk
                    + (
                        coeffs[0]
                        + coeffs[2]
                        + coeffs[4]
                        + coeffs[6]
                        + coeffs[8]
                        + coeffs[10]
                    )
                    + alpha_ell
                )
                return jnp.array([lhs.reshape(), diag_coeff.reshape()])

            def get_lhs_on_box_boundary(point):
                phi_boundary = self.phi_interp_fn(point[jnp.newaxis])
                u_boundary = u_interp_fn(point[jnp.newaxis]).squeeze()
                lhs = u_boundary * Vol_cell_nominal
                return jnp.array([lhs, Vol_cell_nominal])

            lhs_diagcoeff = jnp.where(
                is_box_boundary_point(point),
                get_lhs_on_box_boundary(point),
                get_lhs_at_interior_point(point),
            )
            lhs, diagcoeff = jnp.split(lhs_diagcoeff, [1], 0)

            # --- RHS
            def get_rhs_at_interior_point(point):
                rhs = self.f_m_interp_fn(point[jnp.newaxis]) * V_m_ijk
                rhs += self.g_integrate_over_interface_at_point(point, dx, dy, dz)
                return rhs

            def get_rhs_on_box_boundary(point):
                return self.dir_bc_fn(point[jnp.newaxis]).reshape() * Vol_cell_nominal

            rhs = jnp.where(
                is_box_boundary_point(point),
                get_rhs_on_box_boundary(point),
                get_rhs_at_interior_point(point),
            )
            lhs_over_diag = jnp.nan_to_num(lhs / diagcoeff)
            rhs_over_diag = jnp.nan_to_num(rhs / diagcoeff)
            return jnp.array([lhs_over_diag, rhs_over_diag])

        lhs_rhs = evaluate_discretization_lhs_rhs_at_point(point, dx, dy, dz)
        return lhs_rhs

        def get_u_mp_by_regression_at_point_discrete_fn(
            self, u_interp_fn, dx, dy, dz, point
        ):
            """
            This function evaluates pairs of u^+ and u^- at each grid point
            in the domain, given the neural network models. COMMENT: DOES NOT USE A NEURAL NETWORK TO DEFINE THE U_M AND U_P

            BIAS SLOW:
                This function evaluates
                    u_m = B_m : u + r_m
                and
                    u_p = B_p : u + r_p
            """
            delta_ijk = self.phi_interp_fn(point[jnp.newaxis])
            Xijk = self.get_Xijk(dx, dy, dz)
            u_ijk = u_interp_fn(point[jnp.newaxis]).squeeze()
            curr_vertices = jnp.add(point, Xijk)
            u_cube_ijk = u_interp_fn(curr_vertices)

            (
                normal_ijk,
                gamma_m_ijk,
                gamma_m_ijk_pqm,
                gamma_p_ijk,
                gamma_p_ijk_pqm,
                zeta_m_ijk,
                zeta_m_ijk_pqm,
                zeta_p_ijk,
                zeta_p_ijk_pqm,
            ) = self.get_regression_coeffs_at_point(point, dx, dy, dz)

            def bulk_point(is_interface_, u_ijk_):
                return jnp.array(
                    [
                        jnp.where(is_interface_ == -1, u_ijk_, 0.0),
                        jnp.where(is_interface_ == 1, u_ijk_, 0.0),
                    ]
                )

            def interface_point(point):
                def mu_minus_bigger_fn(point):
                    def extrapolate_u_m_from_negative_domain(r_ijk):
                        r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                        u_m = -1.0 * jnp.dot(gamma_m_ijk_pqm, u_cube_ijk)
                        u_m += (1.0 - gamma_m_ijk + gamma_m_ijk_pqm[:, 13]) * u_ijk
                        u_m += (
                            -1.0
                            * (1.0 - gamma_m_ijk)
                            * (
                                self.alpha_interp_fn(r_m_proj)
                                + delta_ijk * self.beta_over_mu_p_interp_fn(r_m_proj)
                            )
                        )
                        return u_m.reshape()

                    def extrapolate_u_p_from_positive_domain(r_ijk):
                        r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk[0]
                        u_p = -1.0 * jnp.dot(zeta_m_ijk_pqm, u_cube_ijk)
                        u_p += (1.0 - zeta_m_ijk + zeta_m_ijk_pqm[:, 13]) * u_ijk
                        u_p += self.alpha_interp_fn(
                            r_p_proj
                        ) + delta_ijk * self.beta_over_mu_p_interp_fn(r_p_proj)
                        return u_p.reshape()

                    u_m = jnp.where(
                        delta_ijk > 0,
                        extrapolate_u_m_from_negative_domain(point),
                        u_ijk,
                    )[0]
                    u_p = jnp.where(
                        delta_ijk > 0,
                        u_ijk,
                        extrapolate_u_p_from_positive_domain(point),
                    )[0]
                    return jnp.array([u_m, u_p])

                def mu_plus_bigger_fn(point):
                    def extrapolate_u_m_from_negative_domain_(r_ijk):
                        r_m_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                        u_m = -1.0 * jnp.dot(zeta_p_ijk_pqm, u_cube_ijk)
                        u_m += (1.0 - zeta_p_ijk + zeta_p_ijk_pqm[:, 13]) * u_ijk
                        u_m += (-1.0) * (
                            self.alpha_interp_fn(r_m_proj)
                            + delta_ijk * self.beta_over_mu_m_interp_fn(r_m_proj)
                        )
                        return u_m.reshape()

                    def extrapolate_u_p_from_positive_domain_(r_ijk):
                        r_p_proj = r_ijk[jnp.newaxis] - delta_ijk * normal_ijk
                        u_p = -1.0 * jnp.dot(gamma_p_ijk_pqm, u_cube_ijk)
                        u_p += (1.0 - gamma_p_ijk + gamma_p_ijk_pqm[:, 13]) * u_ijk
                        u_p += (1.0 - gamma_p_ijk) * (
                            self.alpha_interp_fn(r_p_proj)
                            + delta_ijk * self.beta_over_mu_m_interp_fn(r_p_proj)
                        )
                        return u_p.reshape()

                    u_m = jnp.where(
                        delta_ijk > 0,
                        extrapolate_u_m_from_negative_domain_(point),
                        u_ijk,
                    )[0]
                    u_p = jnp.where(
                        delta_ijk > 0,
                        u_ijk,
                        extrapolate_u_p_from_positive_domain_(point),
                    )[0]
                    return jnp.array([u_m, u_p])

                mu_m_ijk = self.mu_m_interp_fn(point[jnp.newaxis])
                mu_p_ijk = self.mu_p_interp_fn(point[jnp.newaxis])
                return jnp.where(
                    mu_m_ijk > mu_p_ijk,
                    mu_minus_bigger_fn(point),
                    mu_plus_bigger_fn(point),
                )

            # 0: crossed by interface, -1: in Omega^-, +1: in Omega^+
            is_interface = self.is_cell_crossed_by_interface(point, dx, dy, dz)
            # is_interface = jnp.where( delta_ijk*delta_ijk <= self.bandwidth_squared,  0, jnp.sign(delta_ijk))
            u_mp = jnp.where(
                is_interface == 0,
                interface_point(point),
                bulk_point(is_interface, u_ijk),
            )
            return u_mp

    def get_u_m_Robin_by_regression_at_point_discrete_fn(
        self, u_interp_fn, dx, dy, dz, point
    ):
        """
        This function gives the value of um at the 27 grid points around u_ijk
        """
        Xijk = self.get_Xijk(dx, dy, dz)
        curr_vertices = jnp.add(point, Xijk)
        u_cube_ijk = u_interp_fn(
            curr_vertices
        )  # Uses the non-oscillatory-interpolation

        return u_cube_ijk
