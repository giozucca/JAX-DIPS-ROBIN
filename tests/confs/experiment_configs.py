from functools import partial

from jax import grad, jit, jvp, lax
from jax import numpy as jnp
from jax import vmap

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.geometry import level_set

import jax
try:
    jax.devices("gpu")
    COMPILE_BACKEND = "gpu"
except RuntimeError:
    COMPILE_BACKEND = "cpu"
custom_jit = partial(jit, backend=COMPILE_BACKEND)

dim = 3


#####################################################
#
#   Sphere Interface with Jump
#
#####################################################
def sphere():
    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.exp(z)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y) * jnp.cos(x)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + y**2 + z**2) - 0.2380952  # 0.5/2.1, normalised box

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return y * y * jnp.log(x + 2.0) + 4.0

    @jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.exp(-1.0 * z)

    @jit
    def alpha_fn(r):
        """
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @jit
    def beta_fn(r):
        r"""
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)


    # @jit
    # def g_fn(r):    # For Robin BC
    #     return 1
    # @jit
    # def alphaRobin(r): # For Robin BC
    #     return 1

    @jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.0

    @jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.0

    @jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # evaluate_exact_solution_fn(r)

    @jit
    def f_m_fn_(r):
        """
        Source function in $\Omega^-$
        """

        def laplacian_m_fn(x):
            grad_m_fn = grad(exact_sol_m_fn)
            flux_m_fn = lambda p: mu_m_fn(p) * grad_m_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_m_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_m_fn(r) * (-1.0)

    @jit
    def f_p_fn_(r):
        """
        Source function in $\Omega^+$
        """

        def laplacian_p_fn(x):
            grad_p_fn = grad(exact_sol_p_fn)
            flux_p_fn = lambda p: mu_p_fn(p) * grad_p_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jax.jvp(flux_p_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_p_fn(r) * (-1.0)



    @jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return -1.0 * jnp.exp(z) * (y * y * jnp.log(x + 2) + 4)

    @jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 2.0 * jnp.exp(-1.0 * z) * jnp.cos(x) * jnp.sin(y)

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    )

#####################################################
#
#   Sphere Interface with Robin
#
#####################################################
def sphere_Robin():
    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    # below add compute normal as a function and use that to calculate g, look at whiteboard 
    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        # Radius 0.6666667 = 1.4/2.1: every Robin geometry shares the NORMALISED
        # [-1, 1] domain and is sized to a common half-extent of 0.666667, so all
        # three have the same dx AND the same number of cells across the object at
        # equal Nx. For phi = |r| - R the scaling is just the radius -- S*phi_ref(r/S)
        # reduces to |r| - S*R -- so no scaling wrapper is needed here, and
        # |grad phi| stays exactly 1 (a true signed distance function).
        return jnp.sqrt(x**2 + y**2 + z**2) - 0.6666667

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 2

    @jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 2

    def computeNormal(phi, r):
        """Normal used to build the exact Robin data g. CENTRAL differences.

        h is 1e-3 here, the same value all three geometries now use. The previous
        3e-4 was arbitrary and slightly WORSE for a central stencil in float32: the
        sphere's normal error at 3e-4 is max 0.02524 deg against 0.02309 at 1e-3,
        because 3e-4 is small enough to be roundoff-limited. Unifying removes a
        constant that would otherwise need explaining in the paper.

        Caveat, recorded honestly: a6042a0 moved this geometry's stencil AND its h
        (3e-4 -> 2e-3) together, after which it collapsed to u ~ 0, and that collapse
        was never explained. Two variables move here too, so a collapse would again be
        ambiguous between them. Accepted because the stencil change is already
        validated on star_Robin and because the remedy either way is to revert this
        geometry to forward differences.

        A one-sided difference has error (h/2)*phi'', which is independent of the grid
        and so never converges. On star_Robin this produced a level offset that was
        93.5% of the mean-square error at Nx=64; switching to central removed it
        entirely at Nx=32 (+2.52e-03 -> -1.34e-04) and cut it 3.7x at Nx=64, while
        leaving the shape error untouched (order 1.03 before and after). This applies
        the same fix here.
        """
        x=r[0]
        y=r[1]
        z=r[2]
        h=3e-4

        rxp=jnp.array([x+h,y,z]); rxm=jnp.array([x-h,y,z])
        ryp=jnp.array([x,y+h,z]); rym=jnp.array([x,y-h,z])
        rzp=jnp.array([x,y,z+h]); rzm=jnp.array([x,y,z-h])

        n1 = (phi(rxp)-phi(rxm))/(2*h)
        n2 = (phi(ryp)-phi(rym))/(2*h)
        n3 = (phi(rzp)-phi(rzm))/(2*h)
        norm = jnp.sqrt(n1**2 + n2**2 + n3**2)

        n1 = n1/norm
        n2 = n2/norm
        n3 = n3/norm

        return n1,n2,n3
    


    # Changes made: moved the U_ext to be included in the division by norm    
    @jit 
    def g_m_fn(r):    # For Robin BC
        x = r[0]
        y = r[1]
        z = r[2]
        
        n1,n2,n3 = computeNormal(unperturbed_phi_fn,r)
        return mu_m_fn(r)*((-jnp.sin(x)*jnp.sin(y)*jnp.cos(z)*n1)+(jnp.cos(x)*jnp.cos(y)*jnp.cos(z)*n2)+(-jnp.cos(x)*jnp.sin(y)*jnp.sin(z)*n3))+alphaRobin(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)


    @jit
    def g_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        n1,n2,n3 = computeNormal(unperturbed_phi_fn,r)
        return mu_m_fn(r)*((-jnp.sin(x)*jnp.sin(y)*jnp.cos(z)*n1)+(jnp.cos(x)*jnp.cos(y)*jnp.cos(z)*n2)+(-jnp.cos(x)*jnp.sin(y)*jnp.sin(z)*n3))+alphaRobin(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)




    @jit
    def alphaRobin(r): # For Robin BC
        return 0.25 # changed from 1

    @jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.5 #same as below

    @jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.5 # changed from 0

    @jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # evaluate_exact_solution_fn(r)

    @jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 3*mu_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)+ k_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z) # Changed from -6 to 6 # same updates as below

    @jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 3*mu_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)+ k_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z) # changed from hardcoded to 3*mu + k*...

    @jit
    def beta_fn(r):
        # r"""
        # Jump in flux at interface
        # """
        # normal_fn = grad(phi_fn)
        # grad_u_p_fn = grad(exact_sol_p_fn)
        # grad_u_m_fn = grad(exact_sol_m_fn)

        # vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        # vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        # n_vec = normal_fn(r)
        # return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)
        return 0.0

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alphaRobin,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
        g_m_fn,
        g_p_fn,
        beta_fn
    )
# We have stopped here.
# We got rid of some stuff (beta, rename dirichlet), added g+ and g- (need to add to the return)
# Make sure to look at cfg and setup robin bc cfg before you run (otherwise it will blow up)

#####################################################
#
#   Star Interface with Robin
#
#####################################################
def star_Robin():
    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    # below add compute normal as a function and use that to calculate g, look at whiteboard 
    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        # Scaled to the common half-extent of 0.666667 shared by all three Robin
        # geometries on the normalised [-1, 1] box (see the domain comment in
        # test_poisson.py). The reference shape measures 1.3100, so the factor is
        # (1.40 / 1.3100) / 2.1 = 1.0687 / 2.1. The transformation
        #     phi(r) = S * phi_ref(r / S)
        # moves the zero level set by exactly S while leaving |grad phi| unchanged,
        # so the signed distance phi/|grad phi| that the Robin projection uses scales
        # with the geometry. Folding the domain factor into the SAME wrapper is what
        # keeps the shape identical: rewriting the internal constants (the 1e-8 taper
        # guard, the rho^2/10 length) would not, since those are not scale-free.
        S = 0.5089048
        x = r[0] / S
        y = r[1] / S
        z = r[2] / S
        beta1, beta2, beta3 = -0.05, 0.05, -0.10
        theta1, theta2, theta3 = 0.05, 0.05, 0.05
        n1, n2, n3 = 3, 4, 3

        # Intermediate coordinate mappings
        rho2 = x**2 + y**2
        r2 = x**2 + y**2 + z**2
        azimuth = jnp.arctan2(y, x)

        # Calculate perturbations with the generalized polar taper
        perturbation = (
            beta1 * (rho2 / (r2 + 1e-8))**(n1 / 2.0) * jnp.cos(n1 * (azimuth - theta1)) +
            beta2 * (rho2 / (r2 + 1e-8))**(n2 / 2.0) * jnp.cos(n2 * (azimuth - theta2)) +
            beta3 * (rho2 / (r2 + 1e-8))**(n3 / 2.0) * jnp.cos(n3 * (azimuth - theta3))
        )

        # Final level-set / signed distance function
        phi = jnp.sqrt(r2) - 1.183 * (1.0 + (rho2 / 10.0)**2) + perturbation
        return S * phi
        # beta1 = -0.05 
        # beta2 = 0.05 
        # beta3 = -.1
        # n1 = 4
        # n2 = 4
        # n3 = 4
        # theta1 = 0.5
        # theta2 = 0.5
        # theta3 = 0.5
        # # return (jnp.sqrt(x**2 + y**2 + z**2) -
        # # 1.183 * (1 + ((x**2 + y**2)/10)**2) +
        # # (((x**2 + y**2) / (x**2 + y**2 + z**2 + 1e-8)) * 
        # #  ((beta1 * jnp.cos(n1 * (jnp.arctan2(y,x) - theta1))) +
        # #   (beta2 * jnp.cos(n2 * (jnp.arctan2(y,x) - theta2))) +
        # #   (beta3 * jnp.cos(n3 * (jnp.arctan2(y,x) - theta3))))))

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 2

    @jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 2

    def computeNormal(phi, r):
        """Normal used to build the exact Robin data g. CENTRAL differences.

        ONE VARIABLE CHANGED vs the previous version: the stencil, forward -> central.
        The step h stays at 1e-3 and no other geometry is touched, deliberately.
        d8fcb85 ("stencil in normals") changed the stencil AND, for sphere_Robin only,
        h from 3e-4 to 2e-3 at the same time; sphere_Robin then collapsed to u ~ 0
        (RMSE 0.208 against rms|u_exact| 0.212 -- the network sat at its
        initialisation) and 501c234 reverted the lot. Because those two moved
        together, that result does not tell us which one did the damage. Keeping h
        fixed here, and leaving sphere_Robin and star_Robin3 on forward differences,
        isolates the stencil for the one geometry we have measurements for.

        WHY IT MIGHT MATTER: a forward difference has error (h/2)*phi'', which does
        NOT shrink with the grid and grows with curvature. The Star error on [-1,1] at
        Nx=64 is 93.5% a constant offset of +2.71e-03 whose shape component converges
        cleanly at order 1.03, so the offset is not a discretisation or capacity
        limit. It also grew 1.72x when the domain shrank 2.1x, against 2.10x predicted
        from curvature scaling alone.

        WHAT WOULD FALSIFY IT: d8fcb85 measured that central-vs-forward shifts g by
        only ~0.05%, while the offset is ~1% of |u|. That needs roughly 20x
        amplification, which is NOT established. If the offset does not move, this
        candidate is dead and the offset lives elsewhere in the Robin data.
        """
        x=r[0]
        y=r[1]
        z=r[2]
        h=1e-3

        rxp=jnp.array([x+h,y,z]); rxm=jnp.array([x-h,y,z])
        ryp=jnp.array([x,y+h,z]); rym=jnp.array([x,y-h,z])
        rzp=jnp.array([x,y,z+h]); rzm=jnp.array([x,y,z-h])

        n1 = (phi(rxp)-phi(rxm))/(2*h)
        n2 = (phi(ryp)-phi(rym))/(2*h)
        n3 = (phi(rzp)-phi(rzm))/(2*h)
        norm = jnp.sqrt(n1**2 + n2**2 + n3**2)

        n1 = n1/norm
        n2 = n2/norm
        n3 = n3/norm

        return n1,n2,n3

    # Changes made: moved the U_ext to be included in the division by norm
    @jit 
    def g_m_fn(r):    # For Robin BC
        x = r[0]
        y = r[1]
        z = r[2]
        n1,n2,n3 = computeNormal(unperturbed_phi_fn,r)
        return mu_m_fn(r)*((-jnp.sin(x)*jnp.sin(y)*jnp.cos(z)*n1)+(jnp.cos(x)*jnp.cos(y)*jnp.cos(z)*n2)+(-jnp.cos(x)*jnp.sin(y)*jnp.sin(z)*n3))+alphaRobin(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def g_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        n1,n2,n3 = computeNormal(unperturbed_phi_fn,r)
        return mu_m_fn(r)*((-jnp.sin(x)*jnp.sin(y)*jnp.cos(z)*n1)+(jnp.cos(x)*jnp.cos(y)*jnp.cos(z)*n2)+(-jnp.cos(x)*jnp.sin(y)*jnp.sin(z)*n3))+alphaRobin(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)


    @jit
    def alphaRobin(r): # For Robin BC
        return 0.25

    @jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.5

    @jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.5

    @jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # evaluate_exact_solution_fn(r)

    @jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 3*mu_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)+ k_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z) # Changed from -6 to 6

    @jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 3*mu_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)+ k_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z) # Changed from -6 to 6

    @jit
    def beta_fn(r):
        # r"""
        # Jump in flux at interface
        # """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alphaRobin,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
        g_m_fn,
        g_p_fn,
        beta_fn
    )

#####################################################
#
#   Irregular Domain 3 (Min of two stars) with Robin
#
#####################################################
def star_Robin3():
    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface: union of two star domains (Form A)

        Wrapped as phi(r) = S * phi_ref(r / S) with S = 1/2.1 to place the natural
        half-extent of 1.4000 at 0.666667 on the normalised [-1, 1] box. Evaluating
        the UNCHANGED reference shape at r/S is what makes this exact: the lobe
        centres, the 1e-8 taper guard and the rho^2/10 length all scale with it
        automatically, which rewriting them in place would not achieve. min() is
        positively homogeneous, so the wrapper passes through the union unharmed and
        |grad phi| is unchanged.
        """
        S = 0.4761905
        x = r[0] / S
        y = r[1] / S
        z = r[2] / S
        
        # Domain 1 parameters & coordinates
        # xc1, yc1, zc1 = -0.75, 0.75, -0.75
        # X1 = x - xc1
        # Y1 = y - yc1
        # Z1 = z - zc1
        # beta1_1 = -0.05
        # beta2_1 = 0.05
        # beta3_1 = -0.1
        # n1_1 = 5
        # n2_1 = 5
        # n3_1 = 5
        # theta1_1 = 0.5
        # theta2_1 = 0.5
        # theta3_1 = 0.5
        
        # phi1 = (jnp.sqrt(X1**2 + Y1**2 + Z1**2) - 
        #         0.783 * (1.0 + ((X1**2 + Y1**2)/10.0)**2) + 
        #         beta1_1 * jnp.cos(n1_1 * (jnp.arctan2(Y1, X1) - theta1_1)) + 
        #         beta2_1 * jnp.cos(n2_1 * (jnp.arctan2(Y1, X1) - theta2_1)) + 
        #         beta3_1 * jnp.cos(n3_1 * (jnp.arctan2(Y1, X1) - theta3_1)))
                
        # # Domain 2 parameters & coordinates
        # xc2, yc2, zc2 = 0.75, -0.75, 0.75
        # X2 = x - xc2
        # Y2 = y - yc2
        # Z2 = z - zc2
        # beta1_2 = 0.15
        # beta2_2 = -0.01
        # beta3_2 = 0.07
        # n1_2 = 5
        # n2_2 = 5
        # n3_2 = 5
        # theta1_2 = 0.5
        # theta2_2 = 1.8
        # theta3_2 = 0.0
        
        # phi2 = (jnp.sqrt(X2**2 + Y2**2 + Z2**2) - 
        #         0.783 * (1.0 + ((X2**2 + Y2**2)/10.0)**2) + 
        #         beta1_2 * jnp.cos(n1_2 * (jnp.arctan2(Y2, X2) - theta1_2)) + 
        #         beta2_2 * jnp.cos(n2_2 * (jnp.arctan2(Y2, X2) - theta2_2)) + 
        #         beta3_2 * jnp.cos(n3_2 * (jnp.arctan2(Y2, X2) - theta3_2)))

        # return jnp.minimum(phi1, phi2)
        # -------------------------------------------------------------
        # Domain 1
        # -------------------------------------------------------------
        xc1, yc1, zc1 = -0.5, 0.5, -0.5
        X1 = x - xc1
        Y1 = y - yc1
        Z1 = z - zc1

        beta1_1, beta2_1, beta3_1 = -0.05, 0.05, -0.10
        n1_1, n2_1, n3_1 = 4, 4, 4
        theta1_1, theta2_1, theta3_1 = 0.05, 0.05, 0.05

        rho2_1 = X1**2 + Y1**2
        r2_1 = X1**2 + Y1**2 + Z1**2
        azimuth1 = jnp.arctan2(Y1, X1)

        # Smooth angular factor (since n=4, this is taper^2)
        smoothFactor1 = (rho2_1 / (r2_1 + 1e-8))**2

        perturbation1 = smoothFactor1 * (
            beta1_1 * jnp.cos(n1_1 * (azimuth1 - theta1_1)) +
            beta2_1 * jnp.cos(n2_1 * (azimuth1 - theta2_1)) +
            beta3_1 * jnp.cos(n3_1 * (azimuth1 - theta3_1))
        )

        phi1 = jnp.sqrt(r2_1) - 0.8 * (1.0 + (rho2_1 / 10.0)**2) + perturbation1


        # -------------------------------------------------------------
        # Domain 2
        # -------------------------------------------------------------
        xc2, yc2, zc2 = 0.5, -0.5, 0.5
        X2 = x - xc2
        Y2 = y - yc2
        Z2 = z - zc2

        beta1_2, beta2_2, beta3_2 = -0.05, 0.05, -0.10
        n1_2, n2_2, n3_2 = 4, 4, 4
        theta1_2, theta2_2, theta3_2 = 0.05, 0.05, 0.05

        rho2_2 = X2**2 + Y2**2
        r2_2 = X2**2 + Y2**2 + Z2**2
        azimuth2 = jnp.arctan2(Y2, X2)

        # Smooth angular factor (since n=4, this is taper^2)
        smoothFactor2 = (rho2_2 / (r2_2 + 1e-8))**2

        perturbation2 = smoothFactor2 * (
            beta1_2 * jnp.cos(n1_2 * (azimuth2 - theta1_2)) +
            beta2_2 * jnp.cos(n2_2 * (azimuth2 - theta2_2)) +
            beta3_2 * jnp.cos(n3_2 * (azimuth2 - theta3_2))
        )

        phi2 = jnp.sqrt(r2_2) - 0.8 * (1.0 + (rho2_2 / 10.0)**2) + perturbation2


        # -------------------------------------------------------------
        # Union of the two domains
        # -------------------------------------------------------------
        return S * jnp.minimum(phi1, phi2)

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        return 2.0

    @jit
    def mu_p_fn(r):
        return 2.0

    def computeNormal(phi, r):
        """Normal used to build the exact Robin data g. CENTRAL differences.

        Stencil only: h stays at 1e-3, matching this geometry's previous value, so the
        forward -> central switch is the single variable changed. h matches star_Robin, where this change was validated.

        A one-sided difference has error (h/2)*phi'', which is independent of the grid
        and so never converges. On star_Robin this produced a level offset that was
        93.5% of the mean-square error at Nx=64; switching to central removed it
        entirely at Nx=32 (+2.52e-03 -> -1.34e-04) and cut it 3.7x at Nx=64, while
        leaving the shape error untouched (order 1.03 before and after). This applies
        the same fix here.
        """
        x=r[0]
        y=r[1]
        z=r[2]
        h=1e-3

        rxp=jnp.array([x+h,y,z]); rxm=jnp.array([x-h,y,z])
        ryp=jnp.array([x,y+h,z]); rym=jnp.array([x,y-h,z])
        rzp=jnp.array([x,y,z+h]); rzm=jnp.array([x,y,z-h])

        n1 = (phi(rxp)-phi(rxm))/(2*h)
        n2 = (phi(ryp)-phi(rym))/(2*h)
        n3 = (phi(rzp)-phi(rzm))/(2*h)
        norm = jnp.sqrt(n1**2 + n2**2 + n3**2)

        n1 = n1/norm
        n2 = n2/norm
        n3 = n3/norm

        return n1,n2,n3

    @jit 
    def g_m_fn(r):    # For Robin BC
        x = r[0]
        y = r[1]
        z = r[2]
        n1,n2,n3 = computeNormal(unperturbed_phi_fn,r)
        return mu_m_fn(r)*((-jnp.sin(x)*jnp.sin(y)*jnp.cos(z)*n1)+(jnp.cos(x)*jnp.cos(y)*jnp.cos(z)*n2)+(-jnp.cos(x)*jnp.sin(y)*jnp.sin(z)*n3))+alphaRobin(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def g_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        n1,n2,n3 = computeNormal(unperturbed_phi_fn,r)
        return mu_m_fn(r)*((-jnp.sin(x)*jnp.sin(y)*jnp.cos(z)*n1)+(jnp.cos(x)*jnp.cos(y)*jnp.cos(z)*n2)+(-jnp.cos(x)*jnp.sin(y)*jnp.sin(z)*n3))+alphaRobin(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def alphaRobin(r): # For Robin BC
        return 0.25

    @jit
    def k_m_fn(r):
        return 0.5

    @jit
    def k_p_fn(r):
        return 0.5

    @jit
    def initial_value_fn(r):
        return 0.0

    @jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 3*mu_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)+ k_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 3*mu_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)+ k_m_fn(r)*jnp.cos(x)*jnp.sin(y)*jnp.cos(z)

    @jit
    def beta_fn(r):
        # r"""
        # Jump in flux at interface
        # """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec) * (-1.0)
        # return 0.0

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alphaRobin,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
        g_m_fn,
        g_p_fn,
        beta_fn
    )

#####################################################
#
#   Star Interface with Jump
#
#####################################################


def star():
    """Star interface with jump conditions"""

    # -- 3d example according to 4.6 in Guittet 2015 (VIM) paper
    @custom_jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.exp(z)

    @custom_jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        yx3 = (y - x) / 3.0
        return (16.0 * yx3**5 - 20.0 * yx3**3 + 5.0 * yx3) * jnp.log(x + y + 3) * jnp.cos(z)

    @custom_jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @custom_jit
    def unperturbed_phi_fn(r):
        r"""
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]

        r0 = 0.483
        ri = 0.151
        re = 0.911
        n_1 = 3.0
        beta_1 = 0.1
        theta_1 = 0.5
        n_2 = 4.0
        beta_2 = -0.1
        theta_2 = 1.8
        n_3 = 7.0
        beta_3 = 0.15
        theta_3 = 0.0

        core = beta_1 * jnp.cos(n_1 * (jnp.arctan2(y, x) - theta_1))
        core += beta_2 * jnp.cos(n_2 * (jnp.arctan2(y, x) - theta_2))
        core += beta_3 * jnp.cos(n_3 * (jnp.arctan2(y, x) - theta_3))

        phi_ = jnp.sqrt(x**2 + y**2 + z**2)
        phi_ += -1.0 * r0 * (1.0 + ((x**2 + y**2) / (x**2 + y**2 + z**2)) ** 2 * core)

        return jnp.nan_to_num(phi_, -r0 * core)

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @custom_jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @custom_jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 10.0 * (1 + 0.2 * jnp.cos(2 * jnp.pi * (x + y)) * jnp.sin(2 * jnp.pi * (x - y)) * jnp.cos(z))

    @custom_jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @custom_jit
    def alpha_fn(r):
        r"""
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)

    @custom_jit
    def beta_fn(r):
        r"""
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.nan_to_num(jnp.dot(vec_1 - vec_2, n_vec) * (-1.0))

    @custom_jit
    def g_fn(r):
        return 1
    @custom_jit
    def alpha_robin(r):
        return 1
    @custom_jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.0

    @custom_jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.0

    @custom_jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # evaluate_exact_solution_fn(r)

    @custom_jit
    def f_m_fn_(r):
        r"""
        Source function in $\Omega^-$
        """

        def laplacian_m_fn(x):
            grad_m_fn = grad(exact_sol_m_fn)
            flux_m_fn = lambda p: mu_m_fn(p) * grad_m_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jvp(flux_m_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_m_fn(r) * (-1.0)

    @custom_jit
    def f_p_fn_(r):
        r"""
        Source function in $\Omega^+$
        """

        def laplacian_p_fn(x):
            grad_p_fn = grad(exact_sol_p_fn)
            flux_p_fn = lambda p: mu_p_fn(p) * grad_p_fn(p)
            eye = jnp.eye(dim, dtype=f32)

            def _body_fun(i, val):
                primal, tangent = jvp(flux_p_fn, (x,), (eye[i],))
                return val + primal[i] ** 2 + tangent[i]

            return lax.fori_loop(i32(0), i32(dim), _body_fun, 0.0)

        return laplacian_p_fn(r) * (-1.0)

    @custom_jit
    def f_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        fm = (
            -1.0 * mu_m_fn(r) * (-7.0 * jnp.sin(2.0 * x) * jnp.cos(2.0 * y) * jnp.exp(z))
            + -4 * jnp.pi * jnp.cos(z) * jnp.cos(4 * jnp.pi * x) * 2 * jnp.cos(2 * x) * jnp.cos(2 * y) * jnp.exp(z)
            + -4 * jnp.pi * jnp.cos(z) * jnp.cos(4 * jnp.pi * y) * (-2) * jnp.sin(2 * x) * jnp.sin(2 * y) * jnp.exp(z)
            + 2
            * jnp.cos(2 * jnp.pi * (x + y))
            * jnp.sin(2 * jnp.pi * (x - y))
            * jnp.sin(z)
            * jnp.sin(2 * x)
            * jnp.cos(2 * y)
            * jnp.exp(z)
        )

        return fm

    @custom_jit
    def f_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        f_p = -1.0 * (
            (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * (y - x) / 3)
            * (-2)
            * jnp.cos(z)
            / (x + y + 3) ** 2
            + 2
            * (16 * 5 * 4 * (1.0 / 9.0) * ((y - x) / 3) ** 3 - 20 * 3 * 2 * (1.0 / 9.0) * ((y - x) / 3))
            * jnp.log(x + y + 3)
            * jnp.cos(z)
            + -1
            * (16 * ((y - x) / 3) ** 5 - 20 * ((y - x) / 3) ** 3 + 5 * ((y - x) / 3))
            * jnp.log(x + y + 3)
            * jnp.cos(z)
        )
        return f_p

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    )


#####################################################
#
#   No interface jump
#
#####################################################
def no_jump():
    """No interface jump"""

    @jit
    def exact_sol_m_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y) * jnp.cos(x) * jnp.cos(z)

    @jit
    def exact_sol_p_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sin(y) * jnp.cos(x) * jnp.cos(z)

    @jit
    def dirichlet_bc_fn(r):
        return exact_sol_p_fn(r)

    @jit
    def unperturbed_phi_fn(r):
        r"""
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return jnp.sqrt(x**2 + y**2 + z**2) + 0.5

    phi_fn = level_set.perturb_level_set_fn(unperturbed_phi_fn)

    @jit
    def evaluate_exact_solution_fn(r):
        return jnp.where(phi_fn(r) >= 0, exact_sol_p_fn(r), exact_sol_m_fn(r))

    @jit
    def mu_m_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @jit
    def mu_p_fn(r):
        r"""
        Diffusion coefficient function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 1.0

    @jit
    def alpha_fn(r):
        r"""
        Jump in solution at interface
        """
        return exact_sol_p_fn(r) - exact_sol_m_fn(r)
    @jit
    def alpha_robin(r):
        return 1

    @jit
    def beta_fn(r):
        r"""
        Jump in flux at interface
        """
        normal_fn = grad(phi_fn)
        grad_u_p_fn = grad(exact_sol_p_fn)
        grad_u_m_fn = grad(exact_sol_m_fn)

        vec_1 = mu_p_fn(r) * grad_u_p_fn(r)
        vec_2 = mu_m_fn(r) * grad_u_m_fn(r)
        n_vec = normal_fn(r)
        return jnp.dot(vec_1 - vec_2, n_vec)

    @jit
    def g_fn(r):
        return 1
    @jit
    def k_m_fn(r):
        r"""
        Linear term function in $\Omega^-$
        """
        return 0.0

    @jit
    def k_p_fn(r):
        r"""
        Linear term function in $\Omega^+$
        """
        return 0.0

    @jit
    def initial_value_fn(r):
        x = r[0]
        y = r[1]
        z = r[2]
        return y
        # return exact_sol_p_fn(r)   # PAM: testing

    @jit
    def f_m_fn(r):
        r"""
        Source function in $\Omega^-$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 0.0  # 2.0 * jnp.sin(y) * jnp.cos(x)

    @jit
    def f_p_fn(r):
        r"""
        Source function in $\Omega^+$
        """
        x = r[0]
        y = r[1]
        z = r[2]
        return 3.0 * jnp.sin(y) * jnp.cos(x) * jnp.cos(z)

    return (
        initial_value_fn,
        dirichlet_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    )
