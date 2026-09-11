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
        return jnp.sqrt(x**2 + y**2 + z**2) - 0.5

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
        return jnp.sqrt(x**2 + y**2 + z**2) - 0.5

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
        x=r[0]
        y=r[1]
        z=r[2]
        h=3e-4
        
        r1=jnp.array([x+h,y,z])
        r2=jnp.array([x,y+h,z])
        r3=jnp.array([x,y,z+h])

        
        n1 = (phi(r1)-phi(r))/h
        n2 = (phi(r2)-phi(r))/h
        n3 = (phi(r3)-phi(r))/h
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
        x = r[0]
        y = r[1]
        z = r[2]
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
        return phi
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
        x=r[0]
        y=r[1]
        z=r[2]
        h=1e-3  
        
        r1=jnp.array([x+h,y,z])
        r2=jnp.array([x,y+h,z])
        r3=jnp.array([x,y,z+h])

        
        n1 = (phi(r1)-phi(r))/h
        n2 = (phi(r2)-phi(r))/h
        n3 = (phi(r3)-phi(r))/h
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
        """
        x = r[0]
        y = r[1]
        z = r[2]
        
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
        return jnp.minimum(phi1, phi2)

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
        x=r[0]
        y=r[1]
        z=r[2]
        h=1e-3
        
        r1=jnp.array([x+h,y,z])
        r2=jnp.array([x,y+h,z])
        r3=jnp.array([x,y,z+h])

        n1 = (phi(r1)-phi(r))/h
        n2 = (phi(r2)-phi(r))/h
        n3 = (phi(r3)-phi(r))/h
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
