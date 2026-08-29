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
from tests.confs.experiment_configs import star_Robin, star_Robin3
import logging
import os
import sys
import time
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import jax
import jax.profiler
from jax import grad, jit, lax
from jax import numpy as jnp
from jax import vmap
try:
    from jax.config import config
except ImportError:
    from jax import config

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.domain import mesh
from jax_dips.geometry import level_set
from jax_dips.solvers.optimizers import get_optimizer
from jax_dips.solvers.poisson import trainer
from jax_dips.solvers.poisson.deprecated import poisson_solver_scalable
from jax_dips.utils import io
from tests.confs.experiment_configs import no_jump, sphere, sphere_Robin, star

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, ".."))
if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


@hydra.main(config_path="confs", config_name="poisson", version_base="1.1")
def test_poisson(cfg: DictConfig):
    logger.info(f"Starting {__file__}")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.experiment.sphere:
        logger.info("Performing sphere experiment...\n")
        poisson_solve(
            cfg,
            test_name="sphere",
            exp_fn=sphere,
        )
    if cfg.experiment.star:
        logger.info("Performing star experiment...\n")
        poisson_solve(
            cfg,
            test_name="star",
            exp_fn=star,
        )
    if cfg.experiment.no_jump:
        logger.info("Performing bulk/no jump experiment...\n")
        poisson_solve(
            cfg,
            test_name="no_jump",
            exp_fn=no_jump,
        )
    if cfg.experiment.sphere_Robin:
        logger.info("Performing sphere Robin B.C experiment...\n")
        poisson_solve_Robin(cfg, test_name=f"sphere_Robin_{cfg.solver.Nx_tr}", exp_fn=sphere_Robin)
    
    if cfg.experiment.star_Robin:
        logger.info("Performing star robin B.C experimenet...\n")
        poisson_solve_Robin(cfg, test_name=f"star_Robin_{cfg.solver.Nx_tr}", exp_fn=star_Robin)

    if cfg.experiment.star_Robin3:
        logger.info("Performing star robin 3 B.C experiment...\n")
        poisson_solve_Robin(cfg, test_name=f"star_Robin3_{cfg.solver.Nx_tr}", exp_fn=star_Robin3)


def create_dirs(
    results_path: str,
    test_name: str,
):
    results_path = os.path.join(results_path, test_name)
    os.path.exists(results_path) or os.makedirs(results_path)
    return results_path


def poisson_solve(
        cfg: DictConfig,
        test_name: str,
        exp_fn: object,
):
    results_path = create_dirs(results_path=cfg.experiment.results_path, test_name=test_name)
    checkpoint_dir = os.path.join(results_path, "checkpoints")
    checkpoint_interval = cfg.experiment.logging.checkpoint_interval

    algorithm = cfg.solver.algorithm
    multi_gpu = cfg.solver.multi_gpu
    num_epochs = cfg.solver.num_epochs
    batch_size = cfg.solver.batch_size

    dim = i32(3)
    if "star_Robin3" in test_name:
        # Union (jnp.minimum) of two stars, each of base radius 0.8, centered at
        # (-0.5, 0.5, -0.5) and (0.5, -0.5, 0.5) -- both at distance 0.5*sqrt(3) from the origin:
        # worst-case extent from the origin is ~0.5*sqrt(3) + (star's own max radius ~1.008) ~= 1.87.
        # Keep a modest safety margin (previously exactly [-2,2], i.e. ~0.13 margin) without shrinking it.
        xmin = ymin = zmin = f32(-2.1)
        xmax = ymax = zmax = f32(2.1)
    elif "star_Robin" in test_name:
        # Single star, base radius 1.183, harmonics bounded by |beta1|+|beta2|+|beta3|=0.20:
        # solving R = 1.183*(1+(R^2/10)^2) + 0.20 gives a worst-case extent R_max ~= 1.433.
        # [-2,2] left ~40% of each axis empty; tighten to concentrate training points on the object.
        xmin = ymin = zmin = f32(-1.8)
        xmax = ymax = zmax = f32(1.8)
    else:
        xmin = ymin = zmin = f32(-1.0)
        xmax = ymax = zmax = f32(1.0)
    init_mesh_fn, coord_at = mesh.construct(dim)

    # --------- Grid nodes for training

    xc = jnp.linspace(xmin, xmax, cfg.solver.Nx_tr, dtype=f32)
    yc = jnp.linspace(ymin, ymax, cfg.solver.Ny_tr, dtype=f32)
    zc = jnp.linspace(zmin, zmax, cfg.solver.Nz_tr, dtype=f32)
    gstate_tr = init_mesh_fn(xc, yc, zc)

    # --------- Grid nodes for level set
    Nx_lvl = cfg.gridstates.Nx_lvl
    Ny_lvl = cfg.gridstates.Ny_lvl
    Nz_lvl = cfg.gridstates.Nz_lvl
    xc = jnp.linspace(xmin, xmax, Nx_lvl, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny_lvl, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz_lvl, dtype=f32)
    gstate_lvl = init_mesh_fn(xc, yc, zc)

    # ----------  Evaluation Mesh for Visualization
    Nx_eval = cfg.gridstates.Nx_eval
    Ny_eval = cfg.gridstates.Ny_eval
    Nz_eval = cfg.gridstates.Nz_eval
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    # ----------  Set up current experiment
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
        alpha_fn,
        beta_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
    ) = exp_fn()

    # ----------- Set up optimizer
    optimizer = get_optimizer(
        optimizer_name=cfg.solver.optim.optimizer_name,
        scheduler_name=cfg.solver.sched.scheduler_name,
        learning_rate=cfg.solver.optim.learning_rate,
        decay_rate=cfg.solver.sched.decay_rate,
        max_norm=1.0,
    )

    optimizer_dict = {
        "optimizer_name": cfg.solver.optim.optimizer_name,
        "learning_rate": cfg.solver.optim.learning_rate,
        "sched": cfg.solver.sched,
    }




    # --------- Set up first version solver
    if False:
        init_fn, solve_fn = poisson_solver_scalable.setup(
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
        )
        sim_state = init_fn(gstate_tr.R)
        t1 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(
            gstate=gstate_tr,
            eval_gstate=eval_gstate,
            sim_state=sim_state,
            algorithm=0,
            switching_interval=3,
            Nx_tr=cfg.solver.Nx_tr,
            Ny_tr=cfg.solver.Ny_tr,
            Nz_tr=cfg.solver.Nz_tr,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            currDir=results_path,
            print_rate=cfg.solver.print_rate,
        )
        t2 = time.time()
    else:
        # --------- Set up second version solver
        init_fn = trainer.setup(
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
        )

        model_dict = OmegaConf.to_container(cfg.model, resolve=True) if "model" in cfg else None
        sim_state, solve_fn = init_fn(
            lvl_gstate=gstate_lvl,
            tr_gstate=gstate_tr,
            eval_gstate=eval_gstate,
            algorithm=algorithm,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_path,
            loss_plot_name=test_name,
            optimizer_dict=optimizer_dict,
            restart=cfg.solver.restart_from_checkpoint,
            restart_checkpoint_dir=cfg.solver.restart_checkpoint_dir,
            print_rate=cfg.solver.print_rate,
            model_dict=model_dict,
        )
        t1 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
        t2 = time.time()

    logger.info(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile(f"{results_path}/memory_poisson_test_{test_name}.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    exact_sol = vmap(evaluate_exact_solution_fn)(eval_gstate.R)
    error = sim_state.solution - exact_sol
    log = {
        "phi": eval_phi,
        "U": sim_state.solution,
        "U_exact": exact_sol,
        "U-U_exact": error,
    }
    io.write_vtk_manual(
        eval_gstate,
        log,
        filename=os.path.join(results_path, test_name),
    )

    # log = {
    #     'phi': sim_state.phi,
    #     'U': sim_state.solution,
    #     'U_exact': exact_sol,
    #     'U-U_exact': sim_state.solution - exact_sol,
    #     'alpha': sim_state.alpha,
    #     'beta': sim_state.beta,
    #     'mu_m': sim_state.mu_m,
    #     'mu_p': sim_state.mu_p,
    #     'f_m': sim_state.f_m,
    #     'f_p': sim_state.f_p,
    #     'grad_um_x': sim_state.grad_solution[0][:,0],
    #     'grad_um_y': sim_state.grad_solution[0][:,1],
    #     'grad_um_z': sim_state.grad_solution[0][:,2],
    #     'grad_up_x': sim_state.grad_solution[1][:,0],
    #     'grad_up_y': sim_state.grad_solution[1][:,1],
    #     'grad_up_z': sim_state.grad_solution[1][:,2],
    #     'grad_um_n': sim_state.grad_normal_solution[0],
    #     'grad_up_n': sim_state.grad_normal_solution[1]
    # }
    # io.write_vtk_manual(gstate, log)

    rms_err = jnp.square(sim_state.solution - exact_sol).mean() ** 0.5
    L_inf_err = abs(sim_state.solution - exact_sol).max()
    L2_err = jnp.sqrt(((sim_state.solution - exact_sol) ** 2).sum())
    L2_rel_loss = jnp.sqrt(((sim_state.solution - exact_sol) ** 2).sum() / (exact_sol**2).sum())

    logger.info(
        f"Accuracy: \n L_inf : {L_inf_err} \n \n L_2 : {L2_err} \n Rel. L_2 : {L2_rel_loss} \n RMSE Loss: {rms_err}"
    )
    logger.info(f"Experiment {test_name} completed! \n")

    """
    MASK the solution over sphere only
    """
    """
    logger.info("\n GRADIENT ERROR\n")

    grad_um = sim_state.grad_solution[0].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up = sim_state.grad_solution[1].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_exact = vmap(grad(exact_sol_m_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up_exact = vmap(grad(exact_sol_p_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    mask_m = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 #-0.5*dx
    err_x_m = abs(grad_um[mask_m][:,0] - grad_um_exact[mask_m][:,0]).max()
    err_y_m = abs(grad_um[mask_m][:,1] - grad_um_exact[mask_m][:,1]).max()
    err_z_m = abs(grad_um[mask_m][:,2] - grad_um_exact[mask_m][:,2]).max()

    mask_p = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 #0.5*dx
    err_x_p = abs(grad_up[mask_p][:,0] - grad_up_exact[mask_p][:,0]).max()
    err_y_p = abs(grad_up[mask_p][:,1] - grad_up_exact[mask_p][:,1]).max()
    err_z_p = abs(grad_up[mask_p][:,2] - grad_up_exact[mask_p][:,2]).max()

    logger.info(f"L_inf errors in grad u in Omega_minus x: {err_x_m}, \t y: {err_y_m}, \t z: {err_z_m}")
    logger.info(f"L_inf errors in grad u in Omega_plus  x: {err_x_p}, \t y: {err_y_p}, \t z: {err_z_p}")



    #--- normal gradients over interface
    normal_fn = grad(phi_fn)
    normal_vec = vmap(normal_fn)(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_n = sim_state.grad_normal_solution[0].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]
    grad_up_n = sim_state.grad_normal_solution[1].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]

    mask_i_m = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 )
    mask_i_p = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 )

    grad_um_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_um_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))
    grad_up_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_up_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))

    err_um_n = abs(grad_um_n - grad_um_n_exact)[mask_i_m].max()
    err_up_n = abs(grad_up_n - grad_up_n_exact)[mask_i_p].max()


    logger.info(f"L_inf error in normal grad u on interface minus: {err_um_n} \t plus: {err_up_n}")

    #----
    assert L_inf_err<0.2

    """

def poisson_solve_Robin(
        cfg: DictConfig,
        test_name: str,
        exp_fn: object,
):
    results_path = create_dirs(results_path=cfg.experiment.results_path, test_name=test_name)
    checkpoint_dir = os.path.join(results_path, "checkpoints")
    checkpoint_interval = cfg.experiment.logging.checkpoint_interval

    algorithm = cfg.solver.algorithm
    multi_gpu = cfg.solver.multi_gpu
    num_epochs = cfg.solver.num_epochs
    batch_size = cfg.solver.batch_size

    dim = i32(3)
    if "star_Robin3" in test_name:
        # Union (jnp.minimum) of two stars, each of base radius 0.8, centered at
        # (-0.5, 0.5, -0.5) and (0.5, -0.5, 0.5) -- both at distance 0.5*sqrt(3) from the origin:
        # worst-case extent from the origin is ~0.5*sqrt(3) + (star's own max radius ~1.008) ~= 1.87.
        # Keep a modest safety margin (previously exactly [-2,2], i.e. ~0.13 margin) without shrinking it.
        xmin = ymin = zmin = f32(-2.1)
        xmax = ymax = zmax = f32(2.1)
    elif "star_Robin" in test_name:
        # Single star, base radius 1.183, harmonics bounded by |beta1|+|beta2|+|beta3|=0.20:
        # solving R = 1.183*(1+(R^2/10)^2) + 0.20 gives a worst-case extent R_max ~= 1.433.
        # [-2,2] left ~40% of each axis empty; tighten to concentrate training points on the object.
        xmin = ymin = zmin = f32(-1.8)
        xmax = ymax = zmax = f32(1.8)
    else:
        xmin = ymin = zmin = f32(-1.0)
        xmax = ymax = zmax = f32(1.0)
    init_mesh_fn, coord_at = mesh.construct(dim)

    # --------- Grid nodes for training

    xc = jnp.linspace(xmin, xmax, cfg.solver.Nx_tr, dtype=f32)
    yc = jnp.linspace(ymin, ymax, cfg.solver.Ny_tr, dtype=f32)
    zc = jnp.linspace(zmin, zmax, cfg.solver.Nz_tr, dtype=f32)
    gstate_tr = init_mesh_fn(xc, yc, zc)

    # --------- Grid nodes for level set
    Nx_lvl = cfg.gridstates.Nx_lvl
    Ny_lvl = cfg.gridstates.Ny_lvl
    Nz_lvl = cfg.gridstates.Nz_lvl
    xc = jnp.linspace(xmin, xmax, Nx_lvl, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny_lvl, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz_lvl, dtype=f32)
    gstate_lvl = init_mesh_fn(xc, yc, zc)

    # ----------  Evaluation Mesh for Visualization
    Nx_eval = cfg.gridstates.Nx_eval
    Ny_eval = cfg.gridstates.Ny_eval
    Nz_eval = cfg.gridstates.Nz_eval
    exc = jnp.linspace(xmin, xmax, Nx_eval, dtype=f32)
    eyc = jnp.linspace(ymin, ymax, Ny_eval, dtype=f32)
    ezc = jnp.linspace(zmin, zmax, Nz_eval, dtype=f32)
    eval_gstate = init_mesh_fn(exc, eyc, ezc)

    # ----------  Set up current experiment
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
        alpha_fn,
        exact_sol_m_fn,
        exact_sol_p_fn,
        evaluate_exact_solution_fn,
        g_m_fn,
        g_p_fn,
        beta_fn
    ) = exp_fn()

#beta_fn, # do not expect this for robin

    # ----------- Set up optimizer
    optimizer_dict = {
        "optimizer_name": cfg.solver.optim.optimizer_name,
        "learning_rate": cfg.solver.optim.learning_rate,
        "eps": cfg.solver.optim.get("eps", None),
        "weight_decay": cfg.solver.optim.get("weight_decay", None),
        "sched": cfg.solver.sched,
    }
    # # import time
    # # time.sleep(3)
    # optimizer = {
    #     "optimizer_name" : cfg.solver.optim.optimizer_name,
    #     "sched": cfg.solver.sched,
    #     "learning_rate": cfg.solver.optim.learning_rate
    # }


    # --------- Set up first version solver
    if False:
        init_fn, solve_fn = poisson_solver_scalable.setup(
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
        )
        sim_state = init_fn(gstate_tr.R)
        t1 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(
            gstate=gstate_tr,
            eval_gstate=eval_gstate,
            sim_state=sim_state,
            algorithm=0,
            switching_interval=3,
            Nx_tr=cfg.solver.Nx_tr,
            Ny_tr=cfg.solver.Ny_tr,
            Nz_tr=cfg.solver.Nz_tr,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            currDir=results_path,
            print_rate=cfg.solver.print_rate,
        )
        t2 = time.time()
    else:
        # --------- Set up second version solver
        init_fn_Robin = trainer.setup_Robin(
            initial_value_fn,
            dirichlet_bc_fn,
            phi_fn,
            mu_m_fn,
            mu_p_fn,
            k_m_fn,
            k_p_fn,
            f_m_fn,
            f_p_fn,
            g_m_fn,
            g_p_fn,
            alpha_fn,
            beta_fn,
        )
        # init_fn = trainer.setup(
        #     initial_value_fn,
        #     dirichlet_bc_fn,
        #     phi_fn,
        #     mu_m_fn,
        #     mu_p_fn,
        #     k_m_fn,
        #     k_p_fn,
        #     f_m_fn,
        #     f_p_fn,
        #     alpha_fn,
        #     beta_fn,
        # )
        print(f"Optimizer dict: {optimizer_dict}")
        print(f"Function: {init_fn_Robin}")
        model_dict = OmegaConf.to_container(cfg.model, resolve=True) if "model" in cfg else None
        sim_state, solve_fn = init_fn_Robin(
            lvl_gstate=gstate_lvl,
            tr_gstate=gstate_tr,
            eval_gstate=eval_gstate,
            algorithm=algorithm,
            num_epochs=num_epochs,
            multi_gpu=multi_gpu,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_path,
            loss_plot_name=test_name,
            optimizer_dict=optimizer_dict,
            restart=cfg.solver.restart_from_checkpoint,
            restart_checkpoint_dir=cfg.solver.restart_checkpoint_dir,
            print_rate=cfg.solver.print_rate,
            train_omega_minus_only=cfg.solver.get("train_omega_minus_only", False),
            model_dict=model_dict,
        )
        t1 = time.time()
        sim_state, epoch_store, loss_epochs = solve_fn(sim_state=sim_state)
        t2 = time.time()

    logger.info(f"solve took {(t2 - t1)} seconds")
    jax.profiler.save_device_memory_profile(f"{results_path}/memory_poisson_test_{test_name}.prof")

    eval_phi = vmap(phi_fn)(eval_gstate.R)
    exact_sol = vmap(evaluate_exact_solution_fn)(eval_gstate.R)
    error = sim_state.solution - exact_sol
    log = {
        "phi": eval_phi,
        "U": sim_state.solution,
        "U_exact": exact_sol,
        "U-U_exact": error,
    }
    print(f"DIMENSIONS:\neval_phi: {len(eval_phi)}\nU:{len(sim_state.solution)}")
    print(f"log: {log}")

    #value = os.path.join(results_path, test_name)
    #print(f"SAVING INFORMATION INTO : {value}")
    io.write_vtk_manual(
        eval_gstate,
        log,
        filename=os.path.join(results_path, test_name),
    )

    # log = {
    #     'phi': sim_state.phi,
    #     'U': sim_state.solution,
    #     'U_exact': exact_sol,
    #     'U-U_exact': sim_state.solution - exact_sol,
    #     'alpha': sim_state.alpha,
    #     'beta': sim_state.beta,
    #     'mu_m': sim_state.mu_m,
    #     'mu_p': sim_state.mu_p,
    #     'f_m': sim_state.f_m,
    #     'f_p': sim_state.f_p,
    #     'grad_um_x': sim_state.grad_solution[0][:,0],
    #     'grad_um_y': sim_state.grad_solution[0][:,1],
    #     'grad_um_z': sim_state.grad_solution[0][:,2],
    #     'grad_up_x': sim_state.grad_solution[1][:,0],
    #     'grad_up_y': sim_state.grad_solution[1][:,1],
    #     'grad_up_z': sim_state.grad_solution[1][:,2],
    #     'grad_um_n': sim_state.grad_normal_solution[0],
    #     'grad_up_n': sim_state.grad_normal_solution[1]
    # }
    # io.write_vtk_manual(gstate, log)
    
    # TODO: REVIEW ERROR MASKING
    # rms_err = jnp.square(sim_state.solution - exact_sol).mean() ** 0.5
    # L_inf_err = abs(sim_state.solution - exact_sol).max()
    # #OLD L2L2_err = jnp.sqrt(((sim_state.solution - exact_sol) ** 2).sum() / ((Nx_eval-1)*(Ny_eval-1)*(Nz_eval-1))
	
    # dx = (xmax-xmin) / (Nx_eval-1)
    # dy = (ymax-ymin) / (Ny_eval-1)
    # dz = (zmax-zmin) / (Nz_eval-1)

    # L2_err = jnp.sqrt(((sim_state.solution - exact_sol) ** 2).sum() *dx*dy*dz)

    # L2_rel_loss = jnp.sqrt(((sim_state.solution - exact_sol) ** 2).sum() / (exact_sol**2).sum())

    domain_mask = jnp.where(eval_phi <= 0.0, 1.0, 0.0)
    raw_error = sim_state.solution - exact_sol
    masked_error = raw_error * domain_mask

    L_inf_err = jnp.abs(masked_error).max()
    num_interior_points = jnp.sum(domain_mask)
    rms_err = jnp.sqrt(jnp.square(masked_error).sum() / num_interior_points)
    dx = (xmax-xmin) / (Nx_eval-1)
    dy = (ymax-ymin) / (Ny_eval-1)
    dz = (zmax-zmin) / (Nz_eval-1)

    L2_err = jnp.sqrt((masked_error ** 2).sum() * dx * dy * dz)
    masked_exact_sol = exact_sol * domain_mask
    L2_rel_loss = jnp.sqrt((masked_error ** 2).sum() / (masked_exact_sol ** 2).sum())

    # print(num_epochs)
    # logger.info(f"Num_epochs: {num_epochs}")

    # logger.info(
    #     f"Accuracy: \n L_inf : {L_inf_err} \n \n L_2 : {L2_err} \n Rel. L_2 : {L2_rel_loss} \n RMSE Loss: {rms_err}"
    # )
    # logger.info(f"Experiment {test_name} completed! \n")
    print(num_epochs)
    logger.info(f"Num_epochs: {num_epochs}")
    print("Exact Min/Max:", jnp.min(exact_sol), jnp.max(exact_sol))
    print("Prediction Min/Max:", sim_state.solution.min(), sim_state.solution.max())
    test_g = vmap(g_p_fn)(eval_gstate.R)
    print("Does g_p_fn contain NaNs?", jnp.isnan(test_g).any())
    print("-------------------\n")
    logger.info(
        f"Interior (Omega-) Accuracy: \n L_inf : {L_inf_err} \n L_2 : {L2_err} \n Rel. L_2 : {L2_rel_loss} \n RMSE Loss: {rms_err}"
    )
    logger.info(f"Experiment {test_name} completed! \n")

    """
    MASK the solution over sphere only
    """
    """
    logger.info("\n GRADIENT ERROR\n")

    grad_um = sim_state.grad_solution[0].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up = sim_state.grad_solution[1].reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_exact = vmap(grad(exact_sol_m_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]
    grad_up_exact = vmap(grad(exact_sol_p_fn))(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    mask_m = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 #-0.5*dx
    err_x_m = abs(grad_um[mask_m][:,0] - grad_um_exact[mask_m][:,0]).max()
    err_y_m = abs(grad_um[mask_m][:,1] - grad_um_exact[mask_m][:,1]).max()
    err_z_m = abs(grad_um[mask_m][:,2] - grad_um_exact[mask_m][:,2]).max()

    mask_p = sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 #0.5*dx
    err_x_p = abs(grad_up[mask_p][:,0] - grad_up_exact[mask_p][:,0]).max()
    err_y_p = abs(grad_up[mask_p][:,1] - grad_up_exact[mask_p][:,1]).max()
    err_z_p = abs(grad_up[mask_p][:,2] - grad_up_exact[mask_p][:,2]).max()

    logger.info(f"L_inf errors in grad u in Omega_minus x: {err_x_m}, \t y: {err_y_m}, \t z: {err_z_m}")
    logger.info(f"L_inf errors in grad u in Omega_plus  x: {err_x_p}, \t y: {err_y_p}, \t z: {err_z_p}")



    #--- normal gradients over interface
    normal_fn = grad(phi_fn)
    normal_vec = vmap(normal_fn)(gstate.R).reshape((Nx,Ny,Nz,3))[1:-1,1:-1,1:-1]

    grad_um_n = sim_state.grad_normal_solution[0].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]
    grad_up_n = sim_state.grad_normal_solution[1].reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]

    mask_i_m = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] < 0.0 )
    mask_i_p = ( abs(sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1]) < 0.5*dx ) * ( sim_state.phi.reshape((Nx,Ny,Nz))[1:-1,1:-1,1:-1] > 0.0 )

    grad_um_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_um_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))
    grad_up_n_exact = vmap(jnp.dot, (0,0))(normal_vec.reshape(-1,3), grad_up_exact.reshape(-1,3)).reshape((Nx-2,Ny-2,Nz-2))

    err_um_n = abs(grad_um_n - grad_um_n_exact)[mask_i_m].max()
    err_up_n = abs(grad_up_n - grad_up_n_exact)[mask_i_p].max()


    logger.info(f"L_inf error in normal grad u on interface minus: {err_um_n} \t plus: {err_up_n}")

    #----
    assert L_inf_err<0.2

    """


if __name__ == "__main__":
    test_poisson()
