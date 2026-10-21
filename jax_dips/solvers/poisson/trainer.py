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
import logging
import os
import pickle
import signal
import time
from functools import partial
from typing import Callable, Tuple, TypeVar

import numpy as onp

logger = logging.getLogger(__name__)

import jax
import optax
from jax import jit, value_and_grad, grad
from jax import numpy as jnp
from jax import pmap, random, vmap
from optax._src.base import GradientTransformation
import jaxopt
import haiku as hk
import flax
import flax.linen as nn

from jax_dips._jaxmd_modules import dataclasses, util
from jax_dips.data import data_management
from jax_dips.domain.mesh import GridState
from jax_dips.solvers.simulation_states import PoissonSimState, PoissonSimStateFn
from jax_dips.utils.inspect import epoch_timing_summary, print_architecture, progress_bar, reset_epoch_timing
from jax_dips.utils.visualization import plot_loss_epochs
from jax_dips.solvers.optimizers import get_optimizer
from jax_dips.nn.configure import get_model
from jax_dips.solvers.poisson.discretization import Discretization
from jax_dips.nn.preconditioner import Preconditioner

Array = util.Array
i32 = util.i32
f32 = util.f32
f64 = util.f64

T = TypeVar("T")
SolveFn = Callable[[PoissonSimState], PoissonSimState]
InitFn = Callable[
    [GridState, GridState, int, int, int, int, int, int, int, bool, int, str, str, str],
    Tuple[PoissonSimState, SolveFn],
]

InitFn_Robin = Callable[
    [GridState, GridState, int, int, int, int, int, int, int, bool, int, str, str, str],
    Tuple[PoissonSimState, SolveFn],
]



Simulator = Tuple[InitFn, SolveFn]
Simulator_Robin = Tuple[InitFn_Robin, SolveFn]

stop_training = False


def signalHandler(signal_num, frame):
    global stop_training
    stop_training = True
    logger.warning("Signal:", signal_num, " Frame: ", frame)
    logger.warning("Training will stop after the completion of current epoch")


signal.signal(signal.SIGINT, signalHandler)


class Trainer(Discretization):
    """Trainer class for Poisson solvers. Performs all operations for multi/single GPU training.

    Uses loss residuals to train the neural network model. This class relies on the NBM oracle (nbm.py) to
    interact with the neural network surrogate model for evaulating the loss, solution, solution gradients.
    """

    def __init__(
        self,
        lvl_gstate: GridState,
        tr_gstate: GridState,
        eval_gstate: GridState,
        sim_state: PoissonSimState,
        sim_state_fn: PoissonSimStateFn,
        algorithm: int = 0,
        mgrad_over_pgrad_scalefactor: int = 1,
        lvl_set_fn: Callable[..., Array] = None,
        num_epochs: int = 1000,
        multi_gpu: bool = False,
        batch_size: int = 131072,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 2,
        results_dir: str = "./",
        loss_plot_name: str = "solver_loss",
        optimizer_dict: dict = {
            "optimizer_name": "custom",
            "learning_rate": 1e-3,
            "sched": {"scheduler_name": "exponential", "decay_rate": 0.9},
        },
        restart: bool = False,
        restart_checkpoint_dir: str = "./checkpoints",
        print_rate: int = 1,
        train_omega_minus_only: bool = False,
        robin_weight_length=1.0,
        alternate_tets: bool = True,
        jump_scheme: str = "original",
        jump_ell: float = 1.0e4,
        jump_cut_floor: float = 0.05,
        jump_eval_once: bool = False,
        last_layer_ls: bool = False,
        last_layer_ls_rcond: float = 1.0e-6,
        model_dict: dict = {
            "name": None,
            "model_type": "mlp",
            "mlp": {
                "hidden_layers_m": 1,
                "hidden_dim_m": 1,
                "activation_m": "jnp.tanh",
                "hidden_layers_p": 2,
                "hidden_dim_p": 10,
                "activation_p": "jnp.tanh",
            },
            "resnet": {
                "res_blocks_m": 3,
                "res_dim_m": 40,
                "activation_m": "nn.tanh",
                "res_blocks_p": 3,
                "res_dim_p": 80,
                "activation_p": "nn.tanh",
            },
        },
    ) -> None:
        self.alternate_tets = alternate_tets  # read while the parent builds the cell geometry
        self.jump_scheme = jump_scheme  # jump problems; see Discretization.jump_scheme
        self.jump_ell = float(jump_ell)
        self.jump_cut_floor = float(jump_cut_floor)
        self.jump_eval_once = bool(jump_eval_once)
        super().__init__(
            lvl_gstate,
            sim_state,
            sim_state_fn,
            precondition=1,
            algorithm=algorithm,
        )
        logger.info(f"Alternating tetrahedral split: {alternate_tets}")
        if not self.robin:
            logger.info(
                f"Jump scheme: {self.jump_scheme}"
                + (f" (row weight sqrt(h^3 + {self.jump_ell:g} * (max(A_Gamma, {self.jump_cut_floor:g} h^2) + A_ghost)))"
                   if self.jump_scheme == "fixed" else " (rows divided by their diagonal)")
                + f"; network evaluated once per node: {self.jump_eval_once}"
            )
        self.robin_weight_length = robin_weight_length
        logger.info(f"Robin row weight: sqrt(V + l * A) with l = {robin_weight_length}")
        self.last_layer_ls = bool(last_layer_ls)
        self.last_layer_ls_rcond = float(last_layer_ls_rcond)
        if self.last_layer_ls:
            if multi_gpu:
                raise NotImplementedError("last_layer_ls is single-GPU only.")
            logger.info(f"Hybrid training: after the optimizer, the output layers are solved by least squares "
                        f"(singular values kept above {self.last_layer_ls_rcond:g} x the largest)")
        #########################################################################
        global stop_training
        self.key = random.PRNGKey(758493)
        self.eval_gstate = eval_gstate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.multi_gpu = multi_gpu
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.results_dir = results_dir
        self.loss_plot_name = loss_plot_name
        self.print_rate = print_rate
        self.mgrad_over_pgrad_scalefactor = mgrad_over_pgrad_scalefactor
        self.restart_checkpoint_dir = restart_checkpoint_dir
        #########################################################################
        self.TD = data_management.TrainData(
            tr_gstate,
            lvl_set_fn,
            refine=False,
            refine_lod=False,
            refine_normals=False,
            v_cycle_period=2,
            rest_at_level=100,
        )
        self.train_points = self.TD.train_points
        self.train_dx = self.TD.gstate.dx
        self.train_dy = self.TD.gstate.dy
        self.train_dz = self.TD.gstate.dz

        if train_omega_minus_only:
            # For a Robin problem the PDE is solved only in Omega-. Keep every cell
            # that overlaps Omega- (V_m above min_vol_frac), including cut cells centred outside;
            # cells entirely in Omega+ only touch the Omega+ network.
            _V_m = vmap(self.compute_face_centroids_values_plus_minus_at_point, (0, None, None, None))(
                self.train_points, self.train_dx, self.train_dy, self.train_dz
            )[:, 12]
            _keep = _V_m > self.min_vol_frac * self.train_dx * self.train_dy * self.train_dz
            _n_before = int(self.train_points.shape[0])
            self.train_points = self.train_points[_keep]
            _n_after = int(self.train_points.shape[0])
            if _n_after == 0:
                raise ValueError(
                    "train_omega_minus_only=True left 0 training points: no cell overlaps Omega-."
                )
            logger.info(
                f"train_omega_minus_only: keeping {_n_after} of {_n_before} training points "
                f"({100.0 * _n_after / _n_before:.1f}%); Omega+ is no longer trained, so the "
                f"solution there (and in any Omega+ visualization) is meaningless."
            )
        if self.jump_eval_once and not self.robin:
            # The rows read the network's values at the training grid's nodes (Discretization.
            # compute_Ax_and_b_from_values_fn), so the stencil must stay on that grid.
            if self.TD.zoom_levels != 1:
                raise ValueError("jump_eval_once needs the stencil at the grid spacing (zoom_levels == 1).")
            _gs = self.TD.gstate
            self._eval_once_R = _gs.R
            self._eval_once_shape = (len(_gs.x), len(_gs.y), len(_gs.z))
            self.eval_once_origin = jnp.array([_gs.xmin(), _gs.ymin(), _gs.zmin()], dtype=f32)
        # phis_at_points = vmap(lvl_set_fn)(self.train_points)
        # train_points_m, train_points_p = self.TD.split_train_points_by_region(phis_at_points, self.train_points)
        #########################################################################
        if model_dict["model_type"] == "discrete":
            model_dict["discrete"]["xmin"] = float(self.TD.gstate.xmin())
            model_dict["discrete"]["xmax"] = float(self.TD.gstate.xmax())
            model_dict["discrete"]["ymin"] = float(self.TD.gstate.ymin())
            model_dict["discrete"]["ymax"] = float(self.TD.gstate.ymax())
            model_dict["discrete"]["zmin"] = float(self.TD.gstate.zmin())
            model_dict["discrete"]["zmax"] = float(self.TD.gstate.zmax())

        self.forward, self.framework = get_model(model_dict, model_type=model_dict["model_type"])

        def load_params_haiku(params):
            return partial(self.forward.apply, params, None)

        def load_params_flax(params):
            return partial(self.forward.apply, params)

        if self.framework == "haiku":
            self.load_params_fn = load_params_haiku
        elif self.framework == "flax":
            self.load_params_fn = load_params_flax
        else:
            raise NotImplementedError

        ###################################################################

        if optimizer_dict["optimizer_name"] != "lbfgs":
            # The LR schedule is stepped once per optimizer update, i.e. once per batch
            # per epoch -- not once per epoch. Derive the true step count so that a
            # schedule configured with transition_steps=null anneals over the whole run.
            _batch_size = min(self.batch_size, len(self.train_points))
            _num_batches = max(1, -(-len(self.train_points) // _batch_size))
            total_steps = max(1, self.num_epochs * _num_batches)

            _sched_name = optimizer_dict["sched"]["scheduler_name"]
            _cfg_transition_steps = optimizer_dict["sched"].get("transition_steps", None)
            if _cfg_transition_steps:
                transition_steps = _cfg_transition_steps
            elif _sched_name in ("cosine", "polynomial"):
                # These take the full horizon: the LR reaches 0 after transition_steps.
                transition_steps = total_steps
            else:
                # "exponential" treats transition_steps as the decay *period*
                # (lr * decay_rate ** (step / transition_steps)), so the full horizon
                # would mean almost no decay. Keep the historical default here so that
                # existing experiments are unaffected; set it explicitly to anneal.
                transition_steps = 1000
            # Adam's eps sets a resolution-dependent error floor here (see the note in
            # jax_dips/solvers/optimizers.py:get_optimizer). It must stay well below the
            # smallest gradient norm the run will produce, which shrinks like dx^3.
            _eps = optimizer_dict.get("eps", None)
            # float() guards against YAML handing this back as the string "1e-16":
            # optax would take it without complaint and fail deep inside the update.
            _eps = 1e-16 if _eps is None else float(_eps)
            # adamw only. Its decoupled decay sets a resolution-INDEPENDENT error floor
            # (see the note in jax_dips/solvers/optimizers.py:get_optimizer).
            _wd = optimizer_dict.get("weight_decay", None)
            _wd = 1e-4 if _wd is None else float(_wd)
            logger.info(
                f"LR schedule '{optimizer_dict['sched']['scheduler_name']}': "
                f"{total_steps} total steps ({self.num_epochs} epochs x {_num_batches} batches), "
                f"transition_steps={transition_steps}, adam_eps={_eps}, weight_decay={_wd}"
            )
            self.optimizer = get_optimizer(
                optimizer_name=optimizer_dict["optimizer_name"],
                scheduler_name=optimizer_dict["sched"]["scheduler_name"],
                learning_rate=optimizer_dict["learning_rate"],
                decay_rate=optimizer_dict["sched"]["decay_rate"],
                transition_steps=transition_steps,
                eps=_eps,
                weight_decay=_wd,
                loss_fn=self.loss,
            )
            self.solve = self.solve_optax
        else:
            # `tol` is jaxopt's threshold on the gradient infinity-norm. It was 1e-3,
            # but this loss produces gradients of order 1e-8 (the preconditioned residual's
            # sensitivity to the solution error carries dx^3 -- see the note in
            # jax_dips/solvers/optimizers.py:get_optimizer). L-BFGS therefore passed its
            # convergence test at iteration 0 and returned without taking a single step,
            # which is why switching to it appeared to destroy the resolution ordering.
            # Keep the tolerance far below the smallest gradient norm the run can produce.
            _lbfgs_tol = float(optimizer_dict.get("lbfgs_tol", None) or 1e-14)
            _lbfgs_maxiter = int(optimizer_dict.get("lbfgs_maxiter", None) or self.num_epochs)
            logger.info(f"Using jaxopt.LBFGS (maxiter={_lbfgs_maxiter}, tol={_lbfgs_tol})")
            self.optimizer = jaxopt.LBFGS(
                fun=self.loss,
                value_and_grad=True,
                maxiter=_lbfgs_maxiter,
                tol=_lbfgs_tol,
            )
            self.solve = self.solve_jaxopt

        # --- Model + preconditioner are built on BOTH paths. Previously this whole
        # block lived in the `else` of `if restart:`, so a restarted run never set
        # self.precond and died with AttributeError inside precond_fn.
        rng = random.PRNGKey(42)
        self.params = self.forward.init(rng, x=jnp.array([0.0, 0.0, 0.0]), phi_x=f32(0.1))
        try:
            logger.info(self.forward.tabulate(rng, x=jnp.array([0.0, 0.0, 0.0]), phi_x=f32(0.1)))
        except (AttributeError, NotImplementedError):
            print_architecture(self.params)

        if "preconditioner" in model_dict and model_dict["preconditioner"].get("enable", False):
            self.precond = Preconditioner(
                Ds=model_dict["preconditioner"]["layer_widths"],
                out_dim=1,
                scaling_coeff=model_dict["preconditioner"]["scaling_coeff"],
            )
            self.precond_params = self.precond.init(
                rng, jnp.array([0.0] * 26)
            )  # 26 is number of coeffs_ in discretization
            logger.info(self.precond.tabulate(rng, jnp.array([0.0] * 26)))
            self.params = flax.core.unfreeze(self.params)
            precond_params = flax.core.unfreeze(self.precond_params)
            self.params["preconditioner"] = precond_params
            self.params = flax.core.freeze(self.params)
            precond_params = flax.core.freeze(precond_params)
        else:

            class dummy:
                def apply(self, params, coeff):
                    return 1.0

            self.precond = dummy()
            self.params = flax.core.unfreeze(self.params)
            self.params["preconditioner"] = {}
            self.params = flax.core.freeze(self.params)

        if restart:
            state = self.fetch_checkpoint(self.restart_checkpoint_dir)
            if state is None:
                raise FileNotFoundError(f"restart=True but no checkpoint found in {self.restart_checkpoint_dir}")
            # Warm start takes the WEIGHTS ONLY. Optimizer state is deliberately
            # NOT restored: it carries the LR-schedule step count, so a cosine
            # schedule would resume fully annealed and the new run would train at
            # ~zero learning rate. batch_size likewise stays at the configured
            # value rather than the checkpoint's, which may come from a different
            # resolution and would silently switch on mini-batching.
            # Weights take the run's precision (util.f32 is float64 under use_float64), so a float32
            # checkpoint continued in float64 is optimised in float64; a no-op for a float32 run.
            self.params = jax.tree_util.tree_map(
                lambda a: jnp.asarray(a, dtype=util.f32) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a,
                state["params"],
            )
            logger.info(
                f"Warm-starting weights from {self.restart_checkpoint_dir} "
                f"(checkpoint epoch {state.get('epoch')}, resolution {state.get('resolution')}). "
                f"Optimizer state reinitialized; batch_size={self.batch_size} from config."
            )

        if optimizer_dict["optimizer_name"] != "lbfgs":
            self.opt_state = self.init_optax()

        #########################################################################
        # self.mnet_keys, self.pnet_keys = self.split_mp_networks_keys(self.params)  # split keys for each domain

        self.epoch_start = 0
        self.loss_epochs = jnp.zeros(self.num_epochs - self.epoch_start)
        self.epoch_store = onp.arange(self.epoch_start, self.num_epochs)
        #########################################################################
        """ initialize postprocessing methods """
        self.compute_normal_gradient_solution_mp_on_interface = (
            self.compute_normal_gradient_solution_mp_on_interface_neural_network
        )
        self.compute_gradient_solution_mp = self.compute_gradient_solution_mp_neural_network
        self.compute_normal_gradient_solution_on_interface = (
            self.compute_normal_gradient_solution_on_interface_neural_network
        )
        self.compute_gradient_solution = self.compute_gradient_solution_neural_network
        #########################################################################

    @partial(jit, static_argnums=0)
    def init_optax(self):
        r"""This function initializes the neural network with random parameters."""
        opt_state = self.optimizer.init(params=self.params)
        self.mnet_keys, self.pnet_keys = self.split_mp_networks_keys(self.params)  # split keys for each domain
        return opt_state

    # @staticmethod
    # @hk.transform
    # def forward(x, phi_x):
    #     r"""Forward pass of the neural network.

    #     Args:
    #         x: input data

    #     Returns:
    #         output of the neural network
    #     """
    #     model = DoubleMLP()
    #     return model(x, phi_x)

    @staticmethod
    def split_mp_networks_keys(params):
        """Split keys for negative and positive networks
        Expects the layer names include patterns _m_fn and _p_fn for -/+ networks respectively

        Args:
            params [dict]: parameters dictionary of the model with both positive/negative networks

        Returns:
            (mnet_keys, list[str]), (pnet_keys list[str]): pair of lists for key strings of -/+ networks
        """
        import re

        def find_keys_with_pattern(keys, pattern):
            matched_keys = []
            for key in keys:
                if re.search(pattern, key):
                    matched_keys.append(key)
            return matched_keys

        keys = params.keys()
        m_pattern = r"_m_fn"
        mnet_keys = find_keys_with_pattern(keys, m_pattern)
        p_pattern = r"_p_fn"
        pnet_keys = find_keys_with_pattern(keys, p_pattern)
        return mnet_keys, pnet_keys
        # mnet, pnet = hk.data_structures.partition(lambda m, n, p: m != "double_mlp/~resnet_m_fn/linear", params)

    #########################################################################
    @staticmethod
    def fetch_checkpoint(checkpoint_dir):
        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            return None
        else:
            checkpoints = [p for p in os.listdir(checkpoint_dir) if "checkpoint_" in p]
            if checkpoints == []:
                return None
            # Sort by epoch NUMBER. A plain max() is lexicographic, which ranks
            # "checkpoint_9750" above "checkpoint_20000" and silently loads a
            # mid-training checkpoint.
            def _epoch_of(name):
                try:
                    return int(name.rsplit("_", 1)[-1])
                except ValueError:
                    return -1

            checkpoint = os.path.join(checkpoint_dir, max(checkpoints, key=_epoch_of))
            logger.info(f"Loading checkpoint {checkpoint}")
            with open(checkpoint, "rb") as f:
                state = pickle.load(f)
            return state

    @staticmethod
    def save_checkpoint(checkpoint_dir, state):
        if checkpoint_dir is None:
            logger.info("No checkpoint dir. specified. Skipping checkpoint.")
            return
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint_" + str(state["epoch"]))
        logger.info(f"Saving checkpoint {checkpoint}")
        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)
        return checkpoint

    # -------------
    def solve_jaxopt(self):
        """jaxopt based optimizers
        TODO: right now assumes only 1 batch
        """
        self.DD = data_management.DatasetDict(
            batch_size=self.batch_size,
            x_data=self.train_points,
        )
        batched_training_data = self.DD.get_batched_data()
        num_batches = batched_training_data.shape[0]

        # 1.
        solver = jaxopt.ScipyMinimize(
            method="l-bfgs-b",
            fun=self.loss,
            tol=1e-15,
            maxiter=self.num_epochs,
            implicit_diff_solve=True,
        )
        # solver = jaxopt.GradientDescent(fun=self.loss, maxiter=self.num_epochs, implicit_diff=True)
        # 2. Scipy Least Squares
        # solver = jaxopt.ScipyLeastSquares(
        #     method="trf",
        #     fun=self.loss,
        # )  # trf, dogbox, lm

        start_time = time.time()
        solver_sol = solver.run(
            self.params,
            points=batched_training_data[0],
            dx=self.train_dx,
            dy=self.train_dy,
            dz=self.train_dz,
        )
        end_time = time.time()
        logger.info(f"solve took {end_time - start_time} (sec)")
        # L-BFGS-B can stop long before maxiter (line-search failure on noisy
        # float32 gradients, or scipy's maxfun=15000 default). Without this you
        # cannot tell a converged run from one that bailed on iteration 40.
        _st = solver_sol.state
        logger.info(
            f"L-BFGS termination: success={getattr(_st, 'success', '?')} "
            f"status={getattr(_st, 'status', '?')} "
            f"iters={getattr(_st, 'iter_num', '?')} (maxiter={self.num_epochs}) "
            f"final_loss={getattr(_st, 'fun_val', '?')}"
        )
        if len(batched_training_data) > 1:
            logger.warning(
                f"L-BFGS is fitting ONLY batch 0 of {len(batched_training_data)}: "
                f"{batched_training_data[0].shape[0]} of {len(self.train_points)} training points. "
                f"Set batch_size >= Nx_tr**3 to use the full domain."
            )

        # 3
        # solver = jaxopt.ScipyRootFinding(
        #     method="krylov",
        #     optimality_fun=self.residual_vector,
        #     tol=1e-15,
        # )
        # u_res = jnp.zeros(self.eval_gstate.R.shape[0])
        # start_time = time.time()
        # solver_sol = solver.run(
        #     init_params=u_res,
        # )
        # end_time = time.time()
        # logger.info(f"solve took {end_time - start_time} (sec)")

        self.params = solver_sol.params
        self.opt_state = solver_sol.state
        if self.last_layer_ls:
            self.params = self.solve_last_layer(self.params)
        state = {
            "opt_state": self.opt_state,
            "params": self.params,
            "epoch": self.epoch_store[-1] + 1,
            "batch_size": self.batch_size,
            "resolution": f"{self.train_dx}, {self.train_dy}, {self.train_dz}",
        }
        self.save_checkpoint(self.checkpoint_dir, state)

        (
            final_solution,
            grad_u,
            grad_u_normal_to_interface,
        ) = self.evaluate_solution_and_gradients(self.params, self.eval_gstate)
        return (
            final_solution,
            grad_u,
            grad_u_normal_to_interface,
            self.epoch_store,
            self.loss_epochs,
        )
        # opt_state = self.optimizer.init_state(
        #     init_params=self.params,
        #     points=jnp.array([[0.0, 0.0, 0.0]]),
        #     dx=self.train_dx,
        #     dy=self.train_dy,
        #     dz=self.train_dz,
        # )

        # self.optimizer.update()

    # -----------------------
    def solve_optax(self):
        reset_epoch_timing()
        start_time = time.time()
        if self.multi_gpu:
            n_devices = jax.local_device_count()
            self.DD = data_management.DatasetDict(
                batch_size=self.batch_size,
                x_data=self.train_points,
                num_gpus=n_devices,
            )
            (
                self.opt_state,
                self.params,
                self.epoch_store,
                self.loss_epochs,
            ) = self.multi_GPU_train(self.opt_state, self.params)
        else:
            self.DD = data_management.DatasetDict(
                batch_size=self.batch_size,
                x_data=self.train_points,
            )
            batched_training_data = self.DD.get_batched_data()
            _n_batches = batched_training_data.shape[0]
            if _n_batches > 1:
                # DatasetDict slices batches as CONTIGUOUS ranges of gstate.R, which
                # is a lexicographically flattened meshgrid -- so each batch is a
                # spatial slab, not a random sample. The cross-batch shuffle in
                # learn_one_epoch is commented out, so the optimizer sweeps the same
                # slabs in the same order every epoch. Set batch_size >= the number
                # of training points to train full-batch.
                logger.warning(
                    f"batch_size={self.batch_size} < {len(self.train_points)} training points: "
                    f"training on {_n_batches} CONTIGUOUS SPATIAL SLABS per epoch, unshuffled. "
                    f"This is almost certainly not what you want for a resolution study."
                )
            else:
                logger.info(f"Full-batch training: {len(self.train_points)} points, 1 update/epoch.")
            (
                self.opt_state,
                self.params,
                self.epoch_store,
                self.loss_epochs,
            ) = self.single_GPU_train(self.opt_state, self.params, batched_training_data)
            # JAX returns before the GPU finishes; wait, so "solve took" covers every epoch.
            self.loss_epochs.block_until_ready()

        end_time = time.time()
        logger.info(f"solve took {end_time - start_time} (sec)")
        if epoch_timing_summary():
            logger.info(epoch_timing_summary())
        if self.last_layer_ls:
            self.params = self.solve_last_layer(self.params)

        state = {
            "opt_state": self.opt_state,
            "params": self.params,
            "epoch": self.epoch_store[-1] + 1,
            "batch_size": self.batch_size,
            "resolution": f"{self.train_dx}, {self.train_dy}, {self.train_dz}",
        }
        self.save_checkpoint(self.checkpoint_dir, state)

        plot_loss_epochs(
            self.epoch_store,
            self.loss_epochs,
            self.results_dir,
            self.TD.base_level,
            self.TD.alt_res,
            self.loss_plot_name,
        )
        (
            final_solution,
            grad_u,
            grad_u_normal_to_interface,
        ) = self.evaluate_solution_and_gradients(self.params, self.eval_gstate)
        return (
            final_solution,
            grad_u,
            grad_u_normal_to_interface,
            self.epoch_store,
            self.loss_epochs,
        )

    # -----------------------
    @partial(jit, static_argnums=(0))
    def single_GPU_train(self, opt_state, params, batched_training_data):
        # batched_training_data = self.DD.get_batched_data()
        update_fn = self.update
        num_batches = batched_training_data.shape[0]
        # cycle_level = 0

        def learn_one_batch(carry, data_batch):
            # opt_state, params, loss_epoch, train_dx, train_dy, train_dz, cycle_level = carry
            # opt_state, params, loss_epoch_ = update_fn(
            #     opt_state, params, data_batch, train_dx, train_dy, train_dz, cycle_level
            # )
            opt_state, params, loss_epoch, train_dx, train_dy, train_dz = carry
            opt_state, params, loss_epoch_ = update_fn(opt_state, params, data_batch, train_dx, train_dy, train_dz)
            loss_epoch += loss_epoch_
            return (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), None

        def learn_one_epoch(carry, epoch):
            (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ) = carry
            # train_dx, train_dy, train_dz, cycle_level = self.TD.v_cycle_alternate_res(
            #     epoch,
            #     train_dx,
            #     train_dy,
            #     train_dz,
            #     cycle_level,
            # )
            train_dx, train_dy, train_dz = self.TD.alternate_res_sequentially(
                self.num_epochs, epoch, train_dx, train_dy, train_dz
            )
            # batched_training_data = random.permutation(key, batched_training_data, axis=1)
            loss_epoch = 0.0
            (
                opt_state,
                params,
                loss_epoch,
                train_dx,
                train_dy,
                train_dz,
            ), _ = jax.lax.scan(
                learn_one_batch,
                (opt_state, params, loss_epoch, train_dx, train_dy, train_dz),
                batched_training_data,
            )
            loss_epoch = loss_epoch / num_batches
            loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
            loss_epoch = progress_bar((epoch, self.num_epochs, self.print_rate, loss_epoch), loss_epoch)
            return (
                key,
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ), None

        (
            self.key,
            opt_state,
            params,
            self.loss_epochs,
            self.train_dx,
            self.train_dy,
            self.train_dz,
            batched_training_data,
        ), _ = jax.lax.scan(
            learn_one_epoch,
            (
                self.key,
                opt_state,
                params,
                self.loss_epochs,
                self.train_dx,
                self.train_dy,
                self.train_dz,
                batched_training_data,
            ),
            self.epoch_store,
        )
        # self.loss_epochs /= num_batches
        return opt_state, params, self.epoch_store, self.loss_epochs

    # ----------------------------------------------------------------

    def _multi_GPU_train(self, opt_state, params):
        """TODO: This is under development.

        Tensor Model Parallel Training using Positional Sharding

        Args:
            opt_state (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """

        from jax.experimental import mesh_utils
        from jax.experimental.maps import xmap
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P
        from jax.sharding import PositionalSharding

        n_devices = jax.local_device_count()
        devices = mesh_utils.create_device_mesh((n_devices,))
        mesh = Mesh(devices, axis_names=("devices",))

        sharding = PositionalSharding(devices)

        batched_training_data = self.DD.get_batched_data()
        batched_training_data = jax.device_put(batched_training_data, sharding.reshape(n_devices, 1, 1, 1))
        num_batches = batched_training_data.shape[1]

        # -----
        update_fn = self.update_multi_gpu

        def learn_one_batch(carry, data_batch):
            opt_state, params, loss_epoch, train_dx, train_dy, train_dz = carry
            opt_state, params, loss_epoch_ = update_fn(opt_state, params, data_batch, train_dx, train_dy, train_dz)
            loss_epoch += loss_epoch_
            return (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), None

        def learn_one_epoch(
            epoch,
            opt_state,
            params,
            loss_epochs,
            train_dx,
            train_dy,
            train_dz,
            batched_training_data,
        ):
            # train_dx, train_dy, train_dz = self.TD.alternate_res(epoch, train_dx, train_dy, train_dz) #TODO: automate this
            train_dx, train_dy, train_dz = self.TD.alternate_res_sequentially(
                self.num_epochs, epoch, train_dx, train_dy, train_dz
            )
            # batched_training_data = random.permutation(key, batched_training_data, axis=1)
            # TODO: circular shift to the left: lhs = jax.lax.ppermute(lhs, axis_name='i', perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
            loss_epoch = 0.0
            for batch in range(num_batches):
                (
                    opt_state,
                    params,
                    loss_epoch,
                    train_dx,
                    train_dy,
                    train_dz,
                ), _ = learn_one_batch(
                    (opt_state, params, loss_epoch, train_dx, train_dy, train_dz), batched_training_data[batch]
                )
            loss_epoch = loss_epoch / num_batches
            loss_epochs = loss_epochs.at[epoch].set(loss_epoch)
            return (
                opt_state,
                params,
                loss_epochs,
                train_dx,
                train_dy,
                train_dz,
                batched_training_data,
            ), None

        def per_device_update(opt_state, params, per_device_data, train_dx, train_dy, train_dz):
            batched_training_data = per_device_data.squeeze(0)
            for epoch in self.epoch_store:
                (
                    opt_state,
                    params,
                    self.loss_epochs,
                    self.train_dx,
                    self.train_dy,
                    self.train_dz,
                    batched_training_data,
                ), _ = learn_one_epoch(
                    epoch,
                    opt_state,
                    params,
                    self.loss_epochs,
                    self.train_dx,
                    self.train_dy,
                    self.train_dz,
                    batched_training_data,
                )
                loss_epoch = self.loss_epochs[epoch]
                # loss_epoch = progress_bar((epoch, self.num_epochs, self.print_rate, loss_epoch), loss_epoch)
            return opt_state, params, self.epoch_store, self.loss_epochs

        mgpu_train = jit(
            shard_map(
                per_device_update,
                mesh=mesh,
                in_specs=(P(), P(), P("devices", None, None, None), P(), P(), P()),
                out_specs=(P(), P(), P(), P()),
                check_rep=False,
            )
        )
        opt_state, params, self.epoch_store, self.loss_epochs = mgpu_train(
            opt_state, params, batched_training_data, self.train_dx, self.train_dy, self.train_dz
        )
        return opt_state, params, self.epoch_store, self.loss_epochs

    # -----------------------

    def multi_GPU_train(self, opt_state, params):
        """Data Parallel Training using pmap

        Args:
            opt_state (_type_): state of optimizer
            params (_type_): parameters of model

        Returns:
            _type_: optimized state, epochs, losses
        """
        loss_epochs = []
        epoch_store = []
        n_devices = jax.local_device_count()
        self.train_dx = jax.tree_map(lambda x: jnp.array([x] * n_devices), self.train_dx)
        self.train_dy = jax.tree_map(lambda x: jnp.array([x] * n_devices), self.train_dy)
        self.train_dz = jax.tree_map(lambda x: jnp.array([x] * n_devices), self.train_dz)
        params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
        opt_state = jax.tree_map(lambda x: jnp.array([x] * n_devices), opt_state)
        DD = data_management.DatasetDict(
            batch_size=n_devices * self.batch_size,
            x_data=self.train_points,
            num_gpus=n_devices,
        )
        batched_training_data = DD.get_batched_data()
        update_fn = pmap(
            self.update_multi_gpu,
            in_axes=(0, 0, 0, 0, 0, 0),
            axis_name="devices",
        )
        num_batches = batched_training_data.shape[1]
        t0 = time.time()
        for epoch in range(self.epoch_start, self.num_epochs):
            if stop_training:
                break
            loss_epoch = jax.tree_map(lambda x: jnp.array([x] * n_devices), 0.0)
            batched_training_data = random.permutation(self.key, batched_training_data, axis=2)
            for i in range(num_batches):
                opt_state, params, loss_epoch_ = update_fn(
                    opt_state,
                    params,
                    batched_training_data[:, i, ...],
                    self.train_dx,
                    self.train_dy,
                    self.train_dz,
                )
                loss_epoch += loss_epoch_
            if epoch % self.print_rate == 0:
                dt_avg = (time.time() - t0) / self.print_rate
                t0 = time.time()
                logger.info(
                    f"Epoch # {epoch} loss is {jnp.mean(loss_epoch) / num_batches} \t avg epoch time is {dt_avg} (sec)"
                )
            loss_epochs.append(loss_epoch / num_batches)
            epoch_store.append(epoch)
            if (epoch + 1) % self.checkpoint_interval == 0:
                state = {
                    "opt_state": opt_state,
                    "params": params,
                    "epoch": epoch + 1,
                    "batch_size": self.batch_size,
                    "resolution": f"{self.train_dx}, {self.train_dy}, {self.train_dz}",
                }
                self.save_checkpoint(self.checkpoint_dir, state)
        params = jax.device_get(jax.tree_map(lambda x: x[0], params))
        return opt_state, params, epoch_store, loss_epochs

    # -----------------------

    @partial(jit, static_argnums=(0))
    def update(self, opt_state, params, points, dx, dy, dz):
        r"""One step of single-GPU optimization on the neural network model."""
        loss, grads = value_and_grad(self.loss)(params, points, dx, dy, dz)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def __update(self, opt_state, params, points, dx, dy, dz):
        r"""One step of single-GPU optimization on the neural network model.
        Update function must perform update rule on -/+ domains and networks independently
        """

        def _loss(params_trainable, params_untrainable, points, dx, dy, dz):
            params = hk.data_structures.merge(params_trainable, params_untrainable)
            loss = self.loss(params, points, dx, dy, dz)
            return loss

        # split parameters
        m_params, p_params = hk.data_structures.partition(lambda m, n, p: m not in self.pnet_keys, params)
        # -- first get gradients of negative network
        m_loss, m_grads = value_and_grad(_loss)(m_params, p_params, points, dx, dy, dz)
        # -- second get gradients of positive network
        p_loss, p_grads = value_and_grad(_loss)(p_params, m_params, points, dx, dy, dz)
        # --- Choose region to optimize by zeroing the other region
        m_grads = jax.tree_map(lambda x: x * self.mgrad_over_pgrad_scalefactor, m_grads)
        # -- append the grads into m_grads dictionary
        m_grads.update(p_grads)
        updates, opt_state = self.optimizer.update(m_grads, opt_state, params)  # -- get optimizer updates
        params = optax.apply_updates(params, updates)  # -- apply updates on params
        # -- collect losses
        loss = m_loss + p_loss
        return opt_state, params, loss

    # def update_lbfgs(self, params, points, dx, dy, dz, maxiter=10):
    #     solver = jaxopt.LBFGS(fun=self.loss, maxiter=maxiter)
    #     params, opt_state = solver.run(params, points=points, dx=dx, dy=dy, dz=dz)
    #     return opt_state, params

    # @partial(jit, static_argnums=(0))
    def update_multi_gpu(self, opt_state, params, points, dx, dy, dz):
        r"""One step of multi-GPU optimization on the neural network model."""
        loss, grads = value_and_grad(self.loss)(params, points, dx, dy, dz)

        """ Muli-GPU """
        grads = jax.lax.psum(grads, axis_name="devices")
        loss = jax.lax.psum(loss, axis_name="devices")

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    @partial(jit, static_argnums=(0))
    def evaluate_solution_fn(self, params, R_flat):
        phi_flat = self.phi_interp_fn(R_flat)
        # sol_fn = partial(self.forward.apply, params, None) TODO: HAIKU
        # sol_fn = partial(self.forward.apply, params) TODO: FLAX
        sol_fn = self.load_params_fn(params=params)
        pred_sol = vmap(sol_fn, (0, 0))(R_flat, phi_flat)

        return pred_sol

    def precond_fn(self, params, lhs_rhs):
        return self.precond.apply(params["preconditioner"], lhs_rhs)

    def solution_at_point_fn(self, params, r_point, phi_point):
        # TODO: below line is for HAIKU only, commented to try FLAX
        # sol_fn = partial(self.forward.apply, params, None)  TODO: HAIKU
        # sol_fn = partial(self.forward.apply, params)  TODO: FLAX
        sol_fn = self.load_params_fn(params=params)
        return sol_fn(r_point, phi_point).reshape()

    def get_sol_grad_sol_fn(self, params):
        u_at_point_fn = partial(self.solution_at_point_fn, params)
        grad_u_at_point_fn = grad(u_at_point_fn)
        return u_at_point_fn, grad_u_at_point_fn

    def get_mask_plus(self, points):
        """
        For a set of points, returns 1 if in external region
        returns 0 if inside the geometry.
        """
        phi_points = self.phi_interp_fn(points)

        def sign_p_fn(a):
            # returns 1 only if a>0, otherwise is 0
            sgn = jnp.sign(a)
            return jnp.floor(0.5 * sgn + 0.75)

        mask_p = sign_p_fn(phi_points)
        return mask_p

    def rows_lhs_rhs(self, params, points, dx, dy, dz):
        """[lhs, rhs] of the rows at `points`. For a jump problem with jump_eval_once, the network is
        evaluated once at every node of the training grid and each row reads those values;
        otherwise each row evaluates the network itself."""
        if self.jump_eval_once and not self.robin:
            U = self.evaluate_solution_fn(params, self._eval_once_R).reshape(self._eval_once_shape)
            U_pad = jnp.pad(U, 2)
            return vmap(self.compute_Ax_and_b_from_values_fn, (None, 0, None, None, None))(U_pad, points, dx, dy, dz)
        return vmap(self.compute_Ax_and_b_fn, (None, 0, None, None, None))(params, points, dx, dy, dz)

    @partial(jit, static_argnums=(0))
    def residual_vector_params(self, params, points, dx, dy, dz):
        r"""Loss function of the neural network."""
        lhs_rhs = self.rows_lhs_rhs(params, points, dx, dy, dz)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        return lhs - rhs

    # ------------- Hybrid training: the output layers by least squares
    @staticmethod
    def _last_layer_keys(params):
        """Haiku names of the output Linear of the u- and u+ MLPs (the highest-numbered linear)."""
        keys = []
        for side in ("m", "p"):
            prefix = f"double_mlp/~mlp_{side}_fn/linear"
            found = [k for k in params if k == prefix or k.startswith(prefix + "_")]
            if not found:
                raise ValueError(f"last_layer_ls needs model_type 'mlp' (DoubleMLP): no '{prefix}' layers in the params")
            keys.append(max(found, key=lambda k: 0 if k == prefix else int(k[len(prefix) + 1:])))
        return keys

    @staticmethod
    def _with_last_layer(params, keys, c):
        """params with the output layers (w then b, u- then u+) replaced by the flat vector c."""
        new = flax.core.unfreeze(params)
        i = 0
        for k in keys:
            w, b = params[k]["w"], params[k]["b"]
            new[k] = {"w": c[i:i + w.size].reshape(w.shape).astype(w.dtype),
                      "b": c[i + w.size:i + w.size + b.size].reshape(b.shape).astype(b.dtype)}
            i += w.size + b.size
        return flax.core.freeze(new) if isinstance(params, flax.core.FrozenDict) else new

    def solve_last_layer(self, params):
        """Hybrid least-squares / gradient-descent training (Cyr et al., arXiv:1912.04862; variable
        projection, Golub & Pereyra 1973). With the hidden layers fixed, each node value is linear in
        the output layers' weights c (w and b of u- and of u+), and each row is affine in the node
        values, so the residual is exactly r(c) = r(c0) + G (c - c0) and the loss 0.5 mean(r^2) is a
        quadratic in c. One step minimises it: c = c0 - pinv(G) r(c0). Adam and L-BFGS cannot finish
        this part: the directions it needs have singular values 1e-5..1e-8 of the largest
        (tools/jump_last_layer_ls.py). pinv keeps the singular values above last_layer_ls_rcond times
        the largest, which float32 resolves (G's entries carry ~1e-7 relative rounding) and which keeps
        the weights moderate; the solve is the correction to the trained weights, not the weights
        themselves, so no large numbers are subtracted."""
        t0 = time.time()
        keys = self._last_layer_keys(params)
        c0 = jnp.concatenate([jnp.concatenate([params[k]["w"].reshape(-1), params[k]["b"].reshape(-1)]) for k in keys])
        # The grid's spacing, not self.train_dx: single_GPU_train assigns its loop carry back to
        # self.train_dx/dy/dz inside jit, which leaves tracers there after training.
        dx, dy, dz = self.TD.gstate.dx, self.TD.gstate.dy, self.TD.gstate.dz

        def residual(p, c, points):
            return self.residual_vector_params(self._with_last_layer(p, keys, c), points, dx, dy, dz).reshape(-1)

        @jit
        def build(p, c0, points):
            # Column j of G is the derivative of the rows along the j-th output weight, by forward-mode
            # differentiation, one column at a time (the memory of one row evaluation).
            G = jax.lax.map(lambda e: jax.jvp(lambda c: residual(p, c, points), (c0,), (e,))[1],
                            jnp.eye(c0.size, dtype=c0.dtype)).T
            return residual(p, c0, points), G

        @jit
        def correction(G, r0):
            Q, R = jnp.linalg.qr(G)                         # reduced QR, then the SVD of the small R
            Ur, s, Vt = jnp.linalg.svd(R)
            keep = s > self.last_layer_ls_rcond * s[0]
            inv = jnp.where(keep, 1.0 / s, 0.0)
            return -Vt.T @ (inv * (Ur.T @ (Q.T @ r0))), s, keep.sum()

        r0, G = build(params, c0, self.train_points)
        dc, s, kept = correction(G, r0)
        c = c0 + dc
        new_params = self._with_last_layer(params, keys, c)
        r1 = residual(new_params, c, self.train_points)
        r1.block_until_ready()
        logger.info(
            f"Last-layer least squares: loss {0.5 * float(jnp.mean(r0 ** 2)):.4e} -> {0.5 * float(jnp.mean(r1 ** 2)):.4e}; "
            f"{int(kept)} of {c0.size} directions kept (singular values {float(s[0]):.3e} .. {float(s[-1]):.3e}); "
            f"|c| {float(jnp.linalg.norm(c0)):.3e} -> {float(jnp.linalg.norm(c)):.3e}; {time.time() - t0:.1f} s"
        )
        return new_params

    @partial(jit, static_argnums=(0))
    def residual_vector(self, u_vec):
        """u_vec, points and eval_gstate and outputs have same shapes on grids"""
        lhs_rhs = vmap(self.compute_Ax_and_b_discrete_fn, (None, None, 0, None, None, None))(
            self.eval_gstate, u_vec, self.eval_gstate.R, self.eval_gstate.dx, self.eval_gstate.dy, self.eval_gstate.dz
        )
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        return lhs - rhs

    @partial(jit, static_argnums=(0))
    def loss(self, params, points, dx, dy, dz):  # , cycle_level):
        r"""Loss function of the neural network."""
        # one_period = jnp.power(2, 3 * cycle_level)
        # # one_period = jnp.where(one_period < len(points), one_period, 1) # TODO: do we need this?
        # divisible_indices = jnp.arange(len(points)) % one_period == 0

        lhs_rhs = self.rows_lhs_rhs(params, points, dx, dy, dz)
        lhs, rhs = jnp.split(lhs_rhs, [1], axis=1)
        tot_loss = jnp.mean(optax.l2_loss(lhs, rhs))
        # residuals = optax.l2_loss(lhs, rhs)
        # tot_loss = jnp.mean(residuals.squeeze(), where=divisible_indices)

        # du_xmax = (self.evaluate_solution_fn(params, self.tr_gstate.R_xmax_boundary) - self.dir_bc_fn(self.tr_gstate.R_xmax_boundary)[...,jnp.newaxis])
        # du_xmin = (self.evaluate_solution_fn(params, self.tr_gstate.R_xmin_boundary) - self.dir_bc_fn(self.tr_gstate.R_xmin_boundary)[...,jnp.newaxis])
        # du_ymax = (self.evaluate_solution_fn(params, self.tr_gstate.R_ymax_boundary) - self.dir_bc_fn(self.tr_gstate.R_ymax_boundary)[...,jnp.newaxis])
        # du_ymin = (self.evaluate_solution_fn(params, self.tr_gstate.R_ymin_boundary) - self.dir_bc_fn(self.tr_gstate.R_ymin_boundary)[...,jnp.newaxis])
        # du_zmax = (self.evaluate_solution_fn(params, self.tr_gstate.R_zmax_boundary) - self.dir_bc_fn(self.tr_gstate.R_zmax_boundary)[...,jnp.newaxis])
        # du_zmin = (self.evaluate_solution_fn(params, self.tr_gstate.R_zmin_boundary) - self.dir_bc_fn(self.tr_gstate.R_zmin_boundary)[...,jnp.newaxis])
        # tot_loss += 0.01 * (jnp.mean(jnp.square(du_xmax)) + jnp.mean(jnp.square(du_xmin)) + jnp.mean(jnp.square(du_ymax)) + jnp.mean(jnp.square(du_ymin)) + jnp.mean(jnp.square(du_zmax)) + jnp.mean(jnp.square(du_zmin)))
        return tot_loss

    def compute_normal_gradient_solution_mp_on_interface_neural_network(self, params, points, dx, dy, dz):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(points, 1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(points, -1)
        normal_vecs = vmap(self.normal_point_fn, (0, None, None, None))(points, dx, dy, dz)
        grad_n_u_m = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u_m)
        grad_n_u_p = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u_p)
        return grad_n_u_m, grad_n_u_p

    def compute_gradient_solution_mp_neural_network(self, params, points):
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u_p = vmap(grad_u_at_point_fn, (0, None))(points, 1)
        grad_u_m = vmap(grad_u_at_point_fn, (0, None))(points, -1)
        return grad_u_m, grad_u_p

    def compute_normal_gradient_solution_on_interface_neural_network(self, params, points, dx, dy, dz):
        phi_flat = self.phi_interp_fn(points)
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)

        grad_u = vmap(grad_u_at_point_fn, (0, 0))(points, phi_flat)
        normal_vecs = vmap(self.normal_point_fn, (0, None, None, None))(points, dx, dy, dz)
        grad_n_u = vmap(jnp.dot, (0, 0))(jnp.squeeze(normal_vecs), grad_u)
        # grad_u = chunked_vmap(grad_u_at_point_fn, num_chunks=2, in_axes=(0, 0))(
        #     points,
        #     phi_flat,
        # )
        # normal_vecs = chunked_vmap(self.normal_point_fn, num_chunks=2, in_axes=(0, None, None, None))(
        #     points,
        #     dx,
        #     dy,
        #     dz,
        # )
        # grad_n_u = chunked_vmap(jnp.dot, num_chunks=2, in_axes=(0, 0))(
        #     jnp.squeeze(normal_vecs),
        #     grad_u,
        # )

        return grad_n_u

    def compute_gradient_solution_neural_network(self, params, points):
        phi_flat = self.phi_interp_fn(points)
        _, grad_u_at_point_fn = self.get_sol_grad_sol_fn(params)
        grad_u = vmap(grad_u_at_point_fn, (0, 0))(points, phi_flat)
        return grad_u

    # -------------------------------
    def evaluate_solution_and_gradients(self, params, eval_gstate, chunk=1 << 17):
        """Solution, gradient and normal gradient on the evaluation grid, chunk points at a time. Every
        quantity is pointwise, so the result is that of one call over the whole grid, with the memory of
        one chunk (one call over 128^3 points does not fit 8 GB for a 1 x 400 network)."""
        R = eval_gstate.R
        parts = [self._evaluate_chunk(params, R[i:i + chunk], eval_gstate.dx, eval_gstate.dy, eval_gstate.dz)
                 for i in range(0, R.shape[0], chunk)]
        final_solution, grad_u, grad_u_normal_to_interface = (jnp.concatenate([p[k] for p in parts]) for k in range(3))
        return final_solution, grad_u, grad_u_normal_to_interface

    @partial(jit, static_argnums=(0))
    def _evaluate_chunk(self, params, R, dx, dy, dz):
        final_solution = self.evaluate_solution_fn(params, R).reshape(-1)
        grad_u_normal_to_interface = self.compute_normal_gradient_solution_on_interface(params, R, dx, dy, dz)
        grad_u = self.compute_gradient_solution(params, R)
        return final_solution, grad_u, grad_u_normal_to_interface


def setup(
        initial_value_fn: Callable[..., Array],
        dirichlet_bc_fn: Callable[..., Array],
        lvl_set_fn: Callable[..., Array],
        mu_m_fn_: Callable[..., Array],
        mu_p_fn_: Callable[..., Array],
        k_m_fn_: Callable[..., Array],
        k_p_fn_: Callable[..., Array],
        f_m_fn_: Callable[..., Array],
        f_p_fn_: Callable[..., Array],
        alpha_fn_: Callable[..., Array],
        beta_fn_: Callable[..., Array],
        nonlinear_op_m=None,
        nonlinear_op_p=None,
) -> InitFn:
    u_0_fn = vmap(initial_value_fn)
    dir_bc_fn = vmap(dirichlet_bc_fn)
    phi_fn = vmap(lvl_set_fn)
    mu_m_fn = vmap(mu_m_fn_)
    mu_p_fn = vmap(mu_p_fn_)
    k_m_fn = vmap(k_m_fn_)
    k_p_fn = vmap(k_p_fn_)
    f_m_fn = vmap(f_m_fn_)
    f_p_fn = vmap(f_p_fn_)
    alpha_fn = vmap(alpha_fn_)
    beta_fn = vmap(beta_fn_)

    if nonlinear_op_m is None:
        logger.warning("nonlinear_op_m(u) is not defined. Setting it to zero.")

        def nonlinear_op_m(x):
            return 0.0

    if nonlinear_op_p is None:
        logger.warning("nonlinear_op_m(u) is not defined. Setting it to zero.")

        def nonlinear_op_p(x):
            return 0.0

    sim_state_fn = PoissonSimStateFn(
        u_0_fn,
        dir_bc_fn,
        phi_fn,
        mu_m_fn,
        mu_p_fn,
        k_m_fn,
        k_p_fn,
        f_m_fn,
        f_p_fn,
        alpha_fn=alpha_fn,
        beta_fn=beta_fn,
        nonlinear_op_m=nonlinear_op_m,
        nonlinear_op_p=nonlinear_op_p,
    )

    def init_fn(
            lvl_gstate: GridState = None,
            tr_gstate: GridState = None,
            eval_gstate: GridState = None,
            num_epochs: int = 1000,
            batch_size: int = 131072,
            algorithm: int = 0,
            mgrad_over_pgrad_scalefactor: int = 1,
            multi_gpu: bool = False,
            checkpoint_interval: int = 1000,
            checkpoint_dir: str = "./checkpoints",
            results_dir: str = "./",
            loss_plot_name: str = "solver_loss",
            optimizer_dict: dict = {
                "optimizer_name": "custom",
                "learning_rate": 1e-3,
                "sched": {"scheduler_name": "exponential", "decay_rate": 0.9},
            },
            model_dict: dict = {
                "name": None,
                "model_type": "mlp",
                "mlp": {
                    "hidden_layers_m": 1,
                    "hidden_dim_m": 1,
                    "activation_m": "jnp.tanh",
                    "hidden_layers_p": 2,
                    "hidden_dim_p": 10,
                    "activation_p": "jnp.tanh",
                },
                "resnet": {
                    "res_blocks_m": 3,
                    "res_dim_m": 40,
                    "activation_m": "nn.tanh",
                    "res_blocks_p": 3,
                    "res_dim_p": 80,
                    "activation_p": "nn.tanh",
                },
                "preconditioner": {
                    "enable": False,
                },
            },
            restart: bool = False,
            restart_checkpoint_dir: str = "./checkpoints",
            print_rate: int = 1,
            alternate_tets: bool = True,
            jump_scheme: str = "original",
            jump_ell: float = 1.0e4,
            jump_cut_floor: float = 0.05,
            jump_eval_once: bool = False,
            last_layer_ls: bool = False,
            last_layer_ls_rcond: float = 1.0e-6,
    ) -> Tuple[PoissonSimState, SolveFn]:
        R = eval_gstate.R
        PHI = phi_fn(R)
        DIRBC = dir_bc_fn(R)
        U = u_0_fn(R)
        MU_M = mu_m_fn(R)
        MU_P = mu_p_fn(R)
        K_M = k_m_fn(R)
        K_P = k_p_fn(R)
        F_M = f_m_fn(R)
        F_P = f_p_fn(R)
        ALPHA = alpha_fn(R)
        BETA = beta_fn(R)

        def solve_fn(sim_state: PoissonSimState) -> PoissonSimState:
            trainer = Trainer(
                lvl_gstate,
                tr_gstate,
                eval_gstate,
                sim_state,
                sim_state_fn,
                algorithm,
                mgrad_over_pgrad_scalefactor=mgrad_over_pgrad_scalefactor,
                lvl_set_fn=lvl_set_fn,
                num_epochs=num_epochs,
                multi_gpu=multi_gpu,
                batch_size=batch_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                results_dir=results_dir,
                loss_plot_name=loss_plot_name,
                optimizer_dict=optimizer_dict,
                restart=restart,
                restart_checkpoint_dir=restart_checkpoint_dir,
                print_rate=print_rate,
                alternate_tets=alternate_tets,
                jump_scheme=jump_scheme,
                jump_ell=jump_ell,
                jump_cut_floor=jump_cut_floor,
                jump_eval_once=jump_eval_once,
                last_layer_ls=last_layer_ls,
                last_layer_ls_rcond=last_layer_ls_rcond,
                model_dict=model_dict,
            )
            (
                final_solution,
                grad_u,
                grad_u_normal_to_interface,
                epoch_store,
                loss_epochs,
            ) = trainer.solve()
            sim_state = (
                dataclasses.replace(
                    sim_state,
                    solution=final_solution,
                    grad_solution=grad_u,
                    grad_normal_solution=grad_u_normal_to_interface,
                ),
                epoch_store,
                loss_epochs,
            )
            return sim_state

        return (
            PoissonSimState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, None, None, BETA, None, None),
            solve_fn,
        )

    return init_fn


def setup_Robin(
        initial_value_fn: Callable[..., Array],
        dirichlet_bc_fn: Callable[..., Array],
        lvl_set_fn: Callable[..., Array],
        mu_m_fn_: Callable[..., Array],
        mu_p_fn_: Callable[..., Array],
        k_m_fn_: Callable[..., Array],
        k_p_fn_: Callable[..., Array],
        f_m_fn_: Callable[..., Array],
        f_p_fn_: Callable[..., Array],
        g_m_fn_: Callable[..., Array],
        g_p_fn_: Callable[..., Array],
        alpha_fn_: Callable[..., Array],
        beta_fn_: Callable[..., Array],
        nonlinear_op_m=None,
        nonlinear_op_p=None
) -> InitFn_Robin:
    u_0_fn = vmap(initial_value_fn)
    dir_bc_fn = vmap(dirichlet_bc_fn)
    phi_fn = vmap(lvl_set_fn)
    mu_m_fn = vmap(mu_m_fn_)
    mu_p_fn = vmap(mu_p_fn_)
    k_m_fn = vmap(k_m_fn_)
    k_p_fn = vmap(k_p_fn_)
    f_m_fn = vmap(f_m_fn_)
    f_p_fn = vmap(f_p_fn_)
    g_m_fn = vmap(g_m_fn_)
    g_p_fn = vmap(g_p_fn_)
    alpha_fn = vmap(alpha_fn_)
    beta_fn = vmap(beta_fn_)

    if nonlinear_op_m is None:
        logger.warning("nonlinear_op_m(u) is not defined. Setting it to zero.")

        def nonlinear_op_m(x):
            return 0.0

    if nonlinear_op_p is None:
        logger.warning("nonlinear_op_m(u) is not defined. Setting it to zero.")

        def nonlinear_op_p(x):
            return 0.0

   
    sim_state_fn = PoissonSimStateFn(
        u_0_fn,
        dir_bc_fn,
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
        nonlinear_op_m,
        nonlinear_op_p
    )
    

    def init_fn(
            lvl_gstate: GridState = None,
            tr_gstate: GridState = None,
            eval_gstate: GridState = None,
            num_epochs: int = 1000,
            batch_size: int = 131072,
            algorithm: int = 0,
            mgrad_over_pgrad_scalefactor: int = 1,
            multi_gpu: bool = False,
            checkpoint_interval: int = 1000,
            checkpoint_dir: str = "./checkpoints",
            results_dir: str = "./",
            loss_plot_name: str = "solver_loss",
            optimizer_dict: dict = {
                "optimizer_name": "custom",
                "learning_rate": 1e-3,
                "sched": {"scheduler_name": "exponential", "decay_rate": 0.9},
            },
            model_dict: dict = {
                "name": None,
                "model_type": "mlp",
                "mlp": {
                    "hidden_layers_m": 1,
                    "hidden_dim_m": 1,
                    "activation_m": "jnp.tanh",
                    "hidden_layers_p": 2,
                    "hidden_dim_p": 10,
                    "activation_p": "jnp.tanh",
                },
                "resnet": {
                    "res_blocks_m": 3,
                    "res_dim_m": 40,
                    "activation_m": "nn.tanh",
                    "res_blocks_p": 3,
                    "res_dim_p": 80,
                    "activation_p": "nn.tanh",
                },
                "preconditioner": {
                    "enable": False,
                },
            },
            restart: bool = False,
            restart_checkpoint_dir: str = "./checkpoints",
            print_rate: int = 1,
            train_omega_minus_only: bool = False,
            robin_weight_length=1.0,
            alternate_tets: bool = True,
    ) -> Tuple[PoissonSimState, SolveFn]:
        R = eval_gstate.R
        PHI = phi_fn(R)
        DIRBC = dir_bc_fn(R)
        U = u_0_fn(R)
        MU_M = mu_m_fn(R)
        MU_P = mu_p_fn(R)
        K_M = k_m_fn(R)
        K_P = k_p_fn(R)
        F_M = f_m_fn(R)
        F_P = f_p_fn(R)
        G_M = g_m_fn(R)
        G_P = g_p_fn(R)
        ALPHA = alpha_fn(R)
        BETA = beta_fn(R)




        def solve_fn(sim_state: PoissonSimState) -> PoissonSimState:
            trainer = Trainer(
                lvl_gstate,
                tr_gstate,
                eval_gstate,
                sim_state,
                sim_state_fn,
                algorithm,
                mgrad_over_pgrad_scalefactor=mgrad_over_pgrad_scalefactor,
                lvl_set_fn=lvl_set_fn,
                num_epochs=num_epochs,
                multi_gpu=multi_gpu,
                batch_size=batch_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                results_dir=results_dir,
                loss_plot_name=loss_plot_name,
                optimizer_dict=optimizer_dict,
                restart=restart,
                restart_checkpoint_dir=restart_checkpoint_dir,
                print_rate=print_rate,
                train_omega_minus_only=train_omega_minus_only,
                robin_weight_length=robin_weight_length,
                alternate_tets=alternate_tets,
                model_dict=model_dict,
            )
            (
                final_solution,
                grad_u,
                grad_u_normal_to_interface,
                epoch_store,
                loss_epochs,
            ) = trainer.solve()
            sim_state = (
                dataclasses.replace(
                    sim_state,
                    solution=final_solution,
                    grad_solution=grad_u,
                    grad_normal_solution=grad_u_normal_to_interface,
                ),
                epoch_store,
                loss_epochs,
            )
            return sim_state
        # PoissonSimState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, BETA, None, None),
        return (
            PoissonSimState(PHI, U, DIRBC, MU_M, MU_P, K_M, K_P, F_M, F_P, ALPHA, G_M, G_P, BETA, None, None),
            solve_fn,
        )



    return init_fn
