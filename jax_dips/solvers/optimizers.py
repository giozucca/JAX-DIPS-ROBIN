import logging

import optax
from optax._src.base import GradientTransformation

import jaxopt
from jaxopt import LBFGS, LevenbergMarquardt
from jaxopt.loss import huber_loss

logger = logging.getLogger(__name__)


def get_scheduler(
    scheduler_name: str = "exponential",
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    transition_steps: int = 100,
    **kwargs,
):
    if scheduler_name == "exponential":
        logger.info(f"Using Exponential Scheduler (transition_steps={transition_steps}, decay_rate={decay_rate})")
        scheduler = optax.exponential_decay(
            init_value=learning_rate, transition_steps=transition_steps, decay_rate=decay_rate, **kwargs
        )
    elif scheduler_name == "polynomial":
        logger.info(f"Using Polynomial Scheduler (transition_steps={transition_steps})")
        scheduler = optax.polynomial_schedule(
            init_value=learning_rate, end_value=0.0, power=1, transition_steps=transition_steps, **kwargs
        )
    elif scheduler_name == "cosine":
        # Anneals learning_rate -> 0 over transition_steps. Needs no decay_rate tuning,
        # so it is the safest choice when transition_steps spans the whole run.
        logger.info(f"Using Cosine Scheduler (decay_steps={transition_steps})")
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=transition_steps, **kwargs
        )
    elif scheduler_name == "constant":
        logger.info(f"Using constant learning rate {learning_rate} (no annealing)")
        scheduler = optax.constant_schedule(learning_rate)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    return scheduler


def chained_adam(
    scheduler_name: str = "exponential",
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    transition_steps: int = 1000,
    max_norm: float = 1.0,
    eps: float = 1e-16,
    **kwargs,
) -> GradientTransformation:
    scheduler = get_scheduler(
        scheduler_name=scheduler_name,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
        **kwargs,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.scale_by_adam(eps=eps),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )
    return optimizer


def get_optimizer(
    optimizer_name: str = "custom",
    scheduler_name: str = "exponential",
    learning_rate: float = 1e-2,
    decay_rate: float = 0.96,
    transition_steps: int = 1000,
    weight_decay: float = 1e-4,
    max_norm: float = 1.0,
    eps: float = 1e-16,
    loss_fn: object = None,
    **kwargs,
) -> GradientTransformation:
    # NOTE on `eps`: optax defaults to 1e-8, which is far too large here. The loss is a
    # mean over Jacobi-preconditioned FV rows, and the residual's sensitivity to the
    # solution error carries an explicit power of the grid spacing (dx^2 at interior
    # cells, dx at interface cells). Interface cells dominate, so the gradient scales as
    #     ||dL/dtheta|| ~ (pi/8) (alpha/6 mu)^2 * dx^3 * e
    # for a solution error e. Adam's update is g / (sqrt(v) + eps) with sqrt(v) ~ |g|, so
    # it stalls once |g| falls to eps -- i.e. at a solution-error floor
    #     e_floor ~ eps / (C dx^3),   C = (pi/8)(alpha/6 mu)^2 ~ 0.011
    # which GROWS like dx^-3 under refinement. With eps=1e-8 that floor is ~6e-5 at
    # Nx=8 (harmless, truncation dominates) but ~3e-2 at Nx=64, which is exactly where
    # sphere_Robin was plateauing. Below the floor progress freezes logarithmically
    # (halving e halves g, which doubles the throttle), which is why doubling the epoch
    # count moved the answer by ~2% and why widening the network did not help at all.
    # Setting eps well below the smallest expected gradient restores Adam's scale
    # invariance, so the achievable error stops depending on the resolution.
    if optimizer_name == "custom":
        logger.info("Using chained Adam optimizer")
        return chained_adam(
            scheduler_name=scheduler_name,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            transition_steps=transition_steps,
            max_norm=max_norm,
            eps=eps,
            **kwargs,
        )

    elif optimizer_name == "adam":
        logger.info(f"Using Adam optimizer (eps={eps})")
        return optax.adam(
            learning_rate,
            eps=eps,
            **kwargs,
        )

    elif optimizer_name == "rmsprop":
        logger.info(f"Using RMSprop optimizer (eps={eps})")
        return optax.rmsprop(
            learning_rate,
            eps=eps,
            **kwargs,
        )
    
    elif optimizer_name == "adamw":
        # NOTE: learning_rate is a *schedule* here, not a scalar. Previously this branch
        # passed the raw scalar, so the sched: block in the config was silently ignored
        # and training ran at a constant LR for the whole run.
        # ("adam" and "rmsprop" above still use a constant LR.)
        logger.info(f"Using adamw optimizer (eps={eps})")
        return optax.adamw(
            learning_rate=get_scheduler(
                scheduler_name=scheduler_name,
                learning_rate=learning_rate,
                decay_rate=decay_rate,
                transition_steps=transition_steps,
            ),
            weight_decay=weight_decay,
            eps=eps,
        )

     
    #elif optimizer_name == "lbfgs":
    #    logger.info("Using jaxopt.LBFGS optimizer#")
        #lbfgs = LBFGS_adapter(loss_fn)
        #return lbfgs

    else:
        logger.error("Unknown optimizer: {}".format(optimizer_name))
        raise ValueError("Unknown optimizer: {}".format(optimizer_name))


# class LBFGS_adapter(LBFGS):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def init(self, params, *args, **kwargs):
#         params, opt_state = self.init_state(params, *args, **kwargs)
#         return opt_state
