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
import time

import jax

try:
    from jax.experimental import host_callback
except ImportError:
    host_callback = None

try:
    import jax.debug

    has_jax_debug = hasattr(jax.debug, "callback")
except ImportError:
    has_jax_debug = False

logger = logging.getLogger(__name__)


def print_architecture(params):
    logger.info("Architecture Summary (trainable parameters):\n")

    num_params = 0
    for pytree in params:
        leaves = jax.tree_util.tree_leaves(pytree)
        cur_shape = jax.tree_util.tree_map(lambda x: x.shape, params[leaves[0]])
        logger.info(f"{repr(pytree):<45} \t has trainable parameters:\t {cur_shape}")
        shapes = [val for key, val in cur_shape.items()]
        for val in shapes:
            res = 1
            for elem in val:
                res *= elem
            num_params += res
    logger.info(f"Total number of trainable parameters = {num_params} ...\n")


# ----------------------------------------------------------------


# (epoch, wall-clock time) at each progress print, for the per-epoch timing below.
_epoch_times = []


def _print_callback(arg, transforms):
    i, n_iter, message = arg
    _epoch_times.append((int(i), time.perf_counter()))
    logger.info(f"iteration {i}/{n_iter} - loss: {message}")


def reset_epoch_timing():
    _epoch_times.clear()


def epoch_timing_summary():
    """Seconds per epoch between consecutive progress prints. The first print happens after
    compilation, so compilation is excluded."""
    pts = sorted(_epoch_times)
    per = [(t1 - t0) / (e1 - e0) for (e0, t0), (e1, t1) in zip(pts, pts[1:]) if e1 > e0]
    if not per:
        return None
    mean = sum(per) / len(per)
    std = (sum((p - mean) ** 2 for p in per) / len(per)) ** 0.5
    return (f"time per epoch: {mean:.4f} +- {std:.4f} s "
            f"(over {len(per)} intervals of {pts[1][0] - pts[0][0]} epochs, compilation excluded)")


@jax.jit
def progress_bar(arg, result):
    "Print progress of loop only if iteration number is a multiple of the print_rate"
    i, n_iter, print_rate, message = arg

    if has_jax_debug:

        def true_fn(_):
            jax.debug.callback(lambda x: _print_callback(x, None), (i, n_iter, message))
            return result

    elif host_callback is not None:

        def true_fn(_):
            return host_callback.id_tap(
                _print_callback, (i, n_iter, message), result=result
            )

    else:

        def true_fn(_):
            return result

    result = jax.lax.cond(
        i % print_rate == 0,
        true_fn,
        lambda _: result,
        operand=None,
    )
    return result
