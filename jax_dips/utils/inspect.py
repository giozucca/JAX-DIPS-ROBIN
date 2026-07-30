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


def _print_callback(arg, transforms):
    i, n_iter, message = arg
    logger.info(f"iteration {i}/{n_iter} - loss: {message}")


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
