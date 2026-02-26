import dataclasses
from dataclasses import KW_ONLY, InitVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from event2vec.nontrainable import FreezableModule


class ExpAffineLogistic(FreezableModule):
    r"""Returns the following elementwise function

    .. math::
        a(x) &= \exp( (l_{\text{max}} - l_{\text{min}}) \cdot \sigma(x) + l_{\text{min}} ) \\
             &= \exp(l_{\text{max}} \cdot \sigma(x) + l_{\text{min}} \cdot \sigma(-x))

    where :math:`\sigma(x) = \frac{1}{1 + \exp(-x)}` is the logistic function.
    """

    is_static: bool = dataclasses.field(default=True, init=False)

    _: KW_ONLY

    log_min: Float[Array, "#N"]
    r"""Minimum value of the output, in log space, :math:`l_{\text{min}}`."""
    log_max: Float[Array, "#N"]
    r"""Maximum value of the output, in log space, :math:`l_{\text{max}}`."""
    copy: InitVar[bool] = True  # type: ignore[assignment]

    def __post_init__(self, copy):
        self.log_min = jnp.array(self.log_min, copy=copy)
        self.log_max = jnp.array(self.log_max, copy=copy)

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        return jnp.exp(
            self.log_max * jax.nn.sigmoid(x) + self.log_min * jax.nn.sigmoid(-x)
        )


__all__ = ["ExpAffineLogistic"]
