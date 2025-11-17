import dataclasses
from dataclasses import KW_ONLY, InitVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from event2vec.models import Model


class ExpAffineLogistic(Model):
    """Returns the following elementwise function
        output(x) = exp( (log_max - log_min) * logistic(x) + log_min )
                  = exp( log_max * logistic(x) + log_min * logistic(-x))
    where logistic(x) = 1/(1 + exp(-x)).
    """

    is_static: bool = dataclasses.field(default=True, init=False)

    _: KW_ONLY

    log_min: Float[Array, "#N"]
    log_max: Float[Array, "#N"]
    copy: InitVar[bool] = True

    def __post_init__(self, copy):
        self.log_min = jnp.array(self.log_min, copy=copy)
        self.log_max = jnp.array(self.log_max, copy=copy)

    def __call__(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        return jnp.exp(
            self.log_max * jax.nn.sigmoid(x) + self.log_min * jax.nn.sigmoid(-x)
        )
