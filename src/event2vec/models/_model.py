from collections.abc import Iterable
from abc import abstractmethod
from typing import Callable
import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

class Model(eqx.Module):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class MLP(Model):
    out_shape: tuple[int, ...]
    layers: list[Callable]
    activations: list[Callable]

    def __init__(self, *,
                 in_shape: Iterable[int],
                 out_shape: Iterable[int],
                 hidden_widths: Iterable[int],
                 hidden_activation: Callable,
                 final_activation: Callable,
                 use_hidden_bias: bool = True,
                 use_final_bias: bool = True,
                 key: PRNGKeyArray):

        self.out_shape = tuple(out_shape)

        layer_dims = []
        layer_dims.append(math.prod(in_shape))
        layer_dims.extend(hidden_widths)
        layer_dims.append(math.prod(self.out_shape))

        num_layers = len(layer_dims) - 1
        num_hidden_layers = len(layer_dims) - 2

        keys = jax.random.split(key, num_layers)
        use_biases = [use_hidden_bias] * num_hidden_layers + [use_final_bias]

        self.layers = [
            eqx.nn.Linear(
                in_features=layer_dims[i], out_features=layer_dims[i+1],
                use_bias=use_biases[i], key = keys[i]
            )
            for i in range(num_layers)
        ]

        self.activations = (
            [hidden_activation] * num_hidden_layers
            + [final_activation]
        )

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        x = jnp.ravel(x)                        # flatten input

        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        return x.reshape(self.out_shape)        # reshape output
