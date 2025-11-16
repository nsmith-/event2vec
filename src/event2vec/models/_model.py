from abc import abstractmethod
from typing import Callable, Sequence
import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class Model(eqx.Module):
    is_static: eqx.AbstractVar[bool]
    """Whether the class instance should be treated as static by the
    utility function `partition_trainable_and_static`."""

    def __check_init__(self):
        assert isinstance(self.is_static, bool)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MLP(Model):
    out_shape: tuple[int, ...]
    layers: list[Callable]
    activations: list[Callable]
    is_static: bool

    def __init__(
        self,
        *,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        hidden_widths: Sequence[int],
        hidden_activation: Callable,
        final_activation: Callable,
        key: PRNGKeyArray,
        use_hidden_bias: bool = True,
        use_final_bias: bool = True,
        is_static: bool = False,
    ):
        self.out_shape = tuple(out_shape)

        layer_dims = []
        layer_dims.append(math.prod(in_shape))
        layer_dims.extend(hidden_widths)
        layer_dims.append(math.prod(self.out_shape))

        num_layers = len(layer_dims) - 1
        num_hidden_layers = len(layer_dims) - 2

        keys = jax.random.split(key, num_layers)
        use_biases = [use_hidden_bias] * num_hidden_layers + [use_final_bias]

        self.layers = tuple(
            eqx.nn.Linear(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                use_bias=use_biases[i],
                key=keys[i],
            )
            for i in range(num_layers)
        )

        self.activations = tuple(
            [hidden_activation] * num_hidden_layers + [final_activation]
        )

        self.is_static = bool(is_static)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        x = jnp.ravel(x)  # flatten input

        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        return x.reshape(self.out_shape)  # reshape output
