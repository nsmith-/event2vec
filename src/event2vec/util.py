import jax
import jax.numpy as jnp

EPS = jnp.finfo(jnp.float32).eps
"A small constant to avoid numerical issues with log(0) or division by zero."


def tril_outer_product(vec: jax.Array) -> jax.Array:
    """Calculates the lower-triangular part of the outer product of a vector with itself.

    Useful to construct a vector representation of a quadratic formula (e.g. for SMEFT parameterizations).
    """
    outer = vec[..., None, :] * vec[..., None]
    il = jnp.tril_indices(vec.shape[-1])
    return outer[..., il[0], il[1]]
