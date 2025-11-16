import jax.numpy as jnp
from jaxtyping import Float, Array


def mvn_first_moment(
    mean: Float[Array, "N"], cov: Float[Array, "N N"], is_central: bool = False
) -> Float[Array, "N"]:
    if is_central:
        return jnp.zeros_like(mean)

    return mean


def mvn_second_moment(
    mean: Float[Array, "N"], cov: Float[Array, "N N"], is_central: bool = False
) -> Float[Array, "N N"]:
    central_moment = (cov + cov.T) / 2

    if is_central:
        return central_moment

    return central_moment + (mean[None, :] * mean[:, None])


def mvn_third_moment(
    mean: Float[Array, "N"], cov: Float[Array, "N N"], is_central: bool = False
) -> Float[Array, "N N N"]:
    cov = (cov + cov.T) / 2

    tmp = mean[:, None, None] * cov[None, :, :]

    if is_central:
        return jnp.zeros_like(tmp)

    non_central_moment = (
        tmp + jnp.transpose(tmp, axes=(1, 0, 2)) + jnp.transpose(tmp, axes=(1, 2, 0))
    )

    return non_central_moment


def mvn_fourth_moment(
    mean: Float[Array, "N"], cov: Float[Array, "N N"], is_central: bool = False
) -> Float[Array, "N N N N"]:
    cov = (cov + cov.T) / 2

    tmp = cov[:, :, None, None] * cov[None, None, :, :]
    central_moment = (
        tmp
        + jnp.transpose(tmp, axes=(0, 2, 1, 3))
        + jnp.transpose(tmp, axes=(0, 2, 3, 1))
    )

    if is_central:
        return central_moment

    tmp = mean[:, None, None, None] * mean[None, :, None, None] * cov[None, None, :, :]
    shift_1 = (
        tmp
        + jnp.transpose(tmp, axes=(0, 2, 1, 3))
        + jnp.transpose(tmp, axes=(0, 2, 3, 1))
        + jnp.transpose(tmp, axes=(2, 0, 1, 3))
        + jnp.transpose(tmp, axes=(2, 0, 3, 1))
        + jnp.transpose(tmp, axes=(2, 3, 0, 1))
    )

    shift_2 = (
        mean[:, None, None, None]
        * mean[None, :, None, None]
        * mean[None, None, :, None]
        * mean[None, None, None, :]
    )

    return central_moment + shift_1 + shift_2
