import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from ..dataset import ReweightableDataset
from ..prior import ParameterPrior

MEAN = jnp.array([0.0, 0.0])
"""Mean of the multivariate normal distributions used in the toy model. (all the same)"""
COVARIANCES = jnp.array(
    [
        [[1.0, 0.0], [0.0, 1.0]],  # Default
        [[4.0, 0.0], [0.0, 0.5]],  # Option 0
        [[0.5, 0.0], [0.0, 4.0]],  # Option 1
    ]
)
"""Covariance matrices of the multivariate normal distributions used in the toy model."""


def _sample_event(param: jax.Array, *, key: jax.Array):
    if param.shape[-1] != COVARIANCES.shape[0]:
        raise ValueError(
            f"Parameter vector ({param.shape[-1]}) must match number of distributions ({COVARIANCES.shape[0]})"
        )
    dist_key, norm_key = jax.random.split(key, 2)
    selector = jax.random.categorical(dist_key, logits=jnp.log(param))

    return jax.random.multivariate_normal(
        norm_key,
        mean=MEAN,
        cov=COVARIANCES[selector, ...],
    )


def _likelihood_event(observables: jax.Array, param: jax.Array) -> jax.Array:
    pdf_component = jstats.multivariate_normal.pdf(
        observables, mean=MEAN, cov=COVARIANCES
    )
    return pdf_component @ param


class GaussMixtureDataset(ReweightableDataset):
    def likelihood(self, param: jax.Array) -> jax.Array:
        return jax.vmap(_likelihood_event)(self.observables, param)


@dataclasses.dataclass
class GaussMixtureDatasetFactory:
    """Factory for creating a toy dataset."""

    len: int
    """Number of events in the dataset."""
    param_prior: ParameterPrior
    """Prior distribution for the parameters of the dataset."""

    def __call__(self, *, key: jax.Array) -> GaussMixtureDataset:
        param_key, event_key = jax.random.split(key)
        param = jax.vmap(self.param_prior.sample)(
            key=jax.random.split(param_key, self.len)
        )
        event = jax.vmap(_sample_event)(
            param, key=jax.random.split(event_key, self.len)
        )
        return GaussMixtureDataset(observables=event, gen_parameters=param)
