import dataclasses

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from event2vec.dataset import ReweightableDataset
from event2vec.prior import ParameterPrior

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
    observables: jax.Array
    """The observed data points, shape (num_events, 2)"""
    gen_parameters: jax.Array
    """The parameters used to sample this event, shape (num_events, 3)"""

    @property
    def observable_dim(self) -> int:
        return 2

    @property
    def parameter_dim(self) -> int:
        return 3

    def likelihood(self, param: jax.Array) -> jax.Array:
        if param.shape == (3,):
            # Expand dimensions for vmap (as gen_parameters would be (num_events, 3))
            param = jnp.repeat(param[None, :], len(self), axis=0)
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
