from abc import abstractmethod
import dataclasses
from typing import Self
import equinox as eqx
import jax
import jax.scipy.stats as jstats
import jax.numpy as jnp

from event2vec.prior import ParameterPrior


class Dataset(eqx.Module):
    """Abstract dataset

    The leading dimension of all member arrays is assumed to be the event index.
    """

    observables: jax.Array
    """Observables of the dataset, e.g. event kinematics, etc."""

    def __getitem__(self, key) -> Self:
        return jax.tree_util.tree_map(lambda x: x[key], self)

    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return self.observables.shape[0]

    def split(self, fraction: float, key: jax.Array) -> tuple[Self, Self]:
        """Split the dataset into two parts.

        Args:
            fraction (float): Fraction of the dataset to use for the first part.
                The second part will contain the remaining fraction.
            key (jnp.Array): Random key if needed for splitting.

        NOTE: We assume the data is i.i.d. so we don't need to shuffle the data.
        A subclass may override this if it needs to shuffle.
        """
        if not (0.0 < fraction < 1.0):
            raise ValueError("Fraction must be between 0 and 1.")
        split_index = int(len(self) * fraction)
        return self[:split_index], self[split_index:]

    def iter_batch(self, batch_size: int, omit_last: bool = True):
        """Yield batches of data from the dataset.

        Args:
            batch_size (int): Size of each batch.
            omit_last (bool): If True, the last batch will be omitted if it is smaller than batch_size.
                This is useful because then we don't re-JIT the training loop for the last batch.
        """
        dataset_size = len(self.observables)
        for start in range(0, dataset_size, batch_size):
            if omit_last and start + batch_size > dataset_size:
                break
            yield self[start : start + batch_size]


class DatasetWithLikelihood(Dataset):
    """Abstract dataset which has a likelihood function.

    Latent data may be needed beyond the observables, in which case the samples
    are assumed to be from the joint distribution of the observables and the latent variables.
    """

    @abstractmethod
    def likelihood(self, param: jax.Array) -> jax.Array:
        """Compute the likelihood of the dataset w.r.t. the given parameter vector."""
        msg = (
            "This method should be implemented by subclasses of DatasetWithLikelihood."
        )
        raise NotImplementedError(msg)


class ReweightableDataset(DatasetWithLikelihood):
    """A dataset that can be reweighted by a parameter vector."""

    gen_parameters: jax.Array
    """The parameters used to sample this event"""

    def weight(self, param: jax.Array) -> jax.Array:
        """Compute the weight of the dataset w.r.t. the given parameter vector."""
        # TODO: any way to catch denom==0?
        denom = self.likelihood(self.gen_parameters)
        num = self.likelihood(param)
        return num / denom


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


class ToyDataset(ReweightableDataset):
    def likelihood(self, param: jax.Array) -> jax.Array:
        return jax.vmap(_likelihood_event)(self.observables, param)


@dataclasses.dataclass
class ToyDatasetFactory:
    """Factory for creating a toy dataset."""

    len: int
    """Number of events in the dataset."""
    param_prior: ParameterPrior
    """Prior distribution for the parameters of the dataset."""

    def __call__(self, *, key: jax.Array) -> ToyDataset:
        param_key, event_key = jax.random.split(key)
        param = jax.vmap(self.param_prior.sample)(
            key=jax.random.split(param_key, self.len)
        )
        event = jax.vmap(_sample_event)(
            param, key=jax.random.split(event_key, self.len)
        )
        return ToyDataset(observables=event, gen_parameters=param)
