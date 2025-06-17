from abc import abstractmethod
from typing import Self
import equinox as eqx
import jax
import jax.scipy.stats as jstats
import jax.numpy as jnp

from event2vec.prior import ToyParameterPrior


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


def _likelihood_one(observables: jax.Array, param: jax.Array) -> jax.Array:
    pdf_component = jstats.multivariate_normal.pdf(
        observables, mean=MEAN, cov=COVARIANCES
    )
    return pdf_component @ param


class ToyDataset(ReweightableDataset):
    def likelihood(self, param: jax.Array) -> jax.Array:
        return jax.vmap(_likelihood_one)(self.observables, param)


class ToyDatasetFactory(eqx.Module):
    pass


def sample_event(param: jax.Array, *, key: jax.Array):
    if param.shape[-1] != COVARIANCES.shape[0]:
        raise ValueError(
            f"Parameter vector ({param.shape[-1]}) must match number of distributions ({COVARIANCES.shape[0]})"
        )
    dist_key, norm_key = jax.random.split(key, 2)
    selector = jax.random.categorical(dist_key, logits=jnp.log(param))

    event = jax.random.multivariate_normal(
        norm_key,
        mean=MEAN,
        cov=COVARIANCES[selector, ...],
    )

    return event


def true_llr_function(events: ToyDataset, param_0: jax.Array, param_1: jax.Array):
    return jnp.log(events.likelihood(param_1)) - jnp.log(events.likelihood(param_0))


def get_data(
    *,
    dataset_size: int = 100_000,
    validation_fraction: float = 0.15,
    key: jax.Array,
):
    param_key, split_key, event_key = jax.random.split(key, 3)
    param_0 = jnp.full(shape=(dataset_size, 3), fill_value=jnp.array([1.0, 0.0, 0.0]))
    prior = ToyParameterPrior(alpha=jnp.array([9.0, 3.0, 3.0]))
    param_1 = jax.vmap(prior.sample)(key=jax.random.split(param_key, dataset_size))

    label = jax.random.randint(split_key, shape=(dataset_size,), minval=0, maxval=2)
    param = jnp.where(
        (label == 0)[:, None],
        param_0,
        param_1,
    )

    event = jax.vmap(sample_event)(param, key=jax.random.split(event_key, dataset_size))

    N_train = dataset_size - int(dataset_size * validation_fraction)
    data = ToyDataset(observables=event, gen_parameters=param)
    data_train = data[:N_train]
    data_validation = data[N_train:]
    return data_train, data_validation


if __name__ == "__main__":
    data_train, data_validation = get_data(
        dataset_size=1000,
        validation_fraction=0.15,
        key=jax.random.PRNGKey(42),
    )
    print("Training Data:", data_train)
    print("Validation Data:", data_validation)

    llr, score = jax.vmap(
        jax.value_and_grad(true_llr_function, argnums=1),
        in_axes=(0, 0, None),
    )(
        data_train,
        data_train.gen_parameters,
        jnp.array([1.0, 0.0, 0.0]),
    )

    print(f"True LLR: {llr[:5]}")
    print("True Scores:")
    print(f"along a1: {score[:5, 1]}")
    print(f"along a2: {score[:5, 2]}")
