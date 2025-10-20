from abc import abstractmethod
from typing import Self
import equinox as eqx
import jax


class Dataset(eqx.Module):
    """Abstract dataset

    For any arrays where the leading dimension matches the length of observables (i.e. number of events),
    these will be treated as batchable arrays and will be indexed when the dataset is indexed.
    """

    observables: jax.Array
    """Observables of the dataset, e.g. event kinematics, etc."""

    def __getitem__(self, key) -> Self:
        batchable, nonbatchable = eqx.partition(
            self,
            lambda leaf: isinstance(leaf, jax.Array)
            and leaf.ndim > 0
            and leaf.shape[0] == len(self),
        )
        applied = jax.tree_util.tree_map(lambda x: x[key], batchable)
        return eqx.combine(applied, nonbatchable)

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

    @property
    @abstractmethod
    def observable_dim(self) -> int:
        """Dimensionality of the observables."""
        raise NotImplementedError("This property should be implemented by subclasses.")

    @property
    @abstractmethod
    def parameter_dim(self) -> int:
        """Dimensionality of the parameters."""
        raise NotImplementedError("This property should be implemented by subclasses.")

    @abstractmethod
    def likelihood(self, param: jax.Array) -> jax.Array:
        """Compute the likelihood of the dataset w.r.t. the given parameter vector.

        This may be defined only up to a multiplicative constant
        """
        msg = (
            "This method should be implemented by subclasses of DatasetWithLikelihood."
        )
        raise NotImplementedError(msg)


class ReweightableDataset(DatasetWithLikelihood):
    """A dataset that can be reweighted by a parameter vector."""

    gen_parameters: jax.Array
    """The parameters used to sample this event"""

    def weight(self, param: jax.Array) -> jax.Array:
        """Compute the weight of the dataset w.r.t. the given parameter vector.

        This is the likelihood ratio for the parameter point with respect to the generated point
        """
        # denom shouldn't be zero by construction, since the event was sampled
        # from the distribution defined by gen_parameters
        denom = self.likelihood(self.gen_parameters)
        num = self.likelihood(param)
        return num / denom
