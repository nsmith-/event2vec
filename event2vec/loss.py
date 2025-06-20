from abc import abstractmethod
import jax
import equinox as eqx
import jax.numpy as jnp
import optax

from event2vec.dataset import ReweightableDataset
from event2vec.model import LearnedLLR
from event2vec.prior import JointParameterPrior


class Loss(eqx.Module):
    """Abstract class for loss functions."""

    parameter_prior: JointParameterPrior

    def __call__(
        self, model: LearnedLLR, data: ReweightableDataset, *, key: jax.Array
    ) -> jax.Array:
        """Compute the loss, sampling parameters from the prior."""
        param_0, param_1 = jax.vmap(self.parameter_prior.sample)(
            key=jax.random.split(key, len(data))
        )
        llr_pred = jax.vmap(model.log_likelihood_ratio)(
            data.observables, param_0, param_1
        )
        # TODO: use these weights appropriately
        weight_param_1 = jax.vmap(data.weight)(param_1)
        weight_param_0 = jax.vmap(data.weight)(param_0)
        loss = self._elementwise_loss(llr_pred, data, param_0, param_1)
        # reduce with the weights
        return loss.mean()

    @abstractmethod
    def _elementwise_loss(
        self,
        llr_pred: jax.Array,
        data: ReweightableDataset,
        param_0: jax.Array,
        param_1: jax.Array,
    ) -> jax.Array:
        """Compute the loss for the model's learned log-likelihood ratio on the given dataset.

        Does not reduce the loss over the dataset, this is done in the `__call__` method.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class LLRMSELoss(Loss):
    """Mean-squared loss for a log likelihood ratio model."""

    def _elementwise_loss(
        self,
        llr_pred: jax.Array,
        data: ReweightableDataset,
        param_0: jax.Array,
        param_1: jax.Array,
    ) -> jax.Array:
        llr_true = jnp.log(data.likelihood(param_1)) - jnp.log(data.likelihood(param_0))
        return (llr_pred - llr_true) ** 2


class LRMSELoss(Loss):
    """Mean-squared loss for a likelihood ratio model."""

    def _elementwise_loss(
        self,
        llr_pred: jax.Array,
        data: ReweightableDataset,
        param_0: jax.Array,
        param_1: jax.Array,
    ) -> jax.Array:
        lr_true = data.likelihood(param_1) / data.likelihood(param_0)
        return (jnp.exp(llr_pred) - lr_true) ** 2


class LLRBCELoss(Loss):
    """Binary cross-entropy loss for a log likelihood ratio model."""

    def _elementwise_loss(
        self,
        llr_pred: jax.Array,
        data: ReweightableDataset,
        param_0: jax.Array,
        param_1: jax.Array,
    ) -> jax.Array:
        # We 'flip a coin' to label the events, it just swaps
        # the numerator and denominator in the likelihood ratio.
        label = jnp.zeros_like(llr_pred).at[::2].set(1.0)
        pred = llr_pred.at[::2].multiply(-1.0)
        return optax.losses.sigmoid_binary_cross_entropy(pred, label)
