from abc import abstractmethod
from typing import Protocol
import jax
import equinox as eqx
import optax
import jax.numpy as jnp

from event2vec.dataset import ReweightableDataset
from event2vec.model import LearnedLLR
from event2vec.prior import JointParameterPrior


class Loss(Protocol):
    """A loss function for training models."""

    def __call__(
        self, model: LearnedLLR, data: ReweightableDataset, *, key: jax.Array
    ) -> jax.Array:
        """Compute the loss for the model on the given dataset."""
        ...


class BinaryClassLoss(eqx.Module):
    """Abstract class for loss functions."""

    parameter_prior: eqx.AbstractVar[JointParameterPrior]
    """The prior over parameters to sample from during training."""
    continuous_labels: eqx.AbstractVar[bool]
    """Whether to use continuous labels (i.e. regression) instead of binary labels (i.e. classification)."""

    def __call__(
        self, model: LearnedLLR, data: ReweightableDataset, *, key: jax.Array
    ) -> jax.Array:
        """Compute the loss, sampling parameters from the prior.

        TODO: This should not be batching the data internally, that should be done outside.
        """
        prior_sample_key, label_key = jax.random.split(key, 2)
        param_0, param_1 = jax.vmap(self.parameter_prior.sample)(
            key=jax.random.split(prior_sample_key, len(data))
        )  # shapes: (batch, NParameters)

        llr_pred = jax.vmap(model.llr_pred)(
            data.observables, param_0, param_1
        )  # shape: (batch, 1) or (batch, B)

        llr_prob = jax.vmap(model.llr_prob)(
            data.observables, param_0, param_1
        )  # None or shape: (batch, B)

        weight_param_0 = data.weight(param_0)  # shape: (batch, 1)
        weight_param_1 = data.weight(param_1)  # shape: (batch, 1)

        if self.continuous_labels:
            sample_weight = (weight_param_0 + weight_param_1) / 2
            target_label = weight_param_1 / (weight_param_0 + weight_param_1)
        else:
            target_label = jax.random.bernoulli(label_key, p=0.5, shape=(len(data),))
            sample_weight = jnp.where(target_label == 0, weight_param_0, weight_param_1)

        loss = self._elementwise_loss(llr_pred, target_label)

        if llr_prob is not None:
            loss = (loss * llr_prob).sum(axis=-1, keepdims=True)

        return (loss * sample_weight).mean() / sample_weight.mean()

    @abstractmethod
    def _elementwise_loss(
        self, llr_pred: jax.Array, target_label: jax.Array
    ) -> jax.Array:
        """Compute the loss for the model's learned log-likelihood ratio on the given dataset.

        Does not reduce the loss over the dataset, this is done in the `__call__` method.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class BCELoss(BinaryClassLoss):
    parameter_prior: JointParameterPrior
    "The prior over parameters to sample from during training."
    continuous_labels: bool = False
    "By default, use binary labels for BCE loss. Continuous labels can be used too."

    def _elementwise_loss(self, llr_pred, target_label):
        return optax.losses.sigmoid_binary_cross_entropy(
            logits=llr_pred, labels=target_label
        )


class MSELoss(BinaryClassLoss):
    parameter_prior: JointParameterPrior
    "The prior over parameters to sample from during training."
    continuous_labels: bool = True
    "For MSE loss, binary labels don't make much sense, so this defaults to True."

    def _elementwise_loss(self, llr_pred, target_label):
        return (jax.nn.sigmoid(llr_pred) - target_label) ** 2
