from abc import abstractmethod
from typing import Protocol, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from event2vec.dataset import ReweightableDataset
from event2vec.model import AbstractLLR, VecDotLLR
from event2vec.prior import JointParameterPrior
from event2vec.shapes import LLRScalar

M = TypeVar("M", bound=AbstractLLR, contravariant=True)
D = TypeVar("D", bound=ReweightableDataset, contravariant=True)


class Loss(Protocol[M, D]):
    """Loss protocol for training models.

    Models and datasets are generic type variables to allow for more specific
    loss implementations that only work with certain model or dataset types.
    """

    @abstractmethod
    def __call__(self, model: M, data: D, *, key: PRNGKeyArray) -> jax.Array:
        """Compute the loss for the model on the given dataset."""
        ...


class LLRLoss(eqx.Module):
    """A loss function for training models."""

    @abstractmethod
    def __call__(
        self, model: AbstractLLR, data: ReweightableDataset, *, key: PRNGKeyArray
    ) -> jax.Array:
        """Compute the loss for the model on the given dataset."""
        ...


class BinwiseLoss(eqx.Module):
    """A loss for training models with binwise outputs."""

    @abstractmethod
    def __call__(
        self, model: VecDotLLR, data: ReweightableDataset, *, key: PRNGKeyArray
    ) -> jax.Array:
        """Compute the loss for the model on the given dataset."""
        ...


class ElementwiseLoss(eqx.Module):
    """An element-wise loss function for training models."""

    @abstractmethod
    def __call__(
        self, llr_pred: LLRScalar, target_label: Float[Array, ""]
    ) -> Float[Array, ""]:
        """Compute the element-wise loss for the model's learned log-likelihood ratio."""
        ...


class BinarySampledParamLoss(LLRLoss):
    """Samples a pair of parameters from a given prior for each event in the dataset
    and computes a binary classification loss based on the model's predicted log-likelihood ratio.
    """

    parameter_prior: JointParameterPrior
    """The prior over parameters to sample from during training."""
    continuous_labels: bool
    """Whether to use continuous labels (i.e. regression) instead of binary labels (i.e. classification)."""
    elementwise_loss: ElementwiseLoss

    def __call__(
        self, model: AbstractLLR, data: ReweightableDataset, *, key: PRNGKeyArray
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
        )  # shape: (batch,)

        weight_param_0 = data.weight(param_0)  # shape: (batch,)
        weight_param_1 = data.weight(param_1)  # shape: (batch,)

        if self.continuous_labels:
            sample_weight = (weight_param_0 + weight_param_1) / 2
            target_label = weight_param_1 / (weight_param_0 + weight_param_1)
        else:
            target_label = jax.random.bernoulli(label_key, p=0.5, shape=(len(data),))
            sample_weight = jnp.where(target_label == 0, weight_param_0, weight_param_1)

        loss = self.elementwise_loss(llr_pred, target_label)
        return (loss * sample_weight).mean() / sample_weight.mean()


class BinarySampledParamBinwiseLoss(BinwiseLoss):
    """A binwise version of the BinarySampledParamLoss."""

    parameter_prior: JointParameterPrior
    """The prior over parameters to sample from during training."""
    continuous_labels: bool
    """Whether to use continuous labels (i.e. regression) instead of binary labels (i.e. classification)."""
    elementwise_loss: ElementwiseLoss

    def __call__(
        self, model: VecDotLLR, data: ReweightableDataset, *, key: PRNGKeyArray
    ) -> jax.Array:
        """Compute the loss, sampling parameters from the prior."""
        prior_sample_key, label_key = jax.random.split(key, 2)
        param_0, param_1 = jax.vmap(self.parameter_prior.sample)(
            key=jax.random.split(prior_sample_key, len(data))
        )  # shapes: (batch, NParameters)

        bin_llr_0 = jax.vmap(model.param_projection)(param_0)  # shape: (batch, B)
        bin_llr_1 = jax.vmap(model.param_projection)(param_1)  # shape: (batch, B)
        bin_llr = bin_llr_1 - bin_llr_0  # shape: (batch, B)
        bin_prob = jax.vmap(model.event_summary)(data.observables)  # shape: (batch, B)

        weight_param_0 = data.weight(param_0)  # shape: (batch,)
        weight_param_1 = data.weight(param_1)  # shape: (batch,)

        if self.continuous_labels:
            sample_weight = (weight_param_0 + weight_param_1) / 2
            target_label = weight_param_1 / (weight_param_0 + weight_param_1)
        else:
            target_label = jax.random.bernoulli(label_key, p=0.5, shape=(len(data),))
            sample_weight = jnp.where(target_label == 0, weight_param_0, weight_param_1)

        # outer vmap over batch, inner over bins
        bin_loss = jax.vmap(jax.vmap(self.elementwise_loss, in_axes=(0, None)))(
            bin_llr, target_label
        )  # shape: (batch, B)
        assert bin_loss.shape == bin_prob.shape
        loss = (bin_loss * bin_prob).sum(axis=1)

        return (loss * sample_weight).mean() / sample_weight.mean()


class BCELoss(ElementwiseLoss):
    def __call__(
        self, llr_pred: LLRScalar, target_label: Float[Array, ""]
    ) -> Float[Array, ""]:
        return optax.losses.sigmoid_binary_cross_entropy(
            logits=llr_pred, labels=target_label
        )


class MSELoss(ElementwiseLoss):
    def __call__(
        self, llr_pred: LLRScalar, target_label: Float[Array, ""]
    ) -> Float[Array, ""]:
        return (jax.nn.sigmoid(llr_pred) - target_label) ** 2
