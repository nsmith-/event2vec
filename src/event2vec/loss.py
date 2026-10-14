from abc import abstractmethod
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from event2vec.dataset import ReweightableDataset
from event2vec.model import AbstractLLR
from event2vec.models.vecdot import VecDotLLR
from event2vec.prior import JointParameterPrior
from event2vec.shapes import LLRScalar


class Loss[ModelT: AbstractLLR, DatasetT: ReweightableDataset](Protocol):
    """Loss protocol for training models.

    Models and datasets are generic type variables to allow for more specific
    loss implementations that only work with certain model or dataset types.
    """

    @abstractmethod
    def __call__(
        self, model: ModelT, data: DatasetT, *, key: PRNGKeyArray
    ) -> jax.Array:
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
    r"""Sample a pair of parameters from a given prior for each event in the dataset
    and compute a binary classification loss based on the model's predicted log-likelihood ratio.

    With continuous labels and the BCE elementwise loss this is ALICE. With a positive
    ``score_weight`` it becomes ALICES. :math:`\partial \hat\ell / \partial \theta_1`
    and :math:`-\partial \hat\ell / \partial \theta_0` are additionally regressed on the exact
    joint score :math:`\nabla_\theta \log w(x, z | \theta)` at the two sampled points, obtained
    from the dataset's likelihood by automatic differentiation.
    """

    parameter_prior: JointParameterPrior
    """The prior over parameters to sample from during training."""
    continuous_labels: bool
    """Whether to use continuous labels (i.e. regression) instead of binary labels (i.e. classification)."""
    elementwise_loss: ElementwiseLoss
    score_weight: float = 0.0
    r"""Weight :math:`\alpha` of the score-matching term (ALICES); 0 disables it.

    When the labels sit near 0.5 (e.g. a rate-neutral interference and a symmetric prior) the
    cross-entropy gradient is much smaller than the score-term gradient, so values of order 1 let
    the score term dominate; 0.1 or smaller is proabbly safer.
    """

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
        loss = (loss * sample_weight).mean() / sample_weight.mean()
        if self.score_weight > 0:
            loss = loss + self.score_weight * self._score_matching(
                model, data, param_0, param_1, weight_param_0, weight_param_1
            )
        return loss

    @staticmethod
    def _score_matching(
        model: AbstractLLR,
        data: ReweightableDataset,
        param_0: Float[Array, "batch P"],
        param_1: Float[Array, "batch P"],
        weight_param_0: Float[Array, " batch"],
        weight_param_1: Float[Array, " batch"],
    ) -> jax.Array:
        """Mean squared difference between the model's and the exact score at both sampled points,
        each weighted by the event weight at that point (i.e. evaluated under its distribution)."""

        def log_weight_sum(params: Float[Array, "batch P"]) -> jax.Array:
            # event i depends only on row i, so the gradient is the per-event score
            return jnp.sum(jnp.log(data.weight(params)))

        score_0 = jax.grad(log_weight_sum)(param_0)  # shape: (batch, P)
        score_1 = jax.grad(log_weight_sum)(param_1)
        score_pred_1 = jax.vmap(jax.grad(model.llr_pred, argnums=2))(
            data.observables, param_0, param_1
        )
        score_pred_0 = -jax.vmap(jax.grad(model.llr_pred, argnums=1))(
            data.observables, param_0, param_1
        )
        sq_0 = jnp.sum((score_pred_0 - score_0) ** 2, axis=1)
        sq_1 = jnp.sum((score_pred_1 - score_1) ** 2, axis=1)
        return 0.5 * (
            (sq_0 * weight_param_0).mean() / weight_param_0.mean()
            + (sq_1 * weight_param_1).mean() / weight_param_1.mean()
        )


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
