import dataclasses
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from event2vec.dataset import QuadraticReweightableDataset, ReweightableDataset
from event2vec.nontrainable import QuadraticFormNormalization, StandardScalerWrapper
from event2vec.util import EPS, ConstituentModel, tril_outer_product


class LearnedLLR(eqx.Module):
    @abstractmethod
    def llr_pred(
        self, observables: jax.Array, param_0: jax.Array, param_1: jax.Array
    ) -> jax.Array:
        raise NotImplementedError

    @abstractmethod
    def llr_prob(
        self, observables: jax.Array, param_0: jax.Array, param_1: jax.Array
    ) -> jax.Array | None:
        raise NotImplementedError


class RegularVector_LearnedLLR(LearnedLLR):
    """Event vector summary and parameter projection model."""

    event_summary_model: ConstituentModel
    param_projection_model: ConstituentModel

    def llr_pred(self, observables, param_0, param_1):
        summary = self.event_summary_model(observables)
        projection = self.param_projection_model(param_1) - self.param_projection_model(
            param_0
        )

        return summary @ projection

    def llr_prob(self, observables, param_0, param_1):
        return None


class CARLLinear_LearnedLLR(LearnedLLR):
    """Classic SBI model which predicts the linear dependence of the likelihood ratio on the parameters."""

    model: ConstituentModel
    normalization: QuadraticFormNormalization

    def llr_pred(self, observables, param_0, param_1):
        coef = self.model(observables)
        num = (coef @ param_1) / self.normalization(param_1)
        den = (coef @ param_0) / self.normalization(param_0)
        return jnp.log(jnp.maximum(num / den, EPS))

    def llr_prob(self, observables, param_0, param_1):
        return None


class CARLQuadratic_LearnedLLR(LearnedLLR):
    """Classic SBI model which predicts the quadratic dependence of the likelihood ratio on the parameters."""

    model: ConstituentModel
    normalization: QuadraticFormNormalization

    def llr_pred(self, observables, param_0, param_1):
        param_0_quad = tril_outer_product(param_0)
        param_1_quad = tril_outer_product(param_1)
        coef = self.model(observables)
        num = (coef @ param_1_quad) / self.normalization(param_1)
        den = (coef @ param_0_quad) / self.normalization(param_0)
        return jnp.log(jnp.maximum(num / den, EPS))

    def llr_prob(self, observables, param_0, param_1):
        return None


class CARLQuadraticForm_LearnedLLR(LearnedLLR):
    r"""SBI model which predicts a variable-rank quadratic form for the log-likelihood ratio dependence on the parameters.

    This is taking advantage of the expected structure of the event weight:
    $w=\theta^\top A \theta$, where A is a positive semi-definite matrix, and can therefore be decomposed as
    $A = B B^\top$, where B may be generally of rank less than the dimension of $\theta$.
    Then $w = | B^\top \theta |^2$.
    """

    model: ConstituentModel
    normalization: QuadraticFormNormalization
    """Overall normalization (non-trainable)"""
    rank: int
    "Rank of the learned quadratic form (to reshape model output)"

    def llr_pred(self, observables, param_0, param_1):
        coef = self.model(observables).reshape(-1, self.rank)
        pc1 = param_1 @ coef
        pc0 = param_0 @ coef
        llr_num = jnp.log(jnp.vecdot(pc1, pc1) / self.normalization(param_1))
        llr_den = jnp.log(jnp.vecdot(pc0, pc0) / self.normalization(param_0))
        return llr_num - llr_den

    def llr_prob(self, observables, param_0, param_1):
        return None


class ProbOneHotConstMag_LearnedLLR(LearnedLLR):
    """A model that predicts both the binwise log-likelihoods and the bin probabilities.

    This can then be used with a hardmax on the bin probabilities to assign events to bins"""

    binwise_ll_model: ConstituentModel
    bin_prob_model: ConstituentModel

    def llr_pred(self, observables, param_0, param_1):
        return self.binwise_ll_model(param_1) - self.binwise_ll_model(param_0)

    def llr_prob(self, observables, param_0, param_1):
        return self.bin_prob_model(observables)


@dataclasses.dataclass
class E2VMLPConfig:
    """Configuration for the event2vec MLP model."""

    summary_dim: int
    """Dimensionality of the summary vector."""
    hidden_size: int
    """Size of the hidden layers in the MLPs."""
    depth: int
    """Number of hidden layers in the MLPs."""
    standard_scaler: bool = False
    """Whether to standard scale the event observables."""

    def build(self, key: jax.Array, training_data: ReweightableDataset):
        """Build the model from the configuration."""
        key1, key2 = jax.random.split(key, 2)
        event_summary = eqx.nn.MLP(
            in_size=training_data.observable_dim,
            out_size=self.summary_dim,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            key=key1,
        )
        param_map = eqx.nn.MLP(
            in_size=training_data.parameter_dim,
            out_size=self.summary_dim,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            key=key2,
        )
        if self.standard_scaler:
            event_summary = StandardScalerWrapper.build(
                model=event_summary,
                data=training_data.observables,
            )
        return RegularVector_LearnedLLR(
            event_summary_model=event_summary, param_projection_model=param_map
        )


@dataclasses.dataclass
class CARLQuadraticFormMLPConfig:
    """Configuration for the CARL Quadratic Form model with MLP."""

    hidden_size: int
    """Size of the hidden layers in the MLPs."""
    depth: int
    """Number of hidden layers in the MLPs."""
    rank: int
    """Rank of the quadratic form predicted by the model."""
    standard_scaler: bool
    """Whether to standard scale the event observables."""

    def build(self, key: jax.Array, training_data: QuadraticReweightableDataset):
        """Build the model from the configuration."""
        model = eqx.nn.MLP(
            in_size=training_data.observable_dim,
            out_size=training_data.parameter_dim * self.rank,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            # final_activation=jax.nn.identity,
            key=key,
        )
        if self.standard_scaler:
            model = StandardScalerWrapper.build(
                model=model,
                data=training_data.observables,
            )
        return CARLQuadraticForm_LearnedLLR(
            model=model,
            normalization=training_data.normalization,
            rank=self.rank,
        )


@dataclasses.dataclass
class CARLMLPConfig:
    """Configuration for the classic CARL model with MLP."""

    hidden_size: int
    """Size of the hidden layers in the MLPs."""
    depth: int
    """Number of hidden layers in the MLPs."""
    quadratic: bool
    """Whether to include quadratic terms in the parameter dependence."""
    standard_scaler: bool = False
    """Whether to standard scale the event observables."""

    def build(self, key: jax.Array, training_data: QuadraticReweightableDataset):
        """Build the model from the configuration."""
        ncoef = training_data.parameter_dim
        cls = CARLQuadratic_LearnedLLR if self.quadratic else CARLLinear_LearnedLLR
        if self.quadratic:
            ncoef = ncoef * (ncoef + 1) // 2
        model = eqx.nn.MLP(
            in_size=training_data.observable_dim,
            out_size=ncoef,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            # final_activation=jax.nn.identity,
            key=key,
        )
        if self.standard_scaler:
            model = StandardScalerWrapper.build(
                model=model,
                data=training_data.observables,
            )
        return cls(model=model, normalization=training_data.normalization)
