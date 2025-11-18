import dataclasses
from abc import abstractmethod
from typing import Generic, TypeVar
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from event2vec.dataset import QuadraticReweightableDataset, ReweightableDataset
from event2vec.models._psd_matrix_models import PSDMatrixModel
from event2vec.nontrainable import QuadraticFormNormalization, StandardScalerWrapper
from event2vec.shapes import (
    LLRScalar,
    LLRVec,
    ObsVec,
    ParamQuadVec,
    ParamVec,
    ProbVec,
    PSDMatrix,
)
from event2vec.util import EPS, tril_outer_product


class AbstractLLR(eqx.Module):
    """Abstract learned log-likelihood ratio model.

    This is a model that can predict the log-likelihood ratio between two parameter points,
    given the event observables. No assumptions are made about the internal structure of the model.
    """

    @abstractmethod
    def llr_pred(
        self, observables: ObsVec, param_0: ParamVec, param_1: ParamVec
    ) -> LLRScalar:
        raise NotImplementedError


class VecDotLLR(AbstractLLR):
    r"""A model that predicts a latent vector representation for both the observables and parameters.

    The resulting log-likelihood ratio is computed as the dot product of these two vectors:
    $$ \hat{\ell}(x, \theta_0, \theta_1) = s(x) \cdot (\pi(\theta_1) - \pi(\theta_0)) $$

    The event summary model $s(x)$ maps observables, and the parameter projection
    model $\pi(\theta)$ maps parameters.

    If this model is trained with a BinwiseLoss, then the event summary regresses to bin probabilities,
    and the parameter projection regresses to bin average log-likelihood ratios.
    Note that in this case the output of the event summary should be a probability vector
    (e.g. using a softmax final activation)
    """

    event_summary: Callable[[ObsVec], ProbVec]
    param_projection: Callable[[ParamVec], LLRVec]

    def llr_pred(
        self, observables: ObsVec, param_0: ParamVec, param_1: ParamVec
    ) -> LLRScalar:
        bin_probs = self.event_summary(observables)
        binwise_llr_1 = self.param_projection(param_1)
        binwise_llr_0 = self.param_projection(param_0)
        binwise_llr = binwise_llr_1 - binwise_llr_0
        return bin_probs @ binwise_llr


class AbstractPSDMatrixLLR(AbstractLLR):
    """A model that predicts a positive semi-definite matrix for the log-likelihood ratio dependence on the parameters.

    This is useful for models where the log-likelihood ratio has a quadratic dependence on the parameters.
    It is abstract to allow for different implementations of the PSD matrix prediction.
    """

    @abstractmethod
    def psd_matrix(self, observables: ObsVec) -> PSDMatrix:
        raise NotImplementedError


class CARLLinearLLR(AbstractLLR):
    """Classic SBI model which predicts the linear dependence of the likelihood ratio on the parameters."""

    model: Callable[[ObsVec], ParamVec]
    """Mapping from observables to linear coefficients"""
    normalization: QuadraticFormNormalization

    def llr_pred(
        self, observables: ObsVec, param_0: ParamVec, param_1: ParamVec
    ) -> LLRScalar:
        coef = self.model(observables)
        num = (coef @ param_1) / self.normalization(param_1)
        den = (coef @ param_0) / self.normalization(param_0)
        return jnp.log(jnp.maximum(num / den, EPS))


class CARLQuadLLR(AbstractLLR):
    """Classic SBI model which predicts the quadratic dependence of the likelihood ratio on the parameters.

    Without any further structure, this model predicts the full quadratic form and might not guarantee positive semi-definiteness.
    This can be addressed by using the CARLQuadraticForm_LearnedLLR model instead.
    """

    model: Callable[[ObsVec], ParamQuadVec]
    """Mapping from observables to quadratic coefficients"""
    normalization: QuadraticFormNormalization

    def llr_pred(
        self, observables: ObsVec, param_0: ParamVec, param_1: ParamVec
    ) -> LLRScalar:
        param_0_quad = tril_outer_product(param_0)
        param_1_quad = tril_outer_product(param_1)
        coef = self.model(observables)
        num = (coef @ param_1_quad) / self.normalization(param_1)
        den = (coef @ param_0_quad) / self.normalization(param_0)
        return jnp.log(jnp.maximum(num / den, EPS))


class CARLPSDMatrixLLR(AbstractPSDMatrixLLR):
    r"""SBI model which predicts a variable-rank quadratic form for the log-likelihood ratio dependence on the parameters.

    This is taking advantage of the expected structure of the event weight:
    $w=\theta^\top A \theta$, where A is a positive semi-definite matrix, and can therefore be decomposed as
    $A = B B^\top$, where B may be generally of rank less than the dimension of $\theta$.
    Then $w = | B^\top \theta |^2$.

    TODO: this is overlapping with PSDMatrixLLR using the At_A_Model; consider merging these two classes.
    """

    model: Callable[[ObsVec], Float[Array, " P*{self.rank}"]]
    """Mapping from observables to low-rank estimate of quadratic coefficients"""
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

    def psd_matrix(self, observables: ParamVec) -> PSDMatrix:
        coef = self.model(observables).reshape(-1, self.rank)
        return coef @ coef.T


MatrixT = TypeVar("MatrixT", bound=PSDMatrixModel, covariant=True)


class PSDMatrixLLR(AbstractPSDMatrixLLR, Generic[MatrixT]):
    psd_matrix_model: MatrixT
    normalization: QuadraticFormNormalization

    def llr_pred(
        self, observables: ObsVec, param_0: ParamVec, param_1: ParamVec
    ) -> LLRScalar:
        psd_matrix = self.psd_matrix_model(observables)
        l0 = jnp.vecdot((psd_matrix @ param_0), param_0) / self.normalization(param_0)
        l1 = jnp.vecdot((psd_matrix @ param_1), param_1) / self.normalization(param_1)
        return jnp.log(l1) - jnp.log(l0)

    def psd_matrix(self, observables: ObsVec) -> PSDMatrix:
        return self.psd_matrix_model(observables)


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
    bin_probabilities: bool = False
    """Whether to use a softmax final activation for the event summary to get bin probabilities."""

    def build(self, key: PRNGKeyArray, training_data: ReweightableDataset):
        """Build the model from the configuration."""
        key1, key2 = jax.random.split(key, 2)
        event_summary: Callable[[ObsVec], ParamVec] = eqx.nn.MLP(
            in_size=training_data.observable_dim,
            out_size=self.summary_dim,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            # https://github.com/patrick-kidger/equinox/issues/1147
            # final_activation=(
            #     jax.nn.softmax if self.bin_probabilities else jax.nn.identity
            # ),
            key=key1,
        )
        if self.bin_probabilities:
            # Manually add softmax final activation due to Equinox issue
            old_event_summary = event_summary

            def event_summary_with_softmax(x: ObsVec) -> ProbVec:
                return jax.nn.softmax(old_event_summary(x))

            event_summary = event_summary_with_softmax
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
        return VecDotLLR(event_summary=event_summary, param_projection=param_map)


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

    def build(self, key: PRNGKeyArray, training_data: QuadraticReweightableDataset):
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
        return CARLPSDMatrixLLR(
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

    def build(self, key: PRNGKeyArray, training_data: QuadraticReweightableDataset):
        """Build the model from the configuration."""
        ncoef = training_data.parameter_dim
        cls = CARLQuadLLR if self.quadratic else CARLLinearLLR
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
