import dataclasses
import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray
from event2vec.dataset import QuadraticReweightableDataset
from event2vec.model import AbstractLLR, AbstractPSDMatrixLLR
from event2vec.nontrainable import QuadraticFormNormalization, StandardScalerWrapper
from event2vec.shapes import LLRScalar, ObsVec, PSDMatrix, ParamQuadVec, ParamVec


import jax.numpy as jnp


from collections.abc import Callable

from event2vec.util import EPS, tril_outer_product


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
