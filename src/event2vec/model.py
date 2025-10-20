import dataclasses
from abc import abstractmethod
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp

from event2vec.util import EPS, tril_outer_product


class ConstituentModel(Protocol):
    def __call__(self, *args, **kwargs) -> jax.Array: ...


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


class CARL_LearnedLLR(LearnedLLR):
    """Classic SBI model which predicts the linear dependence of the likelihood ratio on the parameters."""

    model: ConstituentModel

    def llr_pred(self, observables, param_0, param_1):
        coef = self.model(observables)
        return jnp.log(jnp.maximum((coef @ param_1) / (coef @ param_0), EPS))

    def llr_prob(self, observables, param_0, param_1):
        return None


class CARL_LearnedLLR_Quadratic(LearnedLLR):
    """Classic SBI model which predicts the quadratic dependence of the likelihood ratio on the parameters."""

    model: ConstituentModel

    def llr_pred(self, observables, param_0, param_1):
        param_0_quad = tril_outer_product(param_0)
        param_1_quad = tril_outer_product(param_1)
        coef = self.model(observables)
        return jnp.log(jnp.maximum((coef @ param_1_quad) / (coef @ param_0_quad), EPS))

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

    event_dim: int
    """Dimensionality of the event observables."""
    param_dim: int
    """Dimensionality of the parameters."""
    summary_dim: int
    """Dimensionality of the summary vector."""
    hidden_size: int
    """Size of the hidden layers in the MLPs."""
    depth: int
    """Number of hidden layers in the MLPs."""

    def build(self, key: jax.Array):
        """Build the model from the configuration."""
        key1, key2 = jax.random.split(key, 2)
        event_summary = eqx.nn.MLP(
            in_size=self.event_dim,
            out_size=self.summary_dim,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            key=key1,
        )
        param_map = eqx.nn.MLP(
            in_size=self.param_dim,
            out_size=self.summary_dim,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            key=key2,
        )
        return RegularVector_LearnedLLR(
            event_summary_model=event_summary, param_projection_model=param_map
        )


@dataclasses.dataclass
class CARLMLPConfig:
    """Configuration for the classic CARL model with MLP."""

    event_dim: int
    """Dimensionality of the event observables."""
    param_dim: int
    """Dimensionality of the parameters."""
    hidden_size: int
    """Size of the hidden layers in the MLPs."""
    depth: int
    """Number of hidden layers in the MLPs."""
    quadratic: bool = False
    """Whether to include quadratic terms in the parameter dependence."""

    def build(self, key: jax.Array):
        """Build the model from the configuration."""
        cls = CARL_LearnedLLR_Quadratic if self.quadratic else CARL_LearnedLLR
        npar = (
            self.param_dim * (self.param_dim + 1) // 2
            if self.quadratic
            else self.param_dim
        )
        model = eqx.nn.MLP(
            in_size=self.event_dim,
            out_size=npar,
            width_size=self.hidden_size,
            depth=self.depth,
            activation=jax.nn.leaky_relu,
            # final_activation=jax.nn.identity,
            key=key,
        )
        return cls(model=model)
