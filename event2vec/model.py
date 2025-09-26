import dataclasses
from abc import abstractmethod
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp


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


class MadMiner_LearnedLLR(LearnedLLR):
    model: ConstituentModel
    biasModel: ConstituentModel

    def llr_pred(self, observables, param_0, param_1):
        log_num = jnp.log(
            self.model(observables) @ param_1 + self.biasModel(observables)
        )
        log_den = jnp.log(
            self.model(observables) @ param_0 + self.biasModel(observables)
        )
        return log_num - log_den

    def llr_prob(self, observables, param_0, param_1):
        return None


class ProbOneHotConstMag_LearnedLLR(LearnedLLR):
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

    def build(self, key: jax.Array) -> LearnedLLR:
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
