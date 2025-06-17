from abc import abstractmethod
import dataclasses

import equinox as eqx
import jax


class LearnedLLR(eqx.Module):
    @abstractmethod
    def log_likelihood_ratio(
        self, observables: jax.Array, param_0: jax.Array, param_1: jax.Array
    ) -> jax.Array:
        raise NotImplementedError()


class E2VMLP(LearnedLLR):
    """Event2Vec MLP model for learning log-likelihood ratios."""

    event_summary: eqx.nn.MLP
    param_map: eqx.nn.MLP

    def log_likelihood_ratio(self, observables, param_0, param_1):
        summary = self.event_summary(observables)
        projection = self.param_map(param_1) - self.param_map(param_0)
        return summary @ projection


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
        return E2VMLP(event_summary, param_map)
