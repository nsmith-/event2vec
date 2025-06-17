from abc import abstractmethod

import equinox as eqx
import jax


class LearnedLLR(eqx.Module):
    @abstractmethod
    def log_likelihood_ratio(
        self, observables: jax.Array, param_0: jax.Array, param_1: jax.Array
    ) -> jax.Array:
        raise NotImplementedError()


class FactorizedParameterizedLLR(LearnedLLR):
    event_summary: eqx.nn.MLP
    param_map: eqx.nn.MLP

    def log_likelihood_ratio(self, observables, param_0, param_1):
        summary = self.event_summary(observables)
        projection = self.param_map(param_1) - self.param_map(param_0)
        return summary @ projection


def build_model(
    event_dim: int,
    param_dim: int,
    summary_dim: int,
    hidden_size: int,
    depth: int,
    key: jax.Array,
):
    key1, key2 = jax.random.split(key, 2)
    event_summary = eqx.nn.MLP(
        in_size=event_dim,
        out_size=summary_dim,
        width_size=hidden_size,
        depth=depth,
        activation=jax.nn.leaky_relu,
        key=key1,
    )
    param_map = eqx.nn.MLP(
        in_size=param_dim,
        out_size=summary_dim,
        width_size=hidden_size,
        depth=depth,
        activation=jax.nn.leaky_relu,
        key=key2,
    )
    return FactorizedParameterizedLLR(event_summary, param_map)
