import dataclasses
import jax
from event2vec.dataset import ReweightableDataset
import equinox as eqx
from jaxtyping import PRNGKeyArray
from event2vec.model import AbstractLLR
from event2vec.nontrainable import StandardScalerWrapper
from event2vec.shapes import LLRScalar, LLRVec, ObsVec, ParamVec, ProbVec


from collections.abc import Callable


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
