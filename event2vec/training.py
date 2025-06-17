import dataclasses
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rich.progress

from event2vec.dataset import ReweightableDataset
from event2vec.model import LearnedLLR
from event2vec.prior import ParameterPrior


def loss_mse(
    model: LearnedLLR, data: ReweightableDataset, param_1: jax.Array
) -> jax.Array:
    """Compute the training loss for the model on the given dataset using Mean Squared Error (MSE).

    We use the generation point as the denominator hypothesis, param_1 as the alternate

    TODO: Do we need ReweightableDataset here or is DatasetWithLikelihood sufficient?
    """
    llr_pred = jax.vmap(model.log_likelihood_ratio)(
        data.observables, data.gen_parameters, param_1
    )
    llr_true = jnp.log(data.likelihood(param_1)) - jnp.log(
        data.likelihood(data.gen_parameters)
    )
    return jnp.mean((llr_pred - llr_true) ** 2)


def loss_bce(
    model: LearnedLLR, data: ReweightableDataset, param_1: jax.Array
) -> jax.Array:
    """Compute the training loss for the model on the given dataset using Binary Cross Entropy (BCE).

    We use the generation point as the denominator hypothesis, param_1 as the alternate
    """
    llr_pred = jax.vmap(model.log_likelihood_ratio)(
        data.observables, data.gen_parameters, param_1
    )
    # We flip a coin to label the events, it just swaps
    # the numerator and denominator in the likelihood ratio.
    # TODO: this is broken
    label = jnp.zeros_like(llr_pred).at[::2].set(1.0)
    pred = llr_pred.at[::2].multiply(-1.0)
    return optax.losses.sigmoid_binary_cross_entropy(pred, label).mean()


# TODO: weighted BCE

_LMAP = {
    "mse": loss_mse,
    "bce": loss_bce,
}


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for the training process."""

    test_fraction: float
    """Fraction of the dataset to use for testing."""
    batch_size: int
    """Batch size for training."""
    learning_rate: float
    """Learning rate for the optimizer."""
    epochs: int
    """Number of epochs to train for."""
    param_prior: ParameterPrior
    """Prior distribution for the parameters used in training."""
    loss_fn: Literal["mse", "bce"]
    """Loss function to use for training."""

    def train(self, model: LearnedLLR, data: ReweightableDataset, key: jax.Array):
        """Train the model using the specified configuration."""
        return _train(
            config=self,
            model=model,
            data=data,
            key=key,
        )


def _train(
    config: TrainingConfig,
    *,
    model: LearnedLLR,
    data: ReweightableDataset,
    key: jax.Array,
):
    loss_fn = _LMAP[config.loss_fn]

    key, subkey = jax.random.split(key)
    data_train, data_test = data.split(config.test_fraction, key=subkey)

    def sample_prior_like(batch: ReweightableDataset, *, key: jax.Array) -> jax.Array:
        """Sample parameters from the prior for each event in the batch."""
        return jax.vmap(config.param_prior.sample)(jax.random.split(key, len(batch)))

    @eqx.filter_jit
    def make_step(
        model: LearnedLLR,
        batch: ReweightableDataset,
        opt_state: optax.OptState,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, LearnedLLR, optax.OptState]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model,
            batch,
            sample_prior_like(batch, key=key),
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adamw(config.learning_rate, b1=0.9)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    for epoch in rich.progress.track(range(config.epochs), description="Training"):
        tmp = []
        for i, batch in enumerate(data_train.iter_batch(config.batch_size)):
            key, subkey = jax.random.split(key)
            loss, model, opt_state = make_step(model, batch, opt_state, key=subkey)
            tmp.append(loss.item())
        train_loss_history.append(sum(tmp) / len(tmp))
        key, subkey = jax.random.split(key)
        test_loss_history.append(
            loss_fn(model, data_test, sample_prior_like(data_test, key=subkey)).item()
        )

    return model, train_loss_history, test_loss_history
