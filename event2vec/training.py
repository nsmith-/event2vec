import dataclasses

import equinox as eqx
import jax
import optax
import rich.progress

from event2vec.dataset import ReweightableDataset
from event2vec.loss import Loss
from event2vec.model import LearnedLLR


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
    loss_fn: Loss
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
    key, subkey = jax.random.split(key)
    data_train, data_test = data.split(config.test_fraction, key=subkey)

    @eqx.filter_jit
    def make_step(
        model: LearnedLLR,
        batch: ReweightableDataset,
        opt_state: optax.OptState,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, LearnedLLR, optax.OptState]:
        loss, grads = eqx.filter_value_and_grad(config.loss_fn)(model, batch, key=key)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adamw(config.learning_rate, b1=0.9)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    for _ in rich.progress.track(range(config.epochs), description="Training"):
        tmp = []
        for i, batch in enumerate(data_train.iter_batch(config.batch_size)):
            key, subkey = jax.random.split(key)
            loss, model, opt_state = make_step(model, batch, opt_state, key=subkey)
            tmp.append(loss.item())
        train_loss_history.append(sum(tmp) / len(tmp))
        key, subkey = jax.random.split(key)
        test_loss_history.append(config.loss_fn(model, data_test, key=subkey).item())

    return model, train_loss_history, test_loss_history
