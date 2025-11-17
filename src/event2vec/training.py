import dataclasses
from typing import Generic, TypeVar

import equinox as eqx
import jax
import optax
from rich.progress import TextColumn

from event2vec.dataset import ReweightableDataset
from event2vec.loss import Loss
from event2vec.model import LearnedLLR
from event2vec.utils import partition_trainable_and_static
from event2vec.util import standard_pbar

ModelT = TypeVar("ModelT", bound=LearnedLLR, contravariant=True)
DatasetT = TypeVar("DatasetT", bound=ReweightableDataset, contravariant=True)


@dataclasses.dataclass
class MetricsHistory:
    """Class to store training metrics history."""

    train_loss: list[float]
    """List of training loss values per epoch."""
    test_loss: list[float]
    """List of testing loss values per epoch."""


@dataclasses.dataclass
class TrainingConfig(Generic[ModelT, DatasetT]):
    """Configuration for the training process."""

    test_fraction: float
    """Fraction of the dataset to use for testing."""
    batch_size: int
    """Batch size for training."""
    learning_rate: float
    """Learning rate for the optimizer."""
    epochs: int
    """Number of epochs to train for."""
    loss_fn: Loss[ModelT, DatasetT]
    """Loss function to use for training."""

    def train(
        self, model: ModelT, data: DatasetT, key: jax.Array
    ) -> tuple[ModelT, list[float], list[float]]:
        """Train the model using the specified configuration.

        TODO: replace lists with a MetricsHistory object.
        """
        return _train(
            config=self,
            model=model,
            data=data,
            key=key,
        )


def _train(
    config: TrainingConfig[ModelT, DatasetT],
    *,
    model: ModelT,
    data: DatasetT,
    key: jax.Array,
) -> tuple[ModelT, list[float], list[float]]:
    key, subkey = jax.random.split(key)
    data_train, data_test = data.split(config.test_fraction, key=subkey)
    diff_model, static_model = partition_trainable_and_static(model)

    @eqx.filter_jit
    def make_step(
        diff_model,
        batch: DatasetT,
        opt_state: optax.OptState,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, ModelT, optax.OptState]:
        @eqx.filter_value_and_grad
        def loss_grad(diff_model, batch, *, key):
            model = eqx.combine(diff_model, static_model)
            return config.loss_fn(model, batch, key=key)

        loss, grads = loss_grad(diff_model, batch, key=key)
        updates, opt_state = optim.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        return loss, diff_model, opt_state

    optim = optax.adamw(config.learning_rate, b1=0.9)
    opt_state = optim.init(diff_model)

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    pbar = standard_pbar(TextColumn("Test loss: {task.fields[loss]:.4f}"))
    with pbar as progress:
        epoch_task = progress.add_task("Training...", total=config.epochs, loss=0.0)
        for _ in range(config.epochs):
            tmp = []
            for i, batch in enumerate(data_train.iter_batch(config.batch_size)):
                key, subkey = jax.random.split(key)
                loss, diff_model, opt_state = make_step(
                    diff_model, batch, opt_state, key=subkey
                )
                tmp.append(loss.item())
            train_loss_history.append(sum(tmp) / len(tmp))
            key, subkey = jax.random.split(key)
            model = eqx.combine(diff_model, static_model)
            test_loss = eqx.filter_jit(config.loss_fn)(
                model, data_test, key=subkey
            ).item()
            test_loss_history.append(test_loss)
            progress.update(epoch_task, advance=1, loss=test_loss)

    return model, train_loss_history, test_loss_history
