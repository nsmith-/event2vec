import dataclasses
from typing import TypeVar

import equinox as eqx
import jax
import optax
from rich.progress import (
    Progress,
    TextColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from event2vec.dataset import ReweightableDataset
from event2vec.loss import BinaryClassLoss
from event2vec.model import LearnedLLR

ModelT = TypeVar("ModelT", bound=LearnedLLR)


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
    loss_fn: BinaryClassLoss
    """Loss function to use for training."""

    def train(
        self, model: ModelT, data: ReweightableDataset, key: jax.Array
    ) -> tuple[ModelT, list[float], list[float]]:
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
    model: ModelT,
    data: ReweightableDataset,
    key: jax.Array,
) -> tuple[ModelT, list[float], list[float]]:
    key, subkey = jax.random.split(key)
    data_train, data_test = data.split(config.test_fraction, key=subkey)

    @eqx.filter_jit
    def make_step(
        model: ModelT,
        batch: ReweightableDataset,
        opt_state: optax.OptState,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, ModelT, optax.OptState]:
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
    pbar = Progress(
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("Remaining:"),
        TimeRemainingColumn(),
        TextColumn("Test loss: {task.fields[loss]:.4f}"),
    )
    with pbar as progress:
        epoch_task = progress.add_task("Training...", total=config.epochs, loss=0.0)
        for _ in range(config.epochs):
            tmp = []
            for i, batch in enumerate(data_train.iter_batch(config.batch_size)):
                key, subkey = jax.random.split(key)
                loss, model, opt_state = make_step(model, batch, opt_state, key=subkey)
                tmp.append(loss.item())
            train_loss_history.append(sum(tmp) / len(tmp))
            key, subkey = jax.random.split(key)
            test_loss = config.loss_fn(model, data_test, key=subkey).item()
            test_loss_history.append(test_loss)
            progress.update(epoch_task, advance=1, loss=test_loss)

        return model, train_loss_history, test_loss_history
