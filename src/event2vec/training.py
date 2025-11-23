import dataclasses

import equinox as eqx
import jax
import optax
from jaxtyping import PRNGKeyArray
from rich.progress import TextColumn

from event2vec.dataset import ReweightableDataset
from event2vec.loss import Loss
from event2vec.model import AbstractLLR
from event2vec.util import standard_pbar
from event2vec.utils import partition_trainable_and_static


@dataclasses.dataclass
class MetricsHistory:
    """Class to store training metrics history."""

    train_loss: list[float]
    """List of training loss values per epoch."""
    val_loss: list[float]
    """List of validation loss values per epoch."""
    test_loss: float | None
    """Test loss value (single evaluation at end of training)."""


@dataclasses.dataclass
class TrainingConfig[ModelT: AbstractLLR, DatasetT: ReweightableDataset]:
    """Configuration for the training process."""

    train_fraction: float
    """Fraction of the dataset to use for training."""
    val_fraction: float
    """Fraction of the dataset to use for validation."""
    batch_size: int
    """Batch size for training."""
    learning_rate: float
    """Learning rate for the optimizer."""
    epochs: int
    """Number of epochs to train for."""
    loss_fn: Loss[ModelT, DatasetT]
    """Loss function to use for training."""

    @property
    def test_fraction(self) -> float:
        """Fraction of the dataset to use for testing (inferred from train and val fractions)."""
        return 1.0 - self.train_fraction - self.val_fraction

    def __post_init__(self):
        """Validate that the fractions are valid."""
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError(
                f"train_fraction must be between 0 and 1, got {self.train_fraction}"
            )
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError(
                f"val_fraction must be between 0 and 1, got {self.val_fraction}"
            )
        if not (0.0 < self.test_fraction < 1.0):
            raise ValueError(
                f"test_fraction (inferred as 1 - train_fraction - val_fraction) must be between 0 and 1, "
                f"got {self.test_fraction} (train_fraction={self.train_fraction}, val_fraction={self.val_fraction})"
            )


def train[ModelT: AbstractLLR, DatasetT: ReweightableDataset](
    config: TrainingConfig[ModelT, DatasetT],
    *,
    model: ModelT,
    data_train: DatasetT,
    data_val: DatasetT,
    key: PRNGKeyArray,
) -> tuple[ModelT, list[float], list[float]]:
    """Use a training configuration to train a model on pre-split datasets.

    Args:
        config: Training configuration
        model: Model to train
        data_train: Training dataset
        data_val: Validation dataset
        key: Random key

    Returns:
        Tuple of (trained model, training loss history, validation loss history)

    TODO: replace output lists with a MetricsHistory object.
    """
    diff_model, static_model = partition_trainable_and_static(model)

    @eqx.filter_jit
    def make_step(
        diff_model,
        batch: DatasetT,
        opt_state: optax.OptState,
        *,
        key: PRNGKeyArray,
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
    val_loss_history: list[float] = []
    pbar = standard_pbar(TextColumn("Val loss: {task.fields[loss]:.4f}"))
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
            val_loss = eqx.filter_jit(config.loss_fn)(
                model, data_val, key=subkey
            ).item()
            val_loss_history.append(val_loss)
            progress.update(epoch_task, advance=1, loss=val_loss)

    return model, train_loss_history, val_loss_history
