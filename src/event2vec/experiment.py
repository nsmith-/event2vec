import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, Self

import jax
from jaxtyping import PRNGKeyArray

from event2vec.dataset import ReweightableDataset
from event2vec.model import AbstractLLR
from event2vec.training import TrainingConfig, train


class DatasetFactory[DatasetT: ReweightableDataset](Protocol):
    def __call__(self, *, key: PRNGKeyArray) -> DatasetT:
        """Create a dataset given a random key."""
        ...


class ModelBuilder[ModelT: AbstractLLR, DatasetT: ReweightableDataset](Protocol):
    def build(self, *, key: PRNGKeyArray, training_data: DatasetT) -> ModelT:
        """Build a model given a random key and training dataset.

        The training dataset is provided to allow the model to infer
        dataset-specific dimensions such as observable and parameter sizes,
        as well as to perform any necessary preprocessing such as normalization.
        """
        ...


class ExperimentConfig(ABC):
    @classmethod
    @abstractmethod
    def register_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-line arguments for this experiment."""
        ...

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> Self:
        """Create an experiment configuration from parsed arguments."""
        ...

    @abstractmethod
    def run(self, output_dir: Path) -> None:
        """Run the experiment, saving outputs to the specified directory."""
        ...


def run_experiment[ModelT: AbstractLLR, DatasetT: ReweightableDataset](
    data_factory: DatasetFactory[DatasetT],
    model_config: ModelBuilder[ModelT, DatasetT],
    train_config: TrainingConfig[ModelT, DatasetT],
    *,
    key: PRNGKeyArray,
) -> tuple[ModelT, DatasetT, list[float], list[float], list[float]]:
    """Run a full experiment: data loading, model building, training.

    Returns:
        Tuple of (trained model, full dataset, train loss history, validation loss history, test loss history)

    TODO: require output directory and save results, including checkpoints.
    """
    data_key, model_key, split_key, train_key = jax.random.split(key, 4)
    data = data_factory(key=data_key)
    
    # Split data into training, validation, and test sets
    split_key1, split_key2 = jax.random.split(split_key)
    
    # First split: separate test set from train+val
    data_trainval, data_test = data.split(1.0 - train_config.test_fraction, key=split_key1)
    
    # Second split: separate train and val sets
    # Calculate the fraction of trainval that should be training
    train_frac_of_trainval = train_config.train_fraction / (train_config.train_fraction + train_config.val_fraction)
    data_train, data_val = data_trainval.split(train_frac_of_trainval, key=split_key2)
    
    # Build model using only training data
    model = model_config.build(key=model_key, training_data=data_train)
    
    # Train the model
    model, loss_train, loss_val = train(
        config=train_config,
        model=model,
        data_train=data_train,
        data_val=data_val,
        key=train_key,
    )
    
    # Evaluate on test set
    test_key = jax.random.split(train_key)[0]
    test_loss = [train_config.loss_fn(model, data_test, key=test_key).item()]
    
    return model, data, loss_train, loss_val, test_loss
