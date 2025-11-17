from abc import ABC, abstractmethod
import argparse
from pathlib import Path
from typing import Protocol, Self, TypeVar

import jax

from event2vec.dataset import ReweightableDataset
from event2vec.model import LearnedLLR
from event2vec.training import TrainingConfig

DatasetT = TypeVar("DatasetT", bound=ReweightableDataset, covariant=True)
ModelDatasetT = TypeVar("ModelDatasetT", bound=ReweightableDataset, contravariant=True)
ModelT = TypeVar("ModelT", bound=LearnedLLR, covariant=True)


class DatasetFactory(Protocol[DatasetT]):
    def __call__(self, *, key: jax.Array) -> DatasetT:
        """Create a dataset given a random key."""
        ...


class ModelBuilder(Protocol[ModelT, ModelDatasetT]):
    def build(self, *, key: jax.Array, training_data: ModelDatasetT) -> ModelT:
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


def run_experiment(
    data_factory: DatasetFactory[DatasetT],
    model_config: ModelBuilder[ModelT, DatasetT],
    train_config: TrainingConfig,
    *,
    key: jax.Array,
) -> tuple[ModelT, DatasetT, list[float], list[float]]:
    """Run a full experiment: data loading, model building, training.

    TODO: require output directory and save results, including checkpoints.
    """
    data_key, model_key, train_key = jax.random.split(key, 3)
    data = data_factory(key=data_key)
    # TODO: restrict to training data only (train-test split needs to move out of train_config.train)
    model = model_config.build(key=model_key, training_data=data)
    model, loss_train, loss_test = train_config.train(
        model=model,
        data=data,
        key=train_key,
    )
    return model, data, loss_train, loss_test
