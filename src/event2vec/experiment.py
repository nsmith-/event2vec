from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp

from event2vec.dataset import ReweightableDataset
from event2vec.datasets import GaussMixtureDatasetFactory
from event2vec.model import E2VMLPConfig, LearnedLLR
from event2vec.prior import DirichletParameterPrior, UncorrelatedJointPrior
from event2vec.training import TrainingConfig
from event2vec.loss import BCELoss

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


def run_experiment(
    data_factory: DatasetFactory[DatasetT],
    model_config: ModelBuilder[ModelT, DatasetT],
    train_config: TrainingConfig,
    *,
    key: jax.Array,
) -> tuple[ModelT, DatasetT, list[float], list[float]]:
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


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    gen_param_prior = DirichletParameterPrior(alpha=jnp.array([9.0, 3.0, 3.0]))
    train_param_prior = gen_param_prior
    data_factory = GaussMixtureDatasetFactory(
        len=100_000,
        param_prior=gen_param_prior,
    )
    model_config = E2VMLPConfig(
        summary_dim=2,
        hidden_size=4,
        depth=3,
    )
    train_config = TrainingConfig(
        test_fraction=0.1,
        batch_size=128,
        learning_rate=0.005,
        epochs=50,
        loss_fn=BCELoss(UncorrelatedJointPrior(gen_param_prior)),
    )
    run_experiment(data_factory, model_config, train_config, key=key)
