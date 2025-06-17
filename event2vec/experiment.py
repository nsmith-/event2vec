import jax
import jax.numpy as jnp

from event2vec.dataset import ToyDatasetFactory
from event2vec.model import E2VMLPConfig
from event2vec.prior import ToyParameterPrior
from event2vec.training import TrainingConfig


def run_experiment(
    data_factory: ToyDatasetFactory,
    model_config: E2VMLPConfig,
    train_config: TrainingConfig,
    *,
    key: jax.Array,
):
    data_key, model_key, train_key = jax.random.split(key, 3)
    data = data_factory(key=data_key)
    model = model_config.build(key=model_key)
    model, loss_train, loss_test = train_config.train(
        model=model,
        data=data,
        key=train_key,
    )
    return model, data, loss_train, loss_test


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    gen_param_prior = ToyParameterPrior(alpha=jnp.array([9.0, 3.0, 3.0]))
    train_param_prior = gen_param_prior
    data_factory = ToyDatasetFactory(
        len=100_000,
        param_prior=gen_param_prior,
    )
    model_config = E2VMLPConfig(
        event_dim=2,
        param_dim=3,
        summary_dim=2,
        hidden_size=4,
        depth=3,
    )
    train_config = TrainingConfig(
        test_fraction=0.1,
        batch_size=128,
        learning_rate=0.005,
        epochs=50,
        param_prior=train_param_prior,
        loss_fn="mse",
    )
    run_experiment(data_factory, model_config, train_config, key=key)
