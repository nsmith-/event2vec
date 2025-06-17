import jax
import jax.numpy as jnp

from event2vec.dataset import get_data
from event2vec.models import build_model
from event2vec.prior import ToyParameterPrior
from event2vec.training import train

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    data_key, model_key, train_key = jax.random.split(key, 3)
    data_train, data_validation = get_data(
        dataset_size=100_000,
        validation_fraction=0.15,
        key=data_key,
    )
    model = build_model(
        event_dim=2,
        param_dim=3,
        summary_dim=2,
        hidden_size=4,
        depth=3,
        key=model_key,
    )
    train(
        model=model,
        data_train=data_train,
        data_test=data_validation,
        param_prior=ToyParameterPrior(alpha=jnp.array([9.0, 3.0, 3.0])),
        batch_size=128,
        learning_rate=0.005,
        epochs=50,
        key=train_key,
    )
