import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rich.progress

from event2vec.dataset import ReweightableDataset
from event2vec.models import LearnedLLR
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
    raise NotImplementedError("TODO")
    # return optax.losses.sigmoid_binary_cross_entropy(pred_y, batch.label).mean()


def train(
    *,
    model: LearnedLLR,
    data_train: ReweightableDataset,
    data_test: ReweightableDataset,
    param_prior: ParameterPrior,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    key: jax.Array,
):
    loss_fn = loss_mse

    def sample_prior_like(batch: ReweightableDataset, *, key: jax.Array) -> jax.Array:
        """Sample parameters from the prior for each event in the batch."""
        return jax.vmap(param_prior.sample)(jax.random.split(key, len(batch)))

    @eqx.filter_jit
    def make_step(
        model: LearnedLLR,
        batch: ReweightableDataset,
        opt_state: optax.OptState,
        *,
        key: jax.Array,
    ):
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

    optim = optax.adamw(learning_rate, b1=0.9)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    for epoch in rich.progress.track(range(epochs), description="Training"):
        tmp = []
        for i, batch in enumerate(data_train.iter_batch(batch_size)):
            key, subkey = jax.random.split(key)
            loss, model, opt_state = make_step(model, batch, opt_state, key=subkey)
            tmp.append(loss.item())
        train_loss_history.append(sum(tmp) / len(tmp))
        key, subkey = jax.random.split(key)
        test_loss_history.append(
            loss_fn(model, data_test, sample_prior_like(data_test, key=subkey)).item()
        )

    return model, train_loss_history, test_loss_history
