# TODO: Implement EvaluationCore for computing metrics
# TODO: Improve shuffling, epoch_generator (make it an Iterator class with
#       a __len__ method for progress bars' benefit).


from typing import TYPE_CHECKING, final

import equinox as eqx
import jax

from .utils import dataset_partition_meta_constant_variable, partition_trainable_frozen

if TYPE_CHECKING:
    from collections.abc import Iterator

    import optax
    from jaxtyping import Array, Float, PRNGKeyArray

    from .data import DataContent, DataSet
    from .loss import Loss
    from .type_aliases import Model


@final
class TrainingCore[ModelT: Model, DataContentT: DataContent]:
    def __init__(
        self,
        *,
        initial_model: ModelT,
        training_dataset: DataSet[DataContentT],
        loss_fn: Loss[ModelT, DataContentT],
        optimizer: optax.GradientTransformation,
        optimizer_state: optax.OptState | None,
    ) -> None:
        dataset_meta, dataset_constant, dataset_variable = (
            dataset_partition_meta_constant_variable(training_dataset)
        )

        initial_model_trainable, model_frozen = partition_trainable_frozen(
            initial_model
        )

        # jax autodiff doesn't support kwargs
        def wrapped_loss_fn(
            model_trainable, databatch_variable, key
        ) -> Float[Array, ""]:
            model: ModelT = eqx.combine(model_trainable, model_frozen)

            databatch: DataSet[DataContentT] = eqx.combine(
                dataset_meta, dataset_constant, databatch_variable
            )

            return loss_fn(model=model, dataset=databatch, key=key, mode="training")

        loss_grad_fn = jax.value_and_grad(wrapped_loss_fn)

        # Using filter_ for the sake of opt_state. All other arguments
        # are guaranteed to only contain array (or NoneType) leaves.
        @eqx.filter_jit
        def train_step_aux(
            *,
            model_trainable,
            databatch_variable,
            opt_state,
            key,
        ) -> tuple[Array, ModelT, optax.OptState]:
            print("Jit compiling train step...")

            loss, grads = loss_grad_fn(model_trainable, databatch_variable, key)

            updates, opt_state = optimizer.update(grads, opt_state, model_trainable)

            model_trainable = eqx.apply_updates(model_trainable, updates)
            return loss, model_trainable, opt_state

        self._train_step_aux = train_step_aux

        self._current_model_trainable = initial_model_trainable
        self._dataset_variable = dataset_variable
        self._dataset_length = len(training_dataset)

        self._model_frozen = model_frozen

        if optimizer_state is None:
            self._current_optimizer_state = optimizer.init(initial_model_trainable)
        else:
            self._current_optimizer_state = optimizer_state

    def current_model(self) -> ModelT:
        return eqx.combine(self._current_model_trainable, self._model_frozen)

    def shuffle_dataset(self, key: PRNGKeyArray) -> None:
        shuffle_idx = jax.random.permutation(key=key, x=self._dataset_length)
        self._dataset_variable = jax.tree.map(
            f=lambda arr: arr[shuffle_idx],
            tree=self._dataset_variable,
        )

    def epoch_generator(
        self,
        *,
        batch_size: int,
        key: PRNGKeyArray,
        omit_last: bool = True,
    ) -> Iterator[Float[Array, ""]]:
        effective_len = self._dataset_length
        if omit_last:
            effective_len = (effective_len // batch_size) * batch_size

        for i, start in enumerate(range(0, effective_len, batch_size)):
            yield self.train_step(
                slice_idx=slice(start, start + batch_size),
                key=jax.random.fold_in(key, i),
            )

    def train_step(self, slice_idx, key) -> Float[Array, ""]:
        databatch_variable = jax.tree.map(
            f=lambda arr: arr[slice_idx], tree=self._dataset_variable
        )

        loss, self._current_model_trainable, self._current_optimizer_state = (
            self._train_step_aux(
                model_trainable=self._current_model_trainable,
                databatch_variable=databatch_variable,
                opt_state=self._current_optimizer_state,
                key=key,
            )
        )

        return loss
