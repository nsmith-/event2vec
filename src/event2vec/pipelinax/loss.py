__all__ = ["Loss", "VmappedLoss"]


from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, final, override

import equinox as eqx
import jax

from .data import DataContent, DataPoint, DataSet
from .utils import dataset_partition_meta_constant_variable

if TYPE_CHECKING:
    from jaxtyping import Array, Float, PRNGKeyArray, PyTree

    from .type_aliases import Model, ModeStr


class Loss[ModelT: Model, DataContentT: DataContent](Protocol):
    "This should be used for type hinting training utils."

    @abstractmethod
    def __call__(
        self,
        *,
        model: ModelT,
        dataset: DataSet[DataContentT],
        key: PRNGKeyArray,
        mode: ModeStr,
    ) -> Float[Array, ""]:
        raise NotImplementedError


class VmappedLoss[ModelT: Model, DataContentT: DataContent](Loss[ModelT, DataContentT]):
    """
    A base loss class that handles vmapping, so subclasses don't have to.
    Intended to be subclassed by loss implementations.
    """

    @final
    @override
    def __call__(
        self,
        *,
        model: ModelT,
        dataset: DataSet[DataContentT],
        key: PRNGKeyArray,
        mode: ModeStr,
    ) -> Float[Array, ""]:
        """
        This vmaps self.elemwise_loss_fn over the dataset, postprocesses the
        result with self.post_process and returns the output.

        dev note: The rule that a DataPoint should be only initialized with
        the contents of a single datapoint is broken in this implementation.
        Doing so ensures that potential checks like
            `assert isinstance(datapoint, DataPoint)`
        within overrides of `elemwise_loss_fn` will pass. The fake DataPoint
        instances are not returned.
        """
        ## Create a DataPoint with only the meta and constant attrs #######
        dataset_meta, dataset_constant, dataset_variable = (
            dataset_partition_meta_constant_variable(dataset)
        )

        datapoint_constant_content = jax.tree.map(
            f=lambda arr: arr.squeeze(axis=0), tree=dataset_constant.content
        )

        datapoint_meta_and_constant = DataPoint(  # type: ignore[var-annotated]
            content=eqx.combine(dataset_meta.content, datapoint_constant_content)
        )
        del dataset_meta, dataset_constant
        ###################################################################

        (
            batchwise_key,
            *elemwise_keys,
            post_process_key,
        ) = jax.random.split(key, len(dataset) + 2)

        def _elemwise_loss_fn(datapoint_variable, elemwise_key):  # type: ignore[no-untyped-def]
            datapoint = eqx.combine(datapoint_meta_and_constant, datapoint_variable)

            return self.elemwise_loss_fn(
                model=model,
                datapoint=datapoint,
                elemwise_key=elemwise_key,
                batchwise_key=batchwise_key,
                mode=mode,
            )

        # TODO: Let/make subclasses provide the out_axes arg vmap?
        vmapped_elemwise_loss_fn = jax.vmap(_elemwise_loss_fn)

        batched_elemwise_loss = vmapped_elemwise_loss_fn(
            DataPoint(dataset_variable.content), elemwise_keys
        )

        return self.post_process(
            batched_elemwise_loss=batched_elemwise_loss,
            model=model,
            dataset=dataset,
            post_process_key=post_process_key,
            mode=mode,
        )

    @abstractmethod
    def elemwise_loss_fn(
        self,
        *,
        model: ModelT,
        datapoint: DataPoint[DataContentT],
        elemwise_key: PRNGKeyArray | None,  # for param sampling, dropout, etc.
        batchwise_key: PRNGKeyArray | None,  # because why not
        mode: ModeStr,
    ) -> PyTree:
        """
        A loss function that acts on a single datapoint. elemwise_key can be
        split further as needed (for dropout, param sampling, etc.).

        The returned PyTree can simply be a one- or multi-dimensional array,
        a tuple of arrays, etc.

        Abstract losses can finalize the implementation of elemwise_loss_fn
        in terms of additional abstractmethods.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self,
        *,
        batched_elemwise_loss: PyTree,  # vmapped elemwise loss
        model: ModelT,  # for advanced usage
        dataset: DataSet[DataContentT],  # for advanced usage
        post_process_key: PRNGKeyArray,  # for advanced usage
        mode: ModeStr,
    ) -> Float[Array, ""]:
        """
        This is used to post-process the batched output of elemwise_loss_fn.
        In typical usage, this would just involve computing a mean
        or a weighted mean.

        The kwargs model, dataset, and post_process_key are passed on for
        more advanced use cases. In principle, the entire loss implementation
        can be within post_process, completely ignoring elemwise_loss_fn.
        """
        raise NotImplementedError
