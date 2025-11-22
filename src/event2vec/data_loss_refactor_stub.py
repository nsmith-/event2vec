# This is a stub file. It is not intended to be merged into main.
# This can form a submodule (for basic ML pipelining) that makes
# no reference to EFTs, making it easy to recycle for other projects.

from abc import abstractmethod
from typing import Protocol, overload, override

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

"""
Requirements notes:
1. When implementing a concrete dataset (from event files), it will likely be
   easier to construct a full dataset directly than to package each datapoint
   into a Datapoint instance and then package those into a Dataset instance.
   So try and support the former pattern.
2. It'll be nice to not have to mess with vmapping over datapoints, within
   loss implementations. So, provide a Datapoint API for losses to work with.
   Ensure that the same attributes exist within compatible (Datapoint, Dataset)
   combinations without repeating attribute-declarations.
   Do this without resorting to dynamic class generation, to allow for
   static typechecking.
3. Provide a vmap implementation that can consume a function defined on
   a compatible Datapoint, and apply it to a Dataset instance vectorized.
   Type checkers should raise an error if an incompatible function is used.
4. Make sure that (2) and (3) work properly with data methods, e.g., to
   compute event weights for given parameters.
5. Optional: Implement a way to do `dataset[idx]` and get a datapoint.
   Make sure that the output datapoint is properly type hinted so that
   compatible losses can act on it.
   Side note: This requirement is kinda limiting, and not very important.
"""


def _is_batchable(leaf):
    return eqx.is_array(leaf) and (leaf.ndim > 0)


def _none_out_all_leaves(pytree):
    return eqx.filter(pytree=pytree, filter_spec=False)


class DataContent(eqx.Module):
    """
    This class can contain the contents of a datapoint or a dataset. Packaging
    it explicitly within `DataPoint` or `Dataset` concretizes its nature.

    Methods can be implemented assuming datapoint or dataset behavior,
    or not assuming any a specific behavior. For safety, it is recommended
    that interactions with `DataContent` (including calling its methods) be
    performed only after packaging it appropriately.

    The leading dimension will be assumed to be the batch dimension
    for any array with ndim >= 1 that is contained
    (a) within a DataContent packaged inside a DataSet, but
    (b) outside DataContent.meta_attrs.
    """

    meta_attrs: PyTree
    "Contains meta attributes that are common to all datapoints."

    @abstractmethod
    def _len(self) -> int:
        """
        Returns the number of datapoints when self represents a dataset.
        Behavior is unspecified when self represents a datapoint.

        This will be used by DataSet.__len__, which is the recommended
        public interface.
        """
        raise NotImplementedError


class DataPoint[ContentT: DataContent](eqx.Module):
    """
    A container for an instance of DataContent. It declares that the
    datacontent should be treated as representing a single datapoint.

    The datacontent of a datapoint can be accessed by calling it: `datapoint()`

    This is a generic-typed final class. It is not intended to be subclassed.
    """

    _content: ContentT
    "Make sure that content contains a single datapoint (no batch dim)."

    def __init__(self, content: ContentT, /):
        self._content = content

    def __call__(self) -> ContentT:
        return self._content


class DataSet[ContentT: DataContent](eqx.Module):
    """
    A container for an instance of DataContent. It declares that the
    datacontent should be treated as representing a dataset.

    The datacontent of a dataset can be accessed by calling it: `dataset()`

    This is a generic-typed final class. It is not intended to be subclassed.
    """

    _content: ContentT
    "Make sure that content represents a dataset (batch axis=0)."

    def __init__(self, content: ContentT, /):
        self._content = content

    def __call__(self) -> ContentT:
        return self._content

    def __len__(self) -> int:
        return self._content._len()

    def get_vmap_in_axes_arg(self) -> PyTree:
        """
        The output can be used with eqx.filter_map to vmap properly over the
        batch axis. Example use:

        ```
        # Given:
        # dset: DataSet

        def point_func(dpoint: Datapoint) -> Array:
            return jnp.array([1.])

        set_func = eqx.filter_vmap(
            fun=point_func,
            in_axes=(dset.get_vmap_in_axes_arg(),)
        )

        set_func(dset)
        ```
        """

        # axis is 0 for batchable leaves, None otherwise ###
        vmap_in_axes_arg = jax.tree.map(
            f=lambda leaf: 0 if _is_batchable(leaf) else None, tree=self
        )
        ####################################################

        ## None out meta_attrs subtree #####################
        vmap_in_axes_arg = eqx.tree_at(
            where=lambda tree: tree._content.meta_attrs,
            pytree=vmap_in_axes_arg,
            replace=None,
        )
        ####################################################

        return vmap_in_axes_arg

    @overload
    def __getitem__(self, key: int) -> DataPoint[ContentT]: ...

    @overload
    def __getitem__(self, key: slice) -> "DataSet"[ContentT]: ...

    def __getitem__(
        self, key: int | slice
    ) -> DataPoint[ContentT] | "DataSet"[ContentT]:
        assert isinstance(key, (int, slice))

        ## Slice batchable leaves, None out the rest #######
        sliced_only_content = jax.tree.map(
            f=lambda leaf: leaf[key] if _is_batchable(leaf) else None,
            tree=self._content,
        )
        ####################################################

        ## None out contents of meta_attrs #################
        sliced_only_content = eqx.tree_at(
            where=lambda content: content.meta_attrs,
            pytree=sliced_only_content,
            replace_fn=_none_out_all_leaves,
        )
        ####################################################

        # combine will use the first non-None candidate (if any) for each leaf
        return_content: ContentT = eqx.combine(
            sliced_only_content,
            self._content,
        )

        if isinstance(key, slice):
            return DataSet(return_content)

        return DataPoint(return_content)


# TODO: Utility function to concatenate datasets
def concatenate_datasets[DataContentT: DataContent](
    datasets: list[DataSet[DataContentT]],
) -> DataSet[DataContentT]:
    raise NotImplementedError


# TODO: Utility function to merge datapoints into a dataset.
# Should be able to handle len(datapoints) = 1.
# Will be useful for calling a Loss instance on a single datapoint.
def make_dataset_from_datapoints[DataContentT: DataContent](
    datapoints: list[DataPoint[DataContentT]],
) -> DataSet[DataContentT]:
    raise NotImplementedError


type Model = eqx.Module  # stub, can substitute with AbstractLLR


class Loss[ModelT: Model, DataContentT: DataContent](Protocol):
    "This should be used for type hinting training utils."

    @abstractmethod
    def __call__(
        self,
        *,
        model: ModelT,
        dataset: DataSet[DataContentT],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        raise NotImplementedError


class LossBase[ModelT: Model, DataContentT: DataContent](Loss[ModelT, DataContentT]):
    """
    A base loss class that handles vmapping, so subclasses don't have to.
    Intended to be subclassed by loss implementations.
    """

    @override
    def __call__(
        self,
        *,
        model: ModelT,
        dataset: DataSet[DataContentT],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """
        This vmaps self.elemwise_loss_fn over the dataset, postprocesses the
        result with self.post_process and returns the output.
        """
        (
            batchwise_key,
            *elemwise_keys,
            post_process_key,
        ) = jax.random.split(key, len(dataset) + 2)

        # https://github.com/patrick-kidger/equinox/issues/405
        # filter_vap doesn't support kwargs, but it is still good to use
        # kwargs in the public elemwise_loss_fn.
        def _elemwise_loss_fn(model, datapoint, elemwise_key, batchwise_key):
            return self.elemwise_loss_fn(
                model=model,
                datapoint=datapoint,
                elemwise_key=elemwise_key,
                batchwise_key=batchwise_key,
            )

        in_axes = {
            "model": None,
            "datapoint": dataset.get_vmap_in_axes_arg(),
            "elemwise_key": 0,
            "batchwise_key": None,
        }

        vmapped_elemwise_loss_fn = eqx.filter_vmap(
            fun=_elemwise_loss_fn, in_axes=in_axes
        )

        batched_elemwise_loss = vmapped_elemwise_loss_fn(
            model, dataset, elemwise_keys, batchwise_key
        )

        return self.post_process(
            batched_elemwise_loss=batched_elemwise_loss,
            model=model,
            dataset=dataset,
            post_process_key=post_process_key,
        )

    @abstractmethod
    def elemwise_loss_fn(
        self,
        *,
        model: ModelT,
        datapoint: DataPoint[DataContentT],
        elemwise_key: PRNGKeyArray | None,  # for param sampling, dropout, etc.
        batchwise_key: PRNGKeyArray | None,  # because why not
    ) -> tuple[Array, ...]:
        """
        A loss function that acts on a single datapoint. elemwise_key can be
        split further as needed (for dropout, param sampling, etc.).

        Abstract losses can finalize the implementation of elemwise_loss_fn
        in terms of additional abstractmethods.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self,
        *,
        batched_elemwise_loss: tuple[Array, ...],  # vmapped elemwise loss
        model: ModelT,  # for advanced usage
        dataset: DataSet[DataContentT],  # for advanced usage
        post_process_key: PRNGKeyArray,  # for advanced usage
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
