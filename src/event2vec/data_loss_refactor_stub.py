# This is a stub file. It is not intended to be merged into main.
# This can form a submodule (for basic ML pipelining) that makes
# no reference to EFTs, making it easy to recycle for other projects.

from abc import abstractmethod
from collections.abc import Sequence
from typing import Protocol, overload, override

import equinox as eqx
import jax
import jax.numpy as jnp
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

Pros of the current implementation:
   This should facilitate strong static type checking, without much
   boilerplate for each narrower abstract DataContent. Also, all public
   methods and classes are meaningfully typed.

Quirk of the current public API:
   To access the contents of a DataPoint instance, say `dp`, one has to do
   `dp._content` or `dp()`. Likewise for DataSet.

Alternative implementation ideas that avoid this quirk:
1. Subclass DataContent + DataPointMixin into DataPoint; likewise for Dataset.
   Issues:
   a) For each new abstract DataContent, say ReweightableData, one has to
      create a corresponding ReweightableDataPoint and ReweightableDataSet,
      which adds boilerplate.
   b) New typing overloads have to be provided for `concatenate_datasets`
      and `stack_datapoints`, which adds boilerplate.
   c) One has to override the `__getitem__` method in each abstract DataSet
      just to provide type hints. Implementation will just involve calling
      `super().__getitem__(key)`. Likewise, in each abstract loss
      implementation on narrowed DataContent, one has to override the
      `LossBase.__call__` method just to provide type hints. Both of these
      add boilerplate. The former issue can be mitigated by forgoing the
      syntactic sugar offered by `__getitem__`.
   d) Constructing DataPoints from DataSets and vice versa will require hacks.
2. Absorb all the special methods of DataPoint and DataSet into DataContent
   itself. Create DataPoint and DataSet via `NewType`. A Literal attribute can
   be used to determine the DataPoint vs DataSet behavior at runtime.
   This solves issue (1d). Issues (1a), (1b) and (1c) are still present.
3. Create a generic DataContent[DataPointOrSetT] class (absorb all the special
   methods of DataPoint and DataSet into DataContent itself). Each new abstract
   DataContent will also be generic. A Literal attribute can be used to infer
   this type variable as well as determine the runtime point vs set behavior.
   This solves issues (1a) and (1d). Issues (1b) and (1c) are still present.
4. Issues (1a-d) as well as the quick of the current implementation can be
   solved using higher kinded types from the `returns` package, by type making
   functions/loss-classes generic in both DataContentType and PointOrSetType.
5. Finally, if allowing strong static type checking is not too important,
   many of these issues become non-issues.
"""


def _is_atleast_1d_array(leaf):  # type: ignore[no-untyped-def]
    return eqx.is_array(leaf) and (leaf.ndim > 0)


# A thin wrapper around jax.tree.map, which allows passing `f` and `tree`
# as keyword args when also providing the `rest` argument.
def _jaxtreemap(f, tree, rest_as_seq=[], is_leaf=None):  # type: ignore[no-untyped-def]
    return jax.tree.map(f, tree, *rest_as_seq, is_leaf=is_leaf)


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

    meta_attrs: eqx.AbstractVar[PyTree]
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

    # TODO: Make this a regular function instead of a DataContent method?
    def _get_batchable_filter_spec(self, prefix_okay: bool = False) -> PyTree:
        """
        Returns a "filter_spec" tree (with Boolean leaves), indicating which
        leaves of the datacontent represent batchable arrays. Such leaves will
        have a leading batch dim iff the datacontent corresponds to a dataset.

        `prefix_okay` indicates whether the structure of returned tree
        (a) can be a prefix of the datacontent's tree structure or
        (b) should match the datacontent's full tree structure exactly.
        """

        atleast_1d_filter_spec = jax.tree.map(
            f=_is_atleast_1d_array,
            tree=self,
        )

        prefix_batchable_filter_spec = eqx.tree_at(
            where=lambda tree: tree.meta_attrs,
            pytree=atleast_1d_filter_spec,
            replace=False,
        )

        if prefix_okay:
            return prefix_batchable_filter_spec

        return jax.tree.broadcast(
            prefix_tree=prefix_batchable_filter_spec,
            full_tree=self,
        )


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

    @overload
    def __getitem__(self, key: int) -> DataPoint[ContentT]: ...

    @overload
    def __getitem__(self, key: slice) -> "DataSet[ContentT]": ...

    # TODO: Offer this feature via util function(s) instead of a dunder method?
    def __getitem__(
        self, key: int | slice
    ) -> DataPoint[ContentT] | "DataSet[ContentT]":
        assert isinstance(key, (int, slice))

        return_content = _jaxtreemap(  # type: ignore[no-untyped-call]
            f=lambda x, batchable: x[key] if batchable else x,
            tree=self._content,
            rest_as_seq=[self._content._get_batchable_filter_spec()],
        )

        if isinstance(key, slice):
            return DataSet(return_content)

        return DataPoint(return_content)


def concatenate_datasets[DataContentT: DataContent](
    datasets: Sequence[DataSet[DataContentT]],
) -> DataSet[DataContentT]:
    """
    Utility function to concatenate datasets.

    The first dataset (in the list `datasets`) sets the structure of the
    content tree and provides the non-batchable leaves.
    """

    assert len(datasets) >= 1

    def _concatenate_where_batchable(batchable, *leaves):  # type: ignore[no-untyped-def]
        if batchable:
            return jnp.concatenate(leaves, axis=0)
        else:
            return leaves[0]  # get non-batchable leaves from the first dataset

    return_content = _jaxtreemap(  # type: ignore[no-untyped-call]
        f=_concatenate_where_batchable,
        tree=datasets[0]._content._get_batchable_filter_spec(),
        rest_as_seq=[dataset._content for dataset in datasets],
    )

    return DataSet(return_content)


def stack_datapoints[DataContentT: DataContent](
    datapoints: Sequence[DataPoint[DataContentT]],
) -> DataSet[DataContentT]:
    """
    Utility function to merge datapoints into a dataset.

    Can handle len(datapoints) = 1, which corresponds to elevating
    a datapoint into a dataset of length 1.

    The first datapoint (in the list `datapoints`) sets the structure of the
    content tree and provides the non-batchable leaves.
    """
    assert len(datapoints) >= 1

    def _stack_where_batchable(batchable, *leaves):  # type: ignore[no-untyped-def]
        if batchable:
            return jnp.stack(leaves, axis=0)
        else:
            return leaves[0]  # get non-batchable leaves from the first dataset

    return_content = _jaxtreemap(  # type: ignore[no-untyped-call]
        f=_stack_where_batchable,
        tree=datapoints[0]._content._get_batchable_filter_spec(),
        rest_as_seq=[datapoint._content for datapoint in datapoints],
    )

    return DataSet(return_content)


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

        dev note: The rule that a DataPoint should be only initialized with
        the contents of a single datapoint is broken in this implementation.
        Doing so ensures that potential checks like
            `assert isinstance(datapoint, DataPoint)`
        within overrides of `elemwise_loss_fn` will pass.
        """
        (
            batchwise_key,
            *elemwise_keys,
            post_process_key,
        ) = jax.random.split(key, len(dataset) + 2)

        # https://github.com/patrick-kidger/equinox/issues/405
        # filter_vap doesn't support kwargs, but it is still good to use
        # kwargs in the public elemwise_loss_fn.
        def _elemwise_loss_fn(model, datapoint, elemwise_key, batchwise_key):  # type: ignore[no-untyped-def]
            return self.elemwise_loss_fn(
                model=model,
                datapoint=datapoint,
                elemwise_key=elemwise_key,
                batchwise_key=batchwise_key,
            )

        datacontent_in_axes = jax.tree.map(
            f=lambda batchable: 0 if batchable else None,
            tree=dataset._content._get_batchable_filter_spec(),
        )

        in_axes = {
            "model": None,
            "datapoint": DataPoint(datacontent_in_axes),
            "elemwise_key": 0,
            "batchwise_key": None,
        }

        vmapped_elemwise_loss_fn = eqx.filter_vmap(
            fun=_elemwise_loss_fn, in_axes=in_axes
        )

        batched_elemwise_loss = vmapped_elemwise_loss_fn(
            model, DataPoint(dataset._content), elemwise_keys, batchwise_key
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
