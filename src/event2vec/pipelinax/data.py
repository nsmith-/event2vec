__all__ = ["DataContent", "DataPoint", "DataSet"]

from typing import Final, final

import equinox as eqx
import jax
import numpy as np
from jaxtyping import PyTree

## TODO: Implement indexing/slicing a dataset.


_allowed_outside_meta_attrs: Final = (jax.Array, np.ndarray)
_allowed_outside_meta_attrs_str: Final = "NoneType, jax.Array, numpy.ndarray"
_allowed_inside_meta_attrs: Final = (
    *_allowed_outside_meta_attrs,
    np.generic,
    float,
    complex,
    bool,
    str,
)
_allowed_inside_meta_attrs_str: Final = (
    _allowed_outside_meta_attrs_str + ", numpy.generic, float, complex, bool, str"
)


class DataContent(eqx.Module):
    f"""
    This class can contain the contents of a datapoint or a dataset. Packaging
    it explicitly within `DataPoint` or `Dataset` concretizes its nature.

    Methods can be implemented assuming datapoint or dataset behavior,
    or not assuming any specific behavior. For safety, it is recommended
    that interactions with `DataContent` (including calling its methods) be
    performed only after packaging it appropriately.

    Meta attributes and constant (broadcasted) attributes:
    --------------------------------------------------
    Leaves contained within `DataContent.meta_attrs` will be treated as
    representing meta attributes common to all datapoints. These leaves are
    assumed to not have a batch dim, in both `DataPoint` and `DataSet` modes.

    Any non-None leaf not within `DataContent.meta_attrs` will be assumed
    to be batchable. Such leaves should have a leading batch dim in `DataSet`
    mode, but not in `DataPoint` mode. The length of this batch dim must be
    either the length of the dataset or 1.

    Each leaf with length-1 batch dim will be treated as representing a
    constant attribute (i.e., intended to be broadcasted over the datapoints).

    As an edgecase, all NoneType leaves act like meta attributes.

    Allowed types for leaves of DataContent:
    ---------------------------------------
    (dev note: being explicit here to avoid surprises)

     - Within meta_attrs: ({_allowed_inside_meta_attrs_str})
     - Outside meta_attrs: ({_allowed_outside_meta_attrs_str})

    `DataContent.meta_attrs` can itself be None.

    Additional notes:
    ----------------
    All numpy.ndarray leaves will be made non-writeable by __post_init__
    to prevent accidental clobbering by operations on a DataContent instance.

    Dev note:
    --------
    A common use case for broadcasted constant attributes is in a dataset with
    all datapoints having the same weight. Alternative approaches for this
    use case:
    - Pack weight inside `DataContent.meta_attrs`.
      Con: A different dataset may need non-constant weights. It will be
      redundant to define different loss objects to handle the two cases,
      namely weights inside and outside meta_attrs.
    - Duplicate the constant weights, so broadcasting isn't needed.
      Con: jit will trace over weight instead of treating it as the constant
      attribute that it is. Additionally, this could ncrease memory footprint.
      These factors could be significant if the number/dimensionality of
      constant attributes is large?
    - Implement the constant weight as a property.
      Con: None. This approach is a fair alternative.
    """

    meta_attrs: eqx.AbstractVar[PyTree]
    "Contains meta attributes that are common to all datapoints."

    def __post_init__(self):
        for leaf in jax.tree.leaves(self):
            if isinstance(leaf, np.ndarray):
                leaf.flags.writeable = False

    def __check_init__(self):
        self_sans_meta = eqx.tree_at(
            where=lambda tree: tree.meta_attrs,
            pytree=self,
            replace=None,
            is_leaf=lambda leaf: leaf is None,
        )

        for leaf in jax.tree.leaves(self_sans_meta):
            if not isinstance(leaf, _allowed_outside_meta_attrs):
                raise ValueError(
                    "Each leaf of DataContent outside DataContent.meta_attrs "
                    f"must be of one of these types: ({_allowed_outside_meta_attrs_str}). "
                    f"Found a leaf of type {type(leaf)}."
                )

        for leaf in jax.tree.leaves(self.meta_attrs):
            if not isinstance(leaf, _allowed_inside_meta_attrs):
                raise ValueError(
                    "Each leaf of DataContent within DataContent.meta_attrs "
                    f"must be of one of these types: ({_allowed_inside_meta_attrs_str}). "
                    f"Found a leaf of type {type(leaf)}."
                )

        for leaf in jax.tree.leaves(self):
            if isinstance(leaf, np.ndarray):
                assert not leaf.flags.writeable


@final
class DataPoint[ContentT: DataContent](eqx.Module):
    """
    A container for an instance of DataContent. It declares that the
    datacontent should be treated as representing a single datapoint.

    This is a generic-typed final class. It is not intended to be subclassed.

    Arrays in `content` should not have a leading batch-dim.
    """

    content: ContentT
    "Make sure that content contains a single datapoint (no batch dim)."


@final
class DataSet[ContentT: DataContent](eqx.Module):
    """
    A container for an instance of DataContent. It declares that the
    datacontent should be treated as representing a dataset.

    This is a generic-typed final class. It is not intended to be subclassed.

    Each non-None leaf of `content` outside `content.meta_attrs` must have
    a leading batch dim. The length of this batch dim must be either the
    length of the dataset or 1. Each leaf with length-1 batch dim
    will be treated as representing a constant attribute (i.e., intended to be
    broadcasted over the datapoints). This logic will be used to infer the
    length of a dataset.

    Note:
    - The length of the dataset will be inferred as 0 if there is no non-None
    leaf outside `content.meta_attrs`.
    - The length of the dataset will be inferred as 1 if every non-None,
    leaf outside `content.meta_attrs` has a length-1 leading dimension.

    A dataset with > 1 datapoints, where every attribute is constant (for some
    reason) can be implemented by adding a dummy datapoint_id attribute.
    """

    content: ContentT
    "Make sure that content represents a dataset (batch axis=0)."

    def __check_init__(self):
        content_sans_meta = eqx.tree_at(
            where=lambda tree: tree.meta_attrs,
            pytree=self.content,
            replace=None,
            is_leaf=lambda leaf: leaf is None,
        )

        leading_dim_lengths = set()
        for leaf in jax.tree.leaves(content_sans_meta):
            assert isinstance(leaf, _allowed_outside_meta_attrs)  # for type narrowing

            if leaf.ndim < 1:
                raise ValueError(
                    "Leaves in `content` outside `content.meta_attrs` "
                    "must have ndim >= 1"
                )

            leading_dim_lengths.add(len(leaf))

        if len(leading_dim_lengths - {1}) > 1:
            raise ValueError(
                "Leading dim of each leaf in `content` but outside "
                "`content.meta_attrs` must have length equal to 1 "
                "or the length of the dataset. Found leaves with the "
                f"following leading-dim-lengths: {leading_dim_lengths}."
            )

    def __len__(self) -> int:
        content_sans_meta = eqx.tree_at(
            where=lambda tree: tree.meta_attrs,
            pytree=self.content,
            replace=None,
            is_leaf=lambda leaf: leaf is None,
        )

        leading_dim_lengths = {len(arr) for arr in jax.tree.leaves(content_sans_meta)}

        assert len(leading_dim_lengths - {1}) <= 1

        if len(leading_dim_lengths) == 0:
            return 0
        else:
            return max(leading_dim_lengths)
