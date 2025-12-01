__all__ = ["partition_trainable_frozen", "dataset_partition_meta_constant_variable"]


from typing import TYPE_CHECKING

import equinox as eqx
import jax

from .nontrainability import is_marked_frozen, is_trainable_array

if TYPE_CHECKING:
    from jaxtyping import PyTree

    from .data import DataContent, DataSet
    from .type_aliases import Model


def partition_trainable_frozen(pytree: Model) -> tuple[PyTree, PyTree]:
    return eqx.partition(
        pytree=pytree,
        filter_spec=is_trainable_array,
        is_leaf=is_marked_frozen,
    )


def dataset_partition_meta_constant_variable[ContentT: DataContent](
    dataset: DataSet[ContentT],
) -> tuple[PyTree, PyTree, PyTree]:
    all_nones = eqx.filter(pytree=dataset, filter_spec=False, replace=None)

    dataset_meta = eqx.tree_at(
        where=lambda dset_tree: dset_tree.content.meta_attrs,
        pytree=all_nones,
        replace=dataset.content.meta_attrs,
        is_leaf=lambda node: node
        is None,  # In case the original `meta_attrs` is a leaf
    )

    dataset_sans_meta = eqx.tree_at(
        where=lambda tree: tree.content.meta_attrs, pytree=dataset, replace=None
    )

    dataset_constant_prefix, dataset_variable_prefix = eqx.partition(
        pytree=dataset_sans_meta, filter_spec=lambda x: len(x) == 1, replace=None
    )

    dataset_constant = jax.tree.broadcast(
        prefix_tree=dataset_constant_prefix, full_tree=dataset
    )

    dataset_variable = jax.tree.broadcast(
        prefix_tree=dataset_variable_prefix, full_tree=dataset
    )

    return dataset_meta, dataset_constant, dataset_variable
