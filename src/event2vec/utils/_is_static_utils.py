from typing import TypeVar
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
from jaxtyping import PyTree

from event2vec.nontrainable import FreezableModule, NonTrainableModule


def partition_trainable_and_static(pytree: PyTree) -> tuple[PyTree, PyTree]:
    return eqx.partition(
        pytree=pytree,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda node: isinstance(node, (NonTrainableModule, FreezableModule))
        and node.is_static,
    )


T = TypeVar("T", bound=FreezableModule)


def set_is_static(model: T, is_static_value: bool) -> T:
    """Returns a copy of model, with the given is_static value."""

    if not isinstance(model, FreezableModule):
        raise TypeError(f"`model` must be an instance of {FreezableModule}.")
    if not isinstance(is_static_value, bool):
        raise TypeError("`is_static_value` must be boolean.")

    return eqx.tree_at(
        where=lambda x: x.is_static, pytree=model, replace=is_static_value
    )


TreeT = TypeVar("TreeT", bound=PyTree)


def set_is_static_at(
    where: Callable[[TreeT], FreezableModule | Sequence[FreezableModule]],
    pytree: TreeT,
    is_static_value: bool,
) -> TreeT:
    """Returns a copy of pytree, with the is_static attribute of either
        i) a specific FreezableModule instance or
        ii) a sequence of FreezableModule instances
    set to is_static_value.

    The original pytree is unaffected. All nodes of the returned
    pytree are copies of the corresponding nodes of the input pytree.

    Parameters
    ----------
    where : Callable[[PyTree], FreezableModule | Sequence[FreezableModule]]
        A callable PyTree -> Model or PyTree -> tuple[Model, ...].
        It should consume a PyTree with the same structure as pytree, and
        return the model or models to be modified.
    pytree : PyTree
        The input pytree whose (modified) copy will be returned.
        pytree should pass equinox.tree_check.
    is_static_value : bool
        The new value of node.is_static_value (under the returned copy).

    Returns
    -------
    PyTree
        Copy of the input pytree with the required changes.
    """

    eqx.tree_check(pytree=pytree)

    if not isinstance(is_static_value, bool):
        raise TypeError("`is_static_value` must be boolean.")

    def replace_fn(node):
        if not isinstance(node, FreezableModule):
            raise TypeError(
                "where(pytree) should be either a single instance "
                "or a sequence of instances of {FreezableModule}."
            )
        return set_is_static(model=node, is_static_value=is_static_value)

    return eqx.tree_at(where=where, pytree=pytree, replace_fn=replace_fn)


def set_is_static_at_node(
    pytree: TreeT, node: FreezableModule, is_static_value: bool
) -> TreeT:
    """Returns a copy of pytree, with node.is_static set to is_static_value.

    The original pytree and node are unaffected. All nodes of the returned
    pytree are copies of the corresponding nodes of the input pytree.

    Parameters
    ----------
    pytree : PyTree
        The input pytree whose (modified) copy will be returned.
        pytree should pass equinox.tree_check.
    node : FreezableModule
        The node within pytree to be modified.
    is_static_value : bool
        The new value of node.is_static_value (under the returned copy).

    Returns
    -------
    PyTree
        Copy of the input pytree with the required changes.
    """

    eqx.tree_check(pytree)

    if not isinstance(node, FreezableModule):
        raise TypeError(f"`node` must be an instance of {FreezableModule}.")

    if not isinstance(is_static_value, bool):
        raise TypeError("`is_static_value` must be boolean.")

    # jax.tree.flatten does not create copies of leaves
    leaves, treedef = jax.tree.flatten(pytree, is_leaf=lambda x: x is node)

    for i, leaf in enumerate(leaves):
        if node is leaf:
            node_idx = i
            break
    else:
        raise RuntimeError("`node` was not found within `pytree`.")

    # equinox.tree_at will create copies of **all** leaves.
    new_leaves = eqx.tree_at(
        where=lambda x: x[node_idx].is_static, pytree=leaves, replace=is_static_value
    )

    return jax.tree.unflatten(treedef=treedef, leaves=new_leaves)
