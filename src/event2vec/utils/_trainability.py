import warnings

import equinox as eqx
from jaxtyping import PyTree

class NonTrainableModule(eqx.Module):
    ...

_nontrainable_flag_name = '_NONTRAINABLE_FLAG'

def set_nontrainable(pytree: PyTree) -> None:
    """Marks the pytree as nontrainable."""

    if isinstance(pytree, NonTrainableModule):
        warnings.warn(
            "Instances of `NonTrainableModule` are never trainable. "
            "Ignoring attempt to mark as nontrainable.",
            RuntimeWarning, stacklevel=2
        )
    elif isinstance(pytree, eqx.Module):
        object.__setattr__(pytree, _nontrainable_flag_name, True)
    elif eqx.is_inexact_array(pytree):
        setattr(pytree, _nontrainable_flag_name, True)
    elif eqx.is_array_like(pytree):
        warnings.warn(
            "\"array_like\" objects that are not \"inexact_arrays\" "
            "are never trainable. Ignoring attempt to mark as nontrainable.",
            RuntimeWarning, stacklevel=2
        )
    elif pytree is None:
        warnings.warn(
            "NoneType objects are never trainable. "
            "Ignoring attempt to mark as nontrainable.",
            RuntimeWarning, stacklevel=2
        )
    else:
        raise RuntimeError(
            f"Objects of `{type(pytree)}` cannot be marked as nontrainable."
        )

def unset_nontrainable(pytree: PyTree) -> None:
    """Removes the nontrainable flag set using `set_nontrainable`.
    Does not work for instances of `NonTrainableModule`."""

    if isinstance(pytree, NonTrainableModule):
        raise RuntimeError(
            "Instances of `NonTrainableModule` cannot be made trainable."
        )
    if isinstance(pytree, eqx.Module) or eqx.is_inexact_array(pytree):
        try:
            delattr(pytree, _nontrainable_flag_name)
        except AttributeError:
            pass
    elif eqx.is_array_like(pytree):
        raise RuntimeError(
            "\"array_like\" objects that are not \"inexact_arrays\" "
            "cannot be made trainable."
        )
    elif pytree is None:
        raise RuntimeError("NoneType objects cannot be made trainable.")
    else:
        warnings.warn(
            f"Objects of type `{type(pytree)}` are always treated as "
            "trainable. Ignoring attempt to mark as trainable.",
            RuntimeWarning, stacklevel=2
        )

def is_nontrainable(pytree: PyTree) -> bool:
    """Returns True if any of these conditions are satisfied:
       a) pytree is an instance of `NonTrainableModule`.
       b) pytree has been marked as nontrainable using `set_nontrainable`.
       c) pytree is array_like but isn't an inexact_array.
       d) pytree is a NoneType object.
    Returns False otherwise."""

    if isinstance(pytree, NonTrainableModule):
        return True

    if getattr(pytree, _nontrainable_flag_name, False):
        return True

    if eqx.is_array_like(pytree) and (not eqx.is_inexact_array(pytree)):
        return True

    if pytree is None:
        return True

    return False

def is_trainable_array(pytree: PyTree) -> bool:
    return (not is_nontrainable(pytree)) and eqx.is_array(pytree)

def partition_trainable_and_static(pytree: PyTree) -> tuple[PyTree, PyTree]:
    return eqx.partition(
        pytree=pytree,
        filter_spec=is_trainable_array,
        is_leaf=is_nontrainable
    )
