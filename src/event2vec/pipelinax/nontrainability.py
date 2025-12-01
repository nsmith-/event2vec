__all__ = ["FreezableModule", "NonTrainableModule", "FrozenNumpyArray"]

from typing import TYPE_CHECKING, Any, Self, final

import equinox as eqx
import numpy as np

if TYPE_CHECKING:
    from jaxtyping import PyTree


class FreezableModule(eqx.Module):
    """A module that can be frozen (i.e., made non-trainable) or unfrozen.

    This is a useful base class for multi-stage training, where some parts
    of the model should be kept frozen during certain training phases.
    """

    is_frozen: eqx.AbstractVar[bool]
    """
    Whether the class instance should be treated as frozen by the utility
    function `partition_trainable_and_frozen`, i.e., whether its parameters
    should be considered frozen in training.
    """

    def __check_init__(self):
        assert isinstance(self.is_frozen, bool)


class NonTrainableModule(eqx.Module):
    """
    A non-trainable module.

    This is a base class for modules that do not have any trainable parameters.
    It can be used to define frozen transformations or computations that are
    part of the model architecture but do not require training.
    """


@final
class FrozenNumpyArray(np.ndarray):
    # TODO: Replace __new__ with a converter function, say, `as_frozen_ndarray`?
    # TODO: Check if runtime type/shape checkers work as expected for
    #       `jaxtyping.Float[FrozenNumpyArray, shape_str]`. Static type checking
    #       should work since `jaxtyping.Float` is identical to `typing.Annotated`.
    # TODO: Add a converter_factory to inject arbitrary type-hint-strings?
    """
    This is a dummy subclass of a `numpy.ndarray`, used simply to **tag**
    an array as frozen for training purposes.

    Subclassing `jax.Array` isn't officially supported, so... good thing
    jax plays well with numpy arrays!

    Guarantee: Training routines within `pipelinax` will not modify these
    arrays (via grad descent, regularization, etc.) or remove the frozen-tag.

    Non-guarantees:
    1. Operations involving FrozenNumpyArray objects are not guaranteed to
       return outputs that are themselves tagged as frozen.
    2. `FrozenNumpyArray` objects, like regular `numpy.ndarrays` are mutable.
       (the "writeable" flag can be turned back on). Furthermore, even the
       immutability of frozen dataclasses (like `equinox.Module` instances),
       which contain the `FrozenNumpyArray` objects is only emulated.
       So, unintentional clobbering of frozen arrays is very much possible.

    Usage:
    ```python
    # `arr_like` can be any valid array-like object, including
    #   - jax/numpy arrays and numpy.generic objects
    #   - lists, tuples, etc.
    #   - Python scalars (float, int, etc.). These are treated as frozen by
    #     default anyway, but one could tag and make the frozen-ness explicit.

    arr_like = [1.0, 2.0]

    frozen_np_arr = FrozenNumpyArray(arr_like)
    np_arr = np.array(arr_like)

    assert np.all(frozen_np_arr == np_arr)
    assert is_marked_frozen(frozen_np_arr)
    assert not is_marked_frozen(np_arr)
    ```

    This class is designed to have almost no flexibility/moving pieces.
    To set dtype, order, etc. use {np | jnp}.{array | asarray} first, before
    converting into a FrozenNumpyArray instance.

    This class can be used both as a type-hint and as a converter for
    frozen attributes of equinox modules. For example:
    ```python
    class Foo(equinox.Module):
        param: FrozenNumpyArray = equinox.field(converter=FrozenNumpyArray)
    ```
    or (with better type-hinting, including within __init__'s docstring)
    ```python
    from jaxtyping import Float, Array


    # equinox copies the type annotation of the converter into __init__'s
    # docstring. This, of course, doesn't help with static typechecking.
    def _converter(value: Float[Array, "5"]):
        return FrozenNumpyArray(value)


    class Foo(equinox.Module):
        param: Float[FrozenNumpyArray, "5"] = equinox.field(converter=_converter)
    ```
    """

    def __new__(cls, value: np.typing.ArrayLike, /) -> Self:
        return np.array(value, copy=True).view(FrozenNumpyArray)

    def __array_finalize__(self: Self, obj: Any) -> None:
        # The "writeable" flag being off should not be relied upon anywhere.
        self.flags.writeable = False


def is_marked_frozen(node: PyTree) -> bool:
    if isinstance(node, (FrozenNumpyArray, NonTrainableModule)):
        return True
    elif isinstance(node, FreezableModule) and node.is_frozen:
        return True
    else:
        return False


def is_trainable_array(node: PyTree) -> bool:
    return eqx.is_inexact_array(node) and (not is_marked_frozen(node))
