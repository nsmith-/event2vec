from copy import deepcopy
from dataclasses import KW_ONLY, InitVar
from typing import Generic, TypeVar

from jaxtyping import Array, Float, PRNGKeyArray

from event2vec.dataset import Dataset
from event2vec.losses import Loss, LossProtocol
from event2vec.models import Model

T = TypeVar("T")


class ModelWrapper(Model, Generic[T]):
    """Wraps an input object into an instance of (a final subclass of) Model.

    The wrapped object can be accessed either as
    `model_instance.wrapped_obj` or as `model_instance(*args, **kwargs)`.
    Here args and kwargs are simply ignored.
    """

    wrapped_obj: T
    "Object to be wrapped."

    _: KW_ONLY

    is_static: bool = False

    copy: InitVar[bool] = True
    "If True, a copy of the input object will be wrapped around."

    def __post_init__(self, copy: bool):
        if copy:
            self.wrapped_obj = deepcopy(self.wrapped_obj)

    def __call__(self, *args, **kwargs) -> T:
        return self.wrapped_obj


class LossWrapper(Loss):
    """Wraps a callable into an instance of (a final subclass of) Loss."""

    wrapped_callable: LossProtocol
    "Callable to be wrapped."

    _: KW_ONLY

    copy: InitVar[bool] = True
    "If True, a copy of the callable will be wrapped around."

    def __post_init__(self, copy: bool):
        if copy:
            self.wrapped_callable = deepcopy(self.wrapped_callable)

    def __call__(
        self, model: Model, data: Dataset, *, key: PRNGKeyArray | None = None
    ) -> Float[Array, ""]:
        return self.wrapped_callable(model=model, data=data, key=key)
