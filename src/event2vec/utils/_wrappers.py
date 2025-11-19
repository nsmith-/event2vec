from copy import deepcopy
from dataclasses import KW_ONLY, InitVar

from event2vec.models import Model


class ModelWrapper[T](Model):
    """Wraps an input object into an instance of (a final subclass of) Model.

    The wrapped object can be accessed either as
    `model_instance.wrapped_obj` or as `model_instance(*args, **kwargs)`.
    Here args and kwargs are simply ignored.
    """

    wrapped_obj: T
    "Object to be wrapped."

    _: KW_ONLY

    is_static: bool = False

    copy: InitVar[bool] = True  # type: ignore[assignment]
    "If True, a copy of the input object will be wrapped around."

    def __post_init__(self, copy: bool):
        if copy:
            self.wrapped_obj = deepcopy(self.wrapped_obj)

    def __call__(self, *args, **kwargs) -> T:
        return self.wrapped_obj
