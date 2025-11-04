from abc import abstractmethod

import equinox as eqx
from jaxtyping import Float, Array

class Loss(eqx.Module):
    # _init_needs_key: bool
    # _call_needs_key: bool

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Float[Array, ""]:
        raise NotImplementedError
