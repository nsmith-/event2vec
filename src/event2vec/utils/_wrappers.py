import jax
import jax.numpy as jnp

from event2vec.models import Model
from event2vec.utils import NonTrainableModule, set_nontrainable

class ArrayAsModel(Model):
    """Model that returns `wrapped_arr` as the output regardless of input."""

    wrapped_arr: jax.Array

    def __call__(self, *args, **kwargs):
        return self.wrapped_arr

class ArrayAsNonTrainableModel(Model, NonTrainableModule):
    """Nontrainable variant of ArrayAsModel."""

    wrapped_arr: jax.Array

    def __call__(self, *args, **kwargs):
        return self.wrapped_arr

def nontrainable_copy(array: jax.Array):
    array_copy = jnp.copy(array)
    set_nontrainable(array_copy)

    return array_copy
