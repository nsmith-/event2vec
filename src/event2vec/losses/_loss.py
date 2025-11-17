from abc import abstractmethod
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray

from event2vec.models import Model

from event2vec.dataset import Dataset


class Loss(eqx.Module):
    @abstractmethod
    def __call__(
        self, model: Model, data: Dataset, *, key: PRNGKeyArray | None
    ) -> Float[Array, ""]:
        raise NotImplementedError
