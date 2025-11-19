from abc import abstractmethod

import equinox as eqx

from event2vec.shapes import (
    LLRScalar,
    ObsVec,
    ParamVec,
    PSDMatrix,
)


class AbstractLLR(eqx.Module):
    """Abstract learned log-likelihood ratio model.

    This is a model that can predict the log-likelihood ratio between two parameter points,
    given the event observables. No assumptions are made about the internal structure of the model.
    """

    @abstractmethod
    def llr_pred(
        self, observables: ObsVec, param_0: ParamVec, param_1: ParamVec
    ) -> LLRScalar:
        raise NotImplementedError


class AbstractPSDMatrixLLR(AbstractLLR):
    """A model that predicts a positive semi-definite matrix for the log-likelihood ratio dependence on the parameters.

    This is useful for models where the log-likelihood ratio has a quadratic dependence on the parameters.
    It is abstract to allow for different implementations of the PSD matrix prediction.
    """

    @abstractmethod
    def psd_matrix(self, observables: ObsVec) -> PSDMatrix:
        raise NotImplementedError
