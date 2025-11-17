from dataclasses import KW_ONLY

import equinox as eqx
import jax.numpy as jnp

from event2vec.models._model import Model


class PSDMatrixModel(Model):
    N: eqx.AbstractVar[int]
    "Dimensionality of the PSD matrix being modeled."


class PSDMatrixModel_WithUD(PSDMatrixModel):
    U_model: eqx.AbstractVar[Model]
    "Expected to return an array of shape (N, K) with orthonormal columns."

    D_model: eqx.AbstractVar[Model]
    "Expected to return an array of shape (K,) containing non-negative elements."


class At_A_Model(PSDMatrixModel):
    """Returns A.T @ A

    A_model should be able to handle the call signature received by the
    resultant model.
    """

    _: KW_ONLY

    N: int

    A_model: Model
    "Expected to return an arbitrary array of shape (K, N)."

    is_static: bool = False

    def __call__(self, *args, **kwargs):
        A = self.A_model(*args, **kwargs)
        A = A.reshape(-1, self.N)
        return A.T @ A


class U_sqrtD_At_A_sqrtD_Ut_Model(PSDMatrixModel_WithUD):
    """Returns U @ D^(1/2) @ A.T @ A @ D^(1/2) @ U.T

    U_model, D_model, and A_model should all be able to handle the
    call signature received by the resultant model.
    """

    _: KW_ONLY

    N: int

    U_model: Model
    "Expected to return an array of shape (N, K) with orthonormal columns."

    D_model: Model
    "Expected to return an array of shape (K,) containing non-negative elements."

    A_model: Model
    "Expected to return an arbitrary array of shape (L, K)."

    normalize_A_cols: bool = True
    "Whether or not to normalize the columns of A before usage."

    is_static: bool = False

    def __call__(self, *args, **kwargs):
        U = self.U_model(*args, **kwargs)
        A = self.A_model(*args, **kwargs)
        D = self.D_model(*args, **kwargs)

        U = U.reshape(self.N, -1)

        K = U.shape[-1]
        A = A.reshape(-1, K)
        D = D.reshape(K)

        if self.normalize_A_cols:
            A = A / jnp.linalg.norm(A, axis=0, keepdims=True)

        tmp = (A * jnp.sqrt(D)) @ U.T

        return tmp.T @ tmp
