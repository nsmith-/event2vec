from abc import abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from event2vec.models.psd_matrix_models import (
    PSDMatrixModel, PSDMatrixModel_WithUD
)
from event2vec.utils import set_nontrainable
from event2vec.losses import Loss

from event2vec.util import tril_to_matrix
from event2vec.dataset import ReweightableDataset

## TODO: Implement non-redundant versions of these losses

class PSDMatrixLoss(Loss):
    def __call__(self,
                 model: PSDMatrixModel,
                 data: ReweightableDataset,
                 **kwargs) -> Float[Array, ""]:
        pred_matrices = jax.vmap(model)(data.observables)
        label_matrices = tril_to_matrix(data.latent_data)

        return (
            jax.vmap(self._per_datapoint_loss)(pred_matrices, label_matrices)
        ).mean()

    @abstractmethod
    def _per_datapoint_loss(
            self,
            pred_matrix: Float[Array, "N N"],
            label_matrix: Float[Array, "N N"]
        ) -> Float[Array, ""]:
        raise NotImplementedError

class PSDMatrixLoss_DiagOnly(Loss):
    def __call__(self,
                 model: PSDMatrixModel_WithUD,
                 data: ReweightableDataset,
                 **kwargs) -> Float[Array, ""]:
        pred_diags = jax.vmap(model.D_model)(data.observables)

        label_matrices = tril_to_matrix(data.latent_data)
        U_matrices = jax.vmap(model.U_model)(data.observables)

        label_diags = jnp.empty_like(pred_diags)
        for i in range(pred_diags.shape[-1]):
            ui_vectors = U_matrices[...,:,i]

            tmp = jnp.sum(label_matrices * ui_vectors[..., None, :], axis=-1)
            tmp = jnp.sum(tmp * ui_vectors, axis=-1)

            label_diags = label_diags.at[..., i].set(tmp)

        return (
            jax.vmap(self._per_datapoint_loss)(pred_diags, label_diags)
        ).mean()

    @abstractmethod
    def _per_datapoint_loss(
            self,
            pred_diag: Float[Array, "K"],
            label_diag: Float[Array, "K"]
        ) -> Float[Array, ""]:
        raise NotImplementedError

class FrobeniusNormLoss(PSDMatrixLoss):
    """Returns the square of the Frobenius norm of `A @ (Y-T)`,
    where Y is the prediction matrix, T is the label matrix,
    and A is a fixed parameter matrix.

    A = None corresponds to the identity matrix."""

    A_matrix: Float[Array, "K N"] | None = None

    def __post_init__(self):
        # In case the loss instance ends up within a gradient argument.
        if self.A_matrix is not None:
            set_nontrainable(self.A_matrix)

    def _per_datapoint_loss(
            self,
            pred_matrix: Float[Array, "N N"],
            label_matrix: Float[Array, "N N"]
        ) -> Float[Array, ""]:

        diff_matrix = (pred_matrix - label_matrix)

        if self.A_matrix is None:
            return jnp.sum(diff_matrix**2)

        return jnp.sum((self.A_matrix @ diff_matrix)**2)

class HyperQuadNormLoss(PSDMatrixLoss):
    """Returns `sum_{ij,kl} P_{ij,kl} (Y-T)_{ij} (Y-T)_{kl}`,
    where Y is the prediction matrix, T is the label matrix,
    and P is a fixed parameter tensor with four indices.

    P is expected to be positive semidefinite, when viewing
    (ij) as the first index and (kl) as the second index.

    P = None corresponds to P_{ij,kl} = delta(i, k) delta(j, l)."""

    P_tensor: Float[Array, "N N N N"] | None = None

    def __post_init__(self):
        # In case the loss instance ends up within a gradient argument.
        if self.P_tensor is not None:
            set_nontrainable(self.P_tensor)

    def _per_datapoint_loss(
            self,
            pred_matrix: Float[Array, "N N"],
            label_matrix: Float[Array, "N N"]
        ) -> Float[Array, ""]:

        diff_matrix = (pred_matrix - label_matrix)

        if self.P_tensor is None:
            return jnp.sum(diff_matrix**2)

        tmp = jnp.sum(self.P_tensor * diff_matrix, axis=(2,3))
        return jnp.sum(tmp * diff_matrix)

class DiagMSELoss(PSDMatrixLoss_DiagOnly):
    def _per_datapoint_loss(
            self,
            pred_diag: Float[Array, "K"],
            label_diag: Float[Array, "K"]
        ) -> Float[Array, ""]:
        return jnp.sum((pred_diag - label_diag)**2)