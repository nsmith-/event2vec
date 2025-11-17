from abc import abstractmethod
from dataclasses import KW_ONLY

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray

from event2vec.models.psd_matrix_models import PSDMatrixModel, PSDMatrixModel_WithUD

from event2vec.util import tril_to_matrix
from event2vec.dataset import QuadraticReweightableDataset

## TODO: Implement non-redundant versions of these losses?


class PSDMatrixLoss:
    def __call__(
        self,
        model: PSDMatrixModel,
        data: QuadraticReweightableDataset,
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        pred_matrices = jax.vmap(model)(data.observables)
        label_matrices = tril_to_matrix(data.quadratic_form)

        return (
            jax.vmap(self._per_datapoint_loss)(pred_matrices, label_matrices)
        ).mean()

    @abstractmethod
    def _per_datapoint_loss(
        self, pred_matrix: Float[Array, "N N"], label_matrix: Float[Array, "N N"]
    ) -> Float[Array, ""]:
        raise NotImplementedError


class PSDMatrixLoss_DiagOnly:
    def __call__(
        self,
        model: PSDMatrixModel_WithUD,
        data: QuadraticReweightableDataset,
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        pred_diags = jax.vmap(model.D_model)(data.observables)

        label_matrices = tril_to_matrix(data.quadratic_form)
        U_matrices = jax.vmap(model.U_model)(data.observables)

        label_diags = jnp.empty_like(pred_diags)
        for i in range(pred_diags.shape[-1]):
            ui_vectors = U_matrices[..., :, i]

            tmp = jnp.sum(label_matrices * ui_vectors[..., None, :], axis=-1)
            tmp = jnp.sum(tmp * ui_vectors, axis=-1)

            label_diags = label_diags.at[..., i].set(tmp)

        return (jax.vmap(self._per_datapoint_loss)(pred_diags, label_diags)).mean()

    @abstractmethod
    def _per_datapoint_loss(
        self, pred_diag: Float[Array, " K"], label_diag: Float[Array, " K"]
    ) -> Float[Array, ""]:
        raise NotImplementedError


class FrobeniusNormLoss(PSDMatrixLoss):
    """Returns the square of the Frobenius norm of `A @ (Y-T)`,
    where Y is the prediction matrix, T is the label matrix,
    and A is a fixed parameter matrix.

    A = None corresponds to the identity matrix."""

    _: KW_ONLY

    A_matrix: Float[Array, "K N"] | None = None

    def _per_datapoint_loss(
        self, pred_matrix: Float[Array, "N N"], label_matrix: Float[Array, "N N"]
    ) -> Float[Array, ""]:
        diff_matrix = pred_matrix - label_matrix

        if self.A_matrix is None:
            return jnp.sum(diff_matrix**2)

        return jnp.sum((self.A_matrix @ diff_matrix) ** 2)


class HyperQuadNormLoss(PSDMatrixLoss):
    """Returns `sum_{ij,kl} P_{ij,kl} (Y-T)_{ij} (Y-T)_{kl}`,
    where Y is the prediction PSD matrix, T is the label PSD matrix,
    and P is a fixed parameter tensor with four indices.

    P = None corresponds to P_{ij,kl} = delta(i, k) delta(j, l).

    More notes:
    ----------

    During initialization, P_{ij,kl} will be symmetrized with respect to the
    following transformations:
        (a)  i <--> j
        (b)  k <--> l
        (c) ij <--> kl
    In other words this transformation will be performed upon initialization:
        P_{ij,kl} := (
              P_{ij,kl} + P_{ji,kl} + P_{ij,lk} + P_{ji,lk}
            + P_{kl,ij} + P_{kl,ji} + P_{lk,ij} + P_{lk,ji}
        ) / 8

    After the transformation above, P is expected to be positive semidefinite,
    when viewing (ij) as the first index and (kl) as the second index.

    Furthermore, under this view, P should have N*(N+1)/2 non-zero eigenvalues,
    in order to regress **all** the free components of the prediction matrix Y.
    """

    _: KW_ONLY

    P_tensor: Float[Array, "N N N N"] | None = None

    def __post_init__(self):
        if self.P_tensor is None:
            return

        self.P_tensor = (
            self.P_tensor + jnp.transpose(self.P_tensor, axes=(1, 0, 2, 3))
        ) / 2

        self.P_tensor = (
            self.P_tensor + jnp.transpose(self.P_tensor, axes=(0, 1, 3, 2))
        ) / 2

        self.P_tensor = (
            self.P_tensor + jnp.transpose(self.P_tensor, axes=(2, 3, 0, 1))
        ) / 2

    def _per_datapoint_loss(
        self, pred_matrix: Float[Array, "N N"], label_matrix: Float[Array, "N N"]
    ) -> Float[Array, ""]:
        diff_matrix = pred_matrix - label_matrix

        if self.P_tensor is None:
            return jnp.sum(diff_matrix**2)

        tmp = jnp.sum(self.P_tensor * diff_matrix, axis=(2, 3))
        return jnp.sum(tmp * diff_matrix)


class DiagMSELoss(PSDMatrixLoss_DiagOnly):
    def _per_datapoint_loss(
        self, pred_diag: Float[Array, " K"], label_diag: Float[Array, " K"]
    ) -> Float[Array, ""]:
        return jnp.sum((pred_diag - label_diag) ** 2)
