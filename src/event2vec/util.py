import jax.numpy as jnp
from rich.progress import (
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from event2vec.shapes import ParamQuadVec, ParamVec, PSDMatrix

EPS = jnp.finfo(jnp.float32).eps
"A small constant to avoid numerical issues with log(0) or division by zero."


def tril_outer_product(vec: ParamVec) -> ParamQuadVec:
    """Calculates the lower-triangular part of the outer product of a vector with itself.

    Useful to construct a vector representation of a quadratic formula (e.g. for SMEFT parameterizations).
    """
    outer = vec[..., None, :] * vec[..., None]
    il = jnp.tril_indices(vec.shape[-1])
    return outer[..., il[0], il[1]]


def tril_to_matrix(tril: ParamQuadVec) -> PSDMatrix:
    """Convert a lower-triangular vector representation back to a square matrix."""
    k = tril.shape[-1]
    # k = n*(n+1)/2  => n = (sqrt(8k+1)-1)/2
    n = int(((8 * k + 1) ** 0.5 - 1) / 2)
    il = jnp.tril_indices(n)
    mat = (
        jnp.zeros(tril.shape[:-1] + (n, n), dtype=tril.dtype)
        .at[..., il[0], il[1]]
        .add(tril / 2)
        .at[..., il[1], il[0]]
        .add(tril / 2)
    )
    return mat


def matrix_to_tril(mat: PSDMatrix) -> ParamQuadVec:
    """Convert a symmetric square matrix to a lower-triangular vector representation."""
    n = mat.shape[-1]
    il = jnp.tril_indices(n)

    diag_indices = [int(i * (i + 3) / 2) for i in range(n)]

    tril = 2 * (mat[..., il[0], il[1]].at[..., diag_indices].divide(2))
    return tril


def standard_pbar(*cols) -> Progress:
    """Standard progress bar with common columns.

    Args:
        *cols: Additional columns to add to the progress bar.
    """
    return Progress(
        TextColumn("[progress.description]{task.description:30s}"),
        TimeElapsedColumn(),
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("Rem:"),
        TimeRemainingColumn(),
    )
