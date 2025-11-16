"""Non-trainable modules for use in defining model architectures."""

from abc import abstractmethod
import warnings
import equinox as eqx

from event2vec.util import ConstituentModel
import jax
import jax.numpy as jnp

from event2vec.util import tril_to_matrix


class FreezableModule(eqx.Module):
    """A module that can be frozen (made non-trainable) or unfrozen.

    This is a useful base class for multi-stage training, where some parts
    of the model should be kept fixed during certain training phases.
    """

    is_static: eqx.AbstractVar[bool]
    """Whether the class instance should be treated as static by the
    utility function `partition_trainable_and_static`, i.e., whether its parameters
    should be considered frozen in training."""

    def __check_init__(self):
        assert isinstance(self.is_static, bool)


class NonTrainableModule(eqx.Module):
    """A non-trainable module.

    This is a base class for modules that do not have any trainable parameters.
    It can be used to define fixed transformations or computations that are part
    of the model architecture but do not require training.
    """

    @property
    def is_static(cls) -> bool:
        return True


class StandardScaler(NonTrainableModule):
    mean: jax.Array
    std: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / self.std


class StandardScalerWrapper(eqx.Module):
    """A wrapper around a model that standardizes its inputs.

    TODO: this could also be extended to check dynamic range and auto-log-transform some variables
    """

    scaler: StandardScaler
    model: ConstituentModel

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(self.scaler(x))

    @classmethod
    def build(cls, model: ConstituentModel, data: jax.Array):
        mean = jnp.mean(data, axis=0)
        std = jnp.std(data, axis=0)
        return cls(scaler=StandardScaler(mean, std), model=model)


class Normalization(NonTrainableModule):
    """Abstract normalization module."""

    @abstractmethod
    def __call__(self, params: jax.Array) -> jax.Array:
        """Return normalization for given parameters (scalar)."""
        raise NotImplementedError


class QuadraticFormNormalization(Normalization):
    """Normalization for a quadratic form in parameters.

    This is used to compute the normalization of the event weights for a dataset,
    so that the likelihood can be computed correctly.
    """

    sqrtcoef: jax.Array
    r"""Square root matrix of the form, shape (p, p) where p is the parameter dimension.
    
    $B$ such that $B B^T = A$, and $\theta^T A theta$ is the quadratic form).
    Stored as the eigendecomposition square root (i.e. for $A = U D U^T$, then $B = U sqrt(D)$).
    TODO: should this be Cholesky or other decomposition instead?
    """

    def __call__(self, params: jax.Array) -> jax.Array:
        """Return normalization for given parameters (scalar)."""
        pc = params @ self.sqrtcoef
        return jnp.vecdot(pc, pc)

    @classmethod
    def from_coefficients(cls, coeffs: jax.Array):
        """Build decomposition from lower triangular coefficients."""
        eigvals, eigvecs = jnp.linalg.eigh(tril_to_matrix(coeffs))
        if jnp.all(abs(eigvals) < 1e-8):
            raise ValueError("Coefficient matrix is zero.")
        cut = eigvals < -1e-8
        if jnp.any(cut):
            msg = f"Coefficient matrix is not positive semi-definite, negative eigenvalues: {eigvals[cut]}"
            warnings.warn(msg, stacklevel=2)
        sqrt_eigvals = jnp.sqrt(jnp.clip(eigvals, min=0.0))
        sqrtcoef = eigvecs @ jnp.diag(sqrt_eigvals)
        return cls(sqrtcoef=sqrtcoef)
