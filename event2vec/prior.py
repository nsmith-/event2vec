from abc import abstractmethod

import equinox as eqx
import jax


class ParameterPrior(eqx.Module):
    """Abstract class for parameter priors.

    This class defines the interface for parameter priors, which can be used to sample
    parameters for the model.
    """

    @abstractmethod
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample parameters from the prior distribution."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class DirichletParameterPrior(ParameterPrior):
    """A simple prior for the parameters of the toy model.
    This prior is a Dirichlet distribution with fixed alpha parameters.
    """

    alpha: jax.Array

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.dirichlet(key, alpha=self.alpha)


class NormalParameterPrior(ParameterPrior):
    """A simple prior for parameter
    This prior is a Normal distribution with fixed mean and standard deviation.
    """

    mean: jax.Array
    cov: jax.Array

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.multivariate_normal(key, mean=self.mean, cov=self.cov)
