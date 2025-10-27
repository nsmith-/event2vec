from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

# TODO: use regular dataclasses rather than equinox for non-neural-net modules?
# This would fix the non-faithful repr issue when printing these objects.


class ParameterPrior(eqx.Module):
    """Abstract class for parameter priors.

    This class defines the interface for parameter priors, which can be used to sample
    parameters for the model.
    """

    @abstractmethod
    def sample(self, key: jax.Array) -> jax.Array:
        """Sample parameters from the prior distribution."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class JointParameterPrior(eqx.Module):
    """Abstract joint prior over two parameter points

    This is used for training
    """

    @abstractmethod
    def sample(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample two parameter points from the joint prior distribution."""
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


class SMPlusNormalParameterPrior(ParameterPrior):
    """A prior that fixes the first parameter (the SM parameter) to 1 and samples the rest from a Normal distribution."""

    mean: jax.Array
    "means for all parameters except the first"
    cov: jax.Array
    "covariance for all parameters except the first"

    def sample(self, key: jax.Array) -> jax.Array:
        return (
            jnp.ones(shape=(len(self.mean) + 1,))
            .at[1:]
            .set(jax.random.multivariate_normal(key, mean=self.mean, cov=self.cov))
        )


class UncorrelatedJointPrior(JointParameterPrior):
    """A joint prior that samples two independent parameters from the same prior."""

    prior: ParameterPrior

    def sample(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key1, key2 = jax.random.split(key)
        return self.prior.sample(key1), self.prior.sample(key2)
