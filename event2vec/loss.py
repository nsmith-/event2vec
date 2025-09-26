from abc import abstractmethod
import jax
import equinox as eqx
import optax

from event2vec.dataset import ReweightableDataset
from event2vec.model import LearnedLLR
from event2vec.prior import JointParameterPrior


class Loss(eqx.Module):
    """Abstract class for loss functions."""

    parameter_prior: JointParameterPrior

    def __call__(
        self, model: LearnedLLR, data: ReweightableDataset, *, key: jax.Array
    ) -> jax.Array:
        """Compute the loss, sampling parameters from the prior."""
        param_0, param_1 = jax.vmap(self.parameter_prior.sample)(
            key=jax.random.split(key, len(data))
        ) # shapes: (batch, 1)
        
        llr_pred = jax.vmap(model.llr_pred)(
            data.observables, param_0, param_1
        ) # shape: (batch, 1) or (batch, B)
        
        llr_prob = jax.vmap(model.llr_prob)(
            data.observables, param_0, param_1
        ) # None or shape: (batch, B)
        
        weight_param_0 = data.weight(param_0) # shape: (batch, 1)
        weight_param_1 = data.weight(param_1) # shape: (batch, 1)
        
        sample_weight = (weight_param_0 + weight_param_1)/2
        target_label = weight_param_1 / (weight_param_0 + weight_param_1)
        
        loss = self._elementwise_loss(llr_pred, target_label)
        
        if llr_prob is not None:
            loss = (loss*llr_prob).sum(axis=-1, keepdims=True)
        
        return (loss*sample_weight).mean() / sample_weight.mean()

    @abstractmethod
    def _elementwise_loss(
        self,
        llr_pred: jax.Array,
        target_label: jax.Array
    ) -> jax.Array:
        """Compute the loss for the model's learned log-likelihood ratio on the given dataset.

        Does not reduce the loss over the dataset, this is done in the `__call__` method.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class BCELoss(Loss):
    def _elementwise_loss(self, llr_pred, target_label):
        return optax.losses.sigmoid_binary_cross_entropy(
            logits=llr_pred, labels=target_label
        )

class MSELoss(Loss):
    def _elementwise_loss(self, llr_pred, target_label):
        return (jax.nn.sigmoid(llr_pred) - target_label)**2
