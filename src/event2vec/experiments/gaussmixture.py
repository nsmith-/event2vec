from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from event2vec.analysis import run_analysis
from event2vec.datasets.gaussmixture import (
    GaussMixtureDataset,
    GaussMixtureDatasetFactory,
)
from event2vec.experiment import ExperimentConfig, run_experiment
from event2vec.loss import (
    BCELoss,
    BinarySampledParamBinwiseLoss,
    BinarySampledParamLoss,
    MSELoss,
)
from event2vec.models.vecdot import E2VMLPConfig, VecDotLLR
from event2vec.prior import DirichletParameterPrior, UncorrelatedJointPrior
from event2vec.training import TrainingConfig


@dataclass
class GaussianMixture(ExperimentConfig):
    """Experiment configuration for Gaussian Mixture dataset."""

    data_factory: GaussMixtureDatasetFactory
    model_config: E2VMLPConfig
    train_config: TrainingConfig[VecDotLLR, GaussMixtureDataset]
    key: PRNGKeyArray

    @classmethod
    def register_parser(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--key",
            type=int,
            default=42,
            help="Random seed key for JAX. (default: %(default)s)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1_000,
            help="Number of training epochs. (default: %(default)s)",
        )
        parser.add_argument(
            "--loss",
            type=str,
            choices=["mse", "bce"],
            default="mse",
            help="Loss function to use. (default: %(default)s)",
        )
        parser.add_argument(
            "--binwise",
            action="store_true",
            help="Use binwise loss instead of standard loss.",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "GaussianMixture":
        gen_param_prior = DirichletParameterPrior(alpha=jnp.array([9.0, 3.0, 3.0]))
        joint_prior = UncorrelatedJointPrior(gen_param_prior)
        data_factory = GaussMixtureDatasetFactory(
            len=200_000,
            param_prior=gen_param_prior,
        )
        model_config = E2VMLPConfig(
            summary_dim=2,
            hidden_size=16,
            depth=3,
            standard_scaler=True,
            bin_probabilities=args.binwise,
        )
        elementwise_loss = MSELoss() if args.loss == "mse" else BCELoss()
        loss_fn = (
            BinarySampledParamBinwiseLoss(
                parameter_prior=joint_prior,
                continuous_labels=True,
                elementwise_loss=elementwise_loss,
            )
            if args.binwise
            else BinarySampledParamLoss(
                parameter_prior=joint_prior,
                continuous_labels=True,
                elementwise_loss=elementwise_loss,
            )
        )
        train_config = TrainingConfig(
            train_fraction=0.8,
            val_fraction=0.1,
            batch_size=128,
            learning_rate=0.005,
            epochs=args.epochs,
            loss_fn=loss_fn,
        )
        return cls(
            data_factory=data_factory,
            model_config=model_config,
            train_config=train_config,
            key=jax.random.PRNGKey(args.key),
        )

    def run(self, output_dir: Path) -> None:
        model, data_test, metrics = run_experiment(
            self.data_factory,
            self.model_config,
            self.train_config,
            key=self.key,
        )
        with open(output_dir / "model.eqx", "wb") as fout:
            eqx.tree_serialise_leaves(fout, model)
        run_analysis(
            model=model,
            data=data_test,
            metrics=metrics,
            study_points={
                "gen_mean": jnp.array([9.0, 3.0, 3.0]) / 15.0,
                "SM": jnp.array([1.0, 0.0, 0.0]),
                "bsm1": jnp.array([0.8, 0.2, 0.0]),
                "bsm2": jnp.array([0.8, 0.0, 0.2]),
            },
            output_dir=output_dir,
        )
