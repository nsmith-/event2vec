from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from event2vec.analysis import run_analysis
from event2vec.datasets import VBFHDataset
from event2vec.experiment import ExperimentConfig, run_experiment
from event2vec.experiments.carl_vbfhiggs import VBFHLoader
from event2vec.loss import (
    BCELoss,
    BinarySampledParamBinwiseLoss,
    BinarySampledParamLoss,
    MSELoss,
)
from event2vec.models.vecdot import E2VMLPConfig, VecDotLLR
from event2vec.prior import SMPlusNormalParameterPrior, UncorrelatedJointPrior
from event2vec.training import MetricsHistory, TrainingConfig


@dataclass
class VecVBFHiggs(ExperimentConfig):
    """Experiment configuration for event2vec on VBF Higgs dataset."""

    data_factory: VBFHLoader
    model_config: E2VMLPConfig
    train_config: TrainingConfig[VecDotLLR, VBFHDataset]
    key: PRNGKeyArray
    study_points: dict[str, jax.Array]

    @classmethod
    def register_parser(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--data-path",
            type=Path,
            required=True,
            help="Path to the VBF Higgs LHE files (supports glob patterns).",
        )
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
        parser.add_argument(
            "--summary-dim",
            type=int,
            default=18,  # same dimensionality as rank-3 6-parameter PSD model
            help="Dimensionality of the summary representation. (default: %(default)s)",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> Self:
        # based on a rough translation of HIG-21-018 uncorrelated uncertainties
        # cHbox, cHDD, cHW, cHB, cHWB
        std = jnp.array([0.5, 2.0, 0.005, 0.002, 0.003]) * 10
        prior = SMPlusNormalParameterPrior(
            mean=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            cov=jnp.diag(std**2),
        )
        run_key = jax.random.PRNGKey(args.key)

        points = {
            "SM": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "cHbox5p0": jnp.array([1.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
            "cHDD20p0": jnp.array([1.0, 0.0, 20.0, 0.0, 0.0, 0.0]),
            "cHW2em2": jnp.array([1.0, 0.0, 0.0, 2.0e-2, 0.0, 0.0]),
            "cHB5em2": jnp.array([1.0, 0.0, 0.0, 0.0, 5.0e-2, 0.0]),
            "cHWB3em2": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 3.0e-2]),
        }
        model_config = E2VMLPConfig(
            summary_dim=args.summary_dim,
            hidden_size=64,
            depth=3,
            standard_scaler=True,
            bin_probabilities=args.binwise,
        )
        if args.loss == "mse":
            elementwise_loss = MSELoss()
        elif args.loss == "bce":
            elementwise_loss = BCELoss()
        else:
            raise ValueError(f"Unknown loss function: {args.loss}")
        joint_prior = UncorrelatedJointPrior(prior)
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
        return cls(
            data_factory=VBFHLoader(data_path=args.data_path.resolve()),
            model_config=model_config,
            train_config=TrainingConfig(
                test_fraction=0.1,
                batch_size=128,
                learning_rate=0.001,
                epochs=args.epochs,
                loss_fn=loss_fn,
            ),
            key=run_key,
            study_points=points,
        )

    def run(self, output_dir: Path) -> None:
        model, data, loss_train, loss_test = run_experiment(
            self.data_factory,
            self.model_config,
            self.train_config,
            key=self.key,
        )
        with open(output_dir / "model.eqx", "wb") as fout:
            eqx.tree_serialise_leaves(fout, model)
        metrics = MetricsHistory(train_loss=loss_train, test_loss=loss_test)
        run_analysis(
            model=model,
            data=data,
            metrics=metrics,
            study_points=self.study_points,
            output_dir=output_dir,
        )
