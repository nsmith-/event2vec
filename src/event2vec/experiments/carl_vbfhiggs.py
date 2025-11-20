from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from event2vec.analysis import run_analysis
from event2vec.datasets import VBFHDataset
from event2vec.experiment import ExperimentConfig, run_experiment
from event2vec.loss import (
    BCELoss,
    BinarySampledParamLoss,
)
from event2vec.models.carl import CARLPSDMatrixLLR, CARLQuadraticFormMLPConfig
from event2vec.prior import SMPlusNormalParameterPrior, UncorrelatedJointPrior
from event2vec.training import MetricsHistory, TrainingConfig


@dataclass
class VBFHLoader:
    data_path: Path

    def __call__(self, *, key) -> VBFHDataset:
        dataset = VBFHDataset.from_lhe(str(self.data_path))
        return dataset


@dataclass
class CARLVBFHiggs(ExperimentConfig):
    """Experiment configuration for CARL on VBF Higgs dataset."""

    data_factory: VBFHLoader
    model_config: CARLQuadraticFormMLPConfig
    train_config: TrainingConfig[CARLPSDMatrixLLR, VBFHDataset]
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

    @classmethod
    def from_args(cls, args: Namespace) -> "CARLVBFHiggs":
        # based on a rough translation of HIG-21-018 uncorrelated uncertainties
        std = jnp.array([0.5, 2.0, 0.005, 0.002, 0.003]) * 10
        prior = SMPlusNormalParameterPrior(
            mean=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            cov=jnp.diag(std**2),
        )
        run_key = jax.random.PRNGKey(args.key)

        points = {
            # cSM, cHbox, cHDD, cHW, cHB, cHWB
            "SM": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "cHW2em3": jnp.array([1.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0]),
            "cHB5em3": jnp.array([1.0, 0.0, 0.0, 0.0, 5.0e-3, 0.0]),
            "cHWB3em3": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 3.0e-3]),
            "cHbox0p5": jnp.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
            "cHDD2p0": jnp.array([1.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        }
        return cls(
            data_factory=VBFHLoader(data_path=args.data_path.resolve()),
            model_config=CARLQuadraticFormMLPConfig(
                hidden_size=64,
                depth=3,
                rank=3,
                standard_scaler=True,
            ),
            train_config=TrainingConfig(
                test_fraction=0.1,
                batch_size=128,
                learning_rate=0.001,
                epochs=args.epochs,
                loss_fn=BinarySampledParamLoss(
                    parameter_prior=UncorrelatedJointPrior(prior),
                    continuous_labels=True,
                    elementwise_loss=BCELoss(),
                ),
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
