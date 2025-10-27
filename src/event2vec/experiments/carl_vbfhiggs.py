from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from event2vec.analysis import run_analysis
from event2vec.datasets import VBFHDataset
from event2vec.experiment import ExperimentConfig, run_experiment
from event2vec.loss import BCELoss
from event2vec.model import CARLQuadraticFormMLPConfig
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
    train_config: TrainingConfig
    key: jax.Array
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
        prior = SMPlusNormalParameterPrior(
            mean=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            cov=jnp.diag(jnp.array([1.0e1, 1.0e1, 1.0e-1, 1.0e-1, 1.0e0]) * 10),
        )
        run_key, *points_keys = jax.random.split(jax.random.PRNGKey(args.key), 4)

        points = {
            "SM": jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "rnd0": prior.sample(points_keys[0]),
            "rnd1": prior.sample(points_keys[1]),
            "rnd2": prior.sample(points_keys[2]),
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
                loss_fn=BCELoss(
                    parameter_prior=UncorrelatedJointPrior(prior),
                    continuous_labels=True,
                ),
            ),
            key=run_key,
            study_points=points,
        )

    def run(self, output_dir: Path):
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
