import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from event2vec.analysis import run_analysis, score_closure
from event2vec.datasets.hzzdataset import FEATURE_SETS, HZZDataset, HZZLoader
from event2vec.experiment import ExperimentConfig, run_experiment
from event2vec.loss import BCELoss, BinarySampledParamLoss
from event2vec.losses.psd_matrix_losses import FrobeniusNormLoss
from event2vec.models.carl import CARLMLPConfig, CARLQuadraticFormMLPConfig
from event2vec.models.vecdot import E2VMLPConfig
from event2vec.prior import SMPlusUniformParameterPrior, UncorrelatedJointPrior
from event2vec.training import TrainingConfig


@dataclass
class HZZ(ExperimentConfig):
    """ALICE, ALICES or matrix regression on H -> ZZ -> 2e2mu events with MELA weights, one BSM coupling.

    Example: ``e2vrun -o runs/hzz HZZ --data-path training_split-0.root --loss alices --model psd``
    (``uproot`` is needed to read the file: ``uv run --with uproot e2vrun ...``).
    """

    data_factory: HZZLoader
    model_config: E2VMLPConfig | CARLQuadraticFormMLPConfig | CARLMLPConfig
    train_config: TrainingConfig
    key: PRNGKeyArray
    study_points: dict[str, jax.Array]

    @classmethod
    def register_parser(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--data-path",
            type=Path,
            required=True,
            help="ROOT file with the HZZ event tree.",
        )
        parser.add_argument(
            "--coupling",
            default="ghz4",
            help="BSM coupling stem of the MELA weight columns. (default: %(default)s)",
        )
        parser.add_argument(
            "--features",
            choices=FEATURE_SETS,
            default="invariants",
            help="angles: the 7 decay angles and Z masses; fourvec: the 16 lab-frame lepton components; "
            "invariants: fourvec + 6 pairwise invariant masses + the parity-odd triple product. (default: %(default)s)",
        )
        parser.add_argument(
            "--per-block",
            type=int,
            default=150_000,
            help="Entries read from each generation block of the file. (default: %(default)s)",
        )
        parser.add_argument(
            "--model",
            choices=["vec", "psd", "quad"],
            default="vec",
            help="vec: VecDotLLR; psd: low-rank PSD quadratic form; quad: free quadratic form. (default: %(default)s)",
        )
        parser.add_argument(
            "--loss",
            choices=["alice", "alices", "frobenius"],
            default="alice",
            help="alice: continuous-label BCE; alices: + score matching; frobenius: matrix regression (psd only). (default: %(default)s)",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.1,
            help="Score-term weight for alices. (default: %(default)s)",
        )
        parser.add_argument(
            "--box",
            type=float,
            default=1.0,
            help="The coupling is sampled uniformly in [-box, box] during training. (default: %(default)s)",
        )
        parser.add_argument(
            "--floor",
            type=float,
            default=1e-3,
            help="Floor of the learned ratio for the psd model; bounds its score, which the alices loss needs. (default: %(default)s)",
        )
        parser.add_argument(
            "--summary-dim",
            type=int,
            default=8,
            help="Dimensionality of the summary representation (vec model). (default: %(default)s)",
        )
        parser.add_argument(
            "--hidden",
            type=int,
            default=64,
            help="Width of the MLP hidden layers. (default: %(default)s)",
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=3,
            help="Number of MLP hidden layers. (default: %(default)s)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help="Number of training epochs. (default: %(default)s)",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help="Early stopping patience in epochs (default: none, train for --epochs).",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=4096,
            help="Training batch size. (default: %(default)s)",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=1e-3,
            help="Learning rate. (default: %(default)s)",
        )
        parser.add_argument(
            "--val-fraction",
            type=float,
            default=0.1,
            help="Fraction of the events used for validation. (default: %(default)s)",
        )
        parser.add_argument(
            "--test-fraction",
            type=float,
            default=0.1,
            help="Fraction of the events used for testing. (default: %(default)s)",
        )
        parser.add_argument(
            "--study-point",
            type=float,
            default=0.5,
            help="Coupling value of the BSM study point for the analysis plots. (default: %(default)s)",
        )
        parser.add_argument(
            "--key",
            type=int,
            default=42,
            help="Random seed key for JAX. (default: %(default)s)",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> Self:
        if args.loss == "frobenius" and args.model != "psd":
            raise ValueError("the frobenius loss needs --model psd")
        model_config: E2VMLPConfig | CARLQuadraticFormMLPConfig | CARLMLPConfig
        if args.model == "vec":
            model_config = E2VMLPConfig(
                summary_dim=args.summary_dim,
                hidden_size=args.hidden,
                depth=args.depth,
                standard_scaler=True,
            )
        elif args.model == "psd":
            model_config = CARLQuadraticFormMLPConfig(
                hidden_size=args.hidden,
                depth=args.depth,
                rank=2,
                standard_scaler=True,
                floor=args.floor,
            )
        else:
            model_config = CARLMLPConfig(
                hidden_size=args.hidden,
                depth=args.depth,
                quadratic=True,
                standard_scaler=True,
            )
        loss_fn: FrobeniusNormLoss | BinarySampledParamLoss
        if args.loss == "frobenius":
            loss_fn = FrobeniusNormLoss()
        else:
            prior = SMPlusUniformParameterPrior(
                low=jnp.array([-args.box]), high=jnp.array([args.box])
            )
            loss_fn = BinarySampledParamLoss(
                parameter_prior=UncorrelatedJointPrior(prior),
                continuous_labels=True,
                elementwise_loss=BCELoss(),
                score_weight=args.alpha if args.loss == "alices" else 0.0,
            )
        return cls(
            data_factory=HZZLoader(
                data_path=args.data_path.resolve(),
                coupling=args.coupling,
                features=args.features,
                per_block=args.per_block,
            ),
            model_config=model_config,
            train_config=TrainingConfig(
                train_fraction=1.0 - args.val_fraction - args.test_fraction,
                val_fraction=args.val_fraction,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                loss_fn=loss_fn,
                patience=args.patience,
            ),
            key=jax.random.PRNGKey(args.key),
            study_points={
                "SM": jnp.array([1.0, 0.0]),
                f"{args.coupling}_{args.study_point:g}".replace("-", "m").replace(
                    ".", "p"
                ): jnp.array([1.0, args.study_point]),
            },
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
            study_points=self.study_points,
            output_dir=output_dir,
        )
        assert isinstance(data_test, HZZDataset)
        closure = score_closure(
            model, data_test, data_test.gen_parameters, output_dir=output_dir
        )
        coupling = self.data_factory.coupling
        summary = {
            "coupling": coupling,
            "features": self.data_factory.features,
            "n_test": len(data_test),
            "epochs_trained": len(metrics.val_loss),
            "val_loss_best": min(metrics.val_loss),
            "test_loss": metrics.test_loss,
            # component 1 is the BSM coupling; component 0 (cSM) carries no information at the SM
            **{k: float(v[1]) for k, v in closure.items()},
        }
        (output_dir / "hzz_summary.json").write_text(json.dumps(summary, indent=1))
        print(
            f"HZZ {coupling}: score correlation {summary['score_corr']:.3f}, "
            f"information retained {summary['information_retained']:.3f}, "
            f"sigma({coupling}) per 1000 SM events {summary['sigma_learned'] / 1000**0.5:.4f} "
            f"(exact {summary['sigma_exact'] / 1000**0.5:.4f})"
        )
