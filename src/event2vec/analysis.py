"""Standard analysis routines for trained models."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from event2vec.dataset import ReweightableDataset
from event2vec.model import AbstractLLR
from event2vec.models.vecdot import VecDotLLR
from event2vec.training import MetricsHistory
from event2vec.util import standard_pbar


def plot_loss(ax: Axes, loss_train: list[float], loss_test: list[float]) -> None:
    """Plot training and testing loss over epochs."""
    ax.plot(loss_train, label="Training Loss")
    ax.plot(loss_test, label="Testing Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()


def plot_llr_distribution(ax: Axes, llr_pred: jax.Array, llr_true: jax.Array):
    amin = min(jnp.min(llr_pred).item(), jnp.min(llr_true).item())
    amax = max(jnp.max(llr_pred).item(), jnp.max(llr_true).item())
    ax.set_xlim(amin, amax)
    ax.set_ylim(amin, amax)
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", transform=ax.transAxes)
    ax.scatter(llr_pred, llr_true, s=1)
    ax.set_xlabel("Predicted LLR")
    ax.set_ylabel("True LLR")


def plot_lr_mean(
    ax: Axes,
    llr_pred: jax.Array,
    llr_true: jax.Array,
    bins: int = 20,
    qthreshold: float = 0.01,
):
    """Plot the mean and std of the true likelihood ratio in quantiles of the predicted likelihood ratio.

    If the predicted likelihood ratio is perfect, the mean should lie on the y=x line and the std should be zero.
    In any case, the mean of the true likelihood ratio should be close to the predicted likelihood ratio.
    """
    lr_pred = jnp.exp(llr_pred)
    lr_true = jnp.exp(llr_true)

    qbins = jnp.quantile(lr_pred, jnp.linspace(qthreshold, 1 - qthreshold, bins + 1))

    sumc, _ = jnp.histogram(lr_pred, bins=qbins)
    sumw, _ = jnp.histogram(lr_pred, bins=qbins, weights=lr_true)
    sumw2, _ = jnp.histogram(lr_pred, bins=qbins, weights=lr_true**2)

    mean = sumw / sumc
    std = jnp.sqrt(sumw2 / sumc - (sumw / sumc) ** 2)

    ax.errorbar(
        0.5 * (qbins[1:] + qbins[:-1]),
        mean,
        xerr=0.5 * (qbins[1:] - qbins[:-1]),
        yerr=std,
        fmt="o",
        markersize=5,
        capsize=3,
    )
    lo, hi = qbins[0], qbins[-1]
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--")

    ax.set_xlabel("Predicted LR")
    ax.set_ylabel("Mean true LR")
    ax.set_aspect("equal")


def plot_observable(
    ax: Axes,
    observable: jax.Array,
    weight0: jax.Array,
    weight1: jax.Array,
    llr_pred: jax.Array,
    bins: int = 20,
):
    """Plot the distribution of an observable under different weights, including reweighted using the learned log-likelihood ratio.

    Assumes llr_pred is the log-likelihood ratio for theta1 vs theta0.
    """
    counts_unweighted, bins = jnp.histogram(observable, bins=bins)  # type: ignore[assignment]
    counts_w0, _ = jnp.histogram(observable, weights=weight0, bins=bins)
    counts_w1, _ = jnp.histogram(observable, weights=weight1, bins=bins)
    counts_rw, _ = jnp.histogram(
        observable, weights=jnp.exp(llr_pred) * weight0, bins=bins
    )

    ax.stairs(counts_unweighted, bins, label="Unweighted", color="k", linestyle="--")
    ax.stairs(counts_w0, bins, label=r"Weighted $\theta_0$", color="C0")
    ax.stairs(counts_w1, bins, label=r"Weighted $\theta_1$", color="C1")
    ax.stairs(
        counts_rw, bins, label=r"Learned weight $\theta_0 \to \theta_1$", color="C2"
    )
    ax.legend()


def plot_vecfield(ax: Axes, model: VecDotLLR, observables: jax.Array):
    """Plot the vector field of the event summary model over 2D observables.

    Args:
        ax: The matplotlib Axes to plot on.
        model: The trained RegularVector_LearnedLLR model.
        observables: A (N, 2) array of observables to define the plotting range.
    """
    x = jnp.linspace(jnp.min(observables[:, 0]), jnp.max(observables[:, 0]), 20)
    y = jnp.linspace(jnp.min(observables[:, 1]), jnp.max(observables[:, 1]), 20)
    X, Y = jnp.meshgrid(x, y)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    summaries = jax.vmap(model.event_summary)(grid_points)
    U = summaries[:, 0].reshape(X.shape)
    V = summaries[:, 1].reshape(Y.shape)

    ax.quiver(X, Y, U, V)
    ax.set_xlabel("Observable 0")
    ax.set_ylabel("Observable 1")


def dump_summary_plots(
    model: VecDotLLR,
    observables: jax.Array,
    output_dir: Path,
) -> None:
    """Dump summary plots for each dimension of the event summary.

    Args:
        model: The trained VecDotLLR model.
        observables: A (N, 2) array of observables to define the plotting range.
        output_dir: The directory to save the plots in.
    """
    xmin, xmax = jnp.min(observables[:, 0]), jnp.max(observables[:, 0])
    ymin, ymax = jnp.min(observables[:, 1]), jnp.max(observables[:, 1])
    # Make a grid of centers
    x = jnp.linspace(xmin, xmax, 100, endpoint=False) + (xmax - xmin) / 200
    y = jnp.linspace(ymin, ymax, 100, endpoint=False) + (ymax - ymin) / 200
    X, Y = jnp.meshgrid(x, y)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    summaries = jax.vmap(model.event_summary)(grid_points).reshape(X.shape + (-1,))
    k = summaries.shape[2]
    fig, axes = plt.subplots(ncols=k, figsize=(4 * k, 4))
    for i in range(k):
        ax = axes[i] if k > 1 else axes
        ax.imshow(
            summaries[:, :, i],
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            aspect="auto",
        )
        ax.set_title(f"Summary dimension {i}")
        ax.set_xlabel("Observable 0")
        ax.set_ylabel("Observable 1")

    fig.tight_layout()
    fig.savefig(output_dir / "summary_fields.png")


def study_point_analysis(
    model: AbstractLLR,
    data: ReweightableDataset,
    param_1: jax.Array,
    param_0: jax.Array,
    output_dir: Path,
) -> None:
    """Run analysis for a specific pair of parameter points."""
    output_dir.mkdir(parents=True, exist_ok=True)

    llr_pred = jax.vmap(model.llr_pred, in_axes=(0, None, None))(
        data.observables, param_0, param_1
    )
    weight0 = data.weight(param_0)
    weight1 = data.weight(param_1)
    llr_true = jnp.log(weight1) - jnp.log(weight0)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 5))
    plot_llr_distribution(axl, llr_pred, llr_true)
    plot_lr_mean(axr, llr_pred, llr_true)
    fig.savefig(output_dir / "llr_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for i in range(data.observables.shape[1]):
        plot_observable(
            ax,
            data.observables[:, i],
            weight0,
            weight1,
            llr_pred,
        )
        ax.set_title(f"Observable {i}")
        fig.savefig(output_dir / f"observable_{i}.png")
        ax.set_yscale("log")
        fig.savefig(output_dir / f"observable_{i}_log.png")
        ax.set_yscale("linear")
        ax.clear()
    plt.close(fig)


def run_analysis(
    model: AbstractLLR,
    data: ReweightableDataset,
    metrics: MetricsHistory,
    study_points: dict[str, jax.Array],
    output_dir: Path,
) -> None:
    """Run standard analysis routines on a trained model"""
    output_dir.mkdir(parents=True, exist_ok=True)

    nplots = 2 + len(study_points) * (len(study_points) - 1)

    with standard_pbar() as progress:
        analysis_task = progress.add_task("Running analysis...", total=nplots)

        fig, ax = plt.subplots()
        plot_loss(ax, metrics.train_loss, metrics.test_loss)
        fig.savefig(output_dir / "loss.png")
        ax.set_yscale("log")
        fig.savefig(output_dir / "loss_log.png")
        plt.close(fig)
        progress.advance(analysis_task)

        if isinstance(model, VecDotLLR) and data.observable_dim == 2:
            fig, ax = plt.subplots()
            plot_vecfield(ax, model, data.observables)
            fig.savefig(output_dir / "vector_field.png")
            plt.close(fig)
            dump_summary_plots(model, data.observables, output_dir)
        progress.advance(analysis_task)

        for p0name, param_0 in study_points.items():
            p0dir = output_dir / f"den_{p0name}"
            for p1name, param_1 in study_points.items():
                if p0name == p1name:
                    continue
                study_point_analysis(
                    model=model,
                    data=data,
                    param_1=param_1,
                    param_0=param_0,
                    output_dir=p0dir / f"num_{p1name}",
                )
                progress.advance(analysis_task)
