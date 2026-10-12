"""Standard analysis routines for trained models."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from event2vec.dataset import ReweightableDataset
from event2vec.model import AbstractLLR
from event2vec.models.vecdot import VecDotLLR
from event2vec.shapes import ParamVec
from event2vec.training import MetricsHistory
from event2vec.util import standard_pbar


def plot_loss(
    ax: Axes,
    loss_train: list[float],
    loss_val: list[float] | None = None,
    loss_test: float | None = None,
) -> None:
    """Plot training, validation, and testing loss over epochs."""
    ax.plot(loss_train, label="Training Loss")
    if loss_val is not None:
        ax.plot(loss_val, label="Validation Loss")
    if loss_test is not None:
        ax.axhline(loss_test, linestyle="--", label="Testing Loss")
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
    weight0: jax.Array,
    bins: int = 20,
    qthreshold: float = 0.01,
):
    """Plot the mean and std of the true likelihood ratio in quantiles of the predicted likelihood ratio.

    If the predicted likelihood ratio is perfect, the mean should lie on the y=x line and the std should be zero.
    In any case, the mean of the true likelihood ratio should be close to the predicted likelihood ratio.

    Args:
        ax: The matplotlib Axes to plot on.
        llr_pred: The predicted log-likelihood ratio phat(x;theta1)/phat(x;theta0) for each event.
        llr_true: The true log-likelihood ratio p(x,z;theta1)/p(x,z;theta0) for each event.
        weight0: The weights of the event sample under theta0, to handle the case the events are not sampled from p(x;theta0).
        bins: The number of quantile bins to use.
        qthreshold: The quantile threshold to exclude extreme outliers in the predicted likelihood ratio.
    """
    lr_pred = jnp.exp(llr_pred)
    lr_true = jnp.exp(llr_true)

    qbins = jnp.quantile(lr_pred, jnp.linspace(qthreshold, 1 - qthreshold, bins + 1))

    sumw, _ = jnp.histogram(lr_pred, bins=qbins, weights=weight0)
    sumwlr, _ = jnp.histogram(lr_pred, bins=qbins, weights=weight0 * lr_true)
    mean = sumwlr / sumw

    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_of_the_weighted_mean_(%CF%80-estimator_for_ratio-mean)
    sumw2lr2, _ = jnp.histogram(lr_pred, bins=qbins, weights=weight0**2 * lr_true**2)
    sumw2lr, _ = jnp.histogram(lr_pred, bins=qbins, weights=weight0**2 * lr_true)
    sumw2, _ = jnp.histogram(lr_pred, bins=qbins, weights=weight0**2)
    # weight0**2 * (lr_true - mean)**2
    # = weight0**2 * lr_true**2 - 2 * weight0**2 * lr_true * mean + weight0**2 * mean**2
    err_mean = jnp.sqrt(sumw2lr2 - 2 * sumw2lr * mean + sumw2 * mean**2) / sumw

    # Put x center in barycenter of bin
    sumxw, _ = jnp.histogram(lr_pred, bins=qbins, weights=weight0 * lr_pred)
    xcenter = sumxw / sumw
    xerr = abs(jnp.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    ax.errorbar(
        xcenter, mean, xerr=xerr, yerr=err_mean, fmt="o", markersize=5, capsize=3
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
    plot_lr_mean(axr, llr_pred, llr_true, weight0)
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
        plot_loss(ax, metrics.train_loss, metrics.val_loss, metrics.test_loss)
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


def score_closure(
    model: AbstractLLR,
    data: ReweightableDataset,
    param_ref: ParamVec,
    output_dir: Path | None = None,
    batch_size: int = 65536,
) -> dict[str, jax.Array]:
    r"""Compare the learned score with the exact score at a reference point, per parameter component.

    The score :math:`t(x) = \nabla_\theta \log p(x|\theta)` at :math:`\theta_{ref}` is the
    sufficient statistic for a measurement near the reference point. The learned score is
    :math:`\nabla_{\theta_1} \hat\ell(x, \theta_{ref}, \theta_1)` at :math:`\theta_1 = \theta_{ref}`;
    the exact one is the gradient of the log event weight, which equals the score of the observables
    when they determine the weight (no unobserved latent variables), as for generator-level samples.

    Per component, the Fisher information carried by the learned score (after linear calibration)
    is :math:`\mathrm{cov}(\hat t, t)^2 / \mathrm{var}(\hat t)`, and the exact information is
    :math:`\mathrm{var}(t)`, so their ratio is the squared weighted correlation. Expectations are
    taken with the event weights at the reference point.

    Args:
        model: The trained model.
        data: Dataset with exact event weights.
        param_ref: The reference parameter point.
        output_dir: If given, a ``score_closure.png`` scatter plot is written there.
        batch_size: Events per call when evaluating the model.

    Returns:
        Dictionary of per-component arrays: ``score_corr``, ``information_retained``
        (learned / exact Fisher information), ``fisher_exact`` and ``fisher_learned`` (per event),
        ``sigma_exact`` and ``sigma_learned`` (expected uncertainty per event, i.e. divide by
        :math:`\sqrt{N}`). Components without exact information are NaN.
    """
    n = len(data)
    weight = data.weight(param_ref)
    weight = weight / jnp.sum(weight)

    def log_weight_sum(params):
        return jnp.sum(jnp.log(data.weight(params)))

    score_exact = jax.grad(log_weight_sum)(
        jnp.broadcast_to(param_ref, (n, len(param_ref)))
    )

    @eqx.filter_jit
    def learned_score(m, observables):
        def per_event(x):
            return jax.grad(lambda p1: m.llr_pred(x, param_ref, p1))(param_ref)

        return jax.vmap(per_event)(observables)

    score_learned = jnp.concatenate(
        [
            learned_score(model, data.observables[i : i + batch_size])
            for i in range(0, n, batch_size)
        ]
    )

    def weighted_cov(a, b):
        return jnp.sum(weight[:, None] * (a - weight @ a) * (b - weight @ b), axis=0)

    var_exact = weighted_cov(score_exact, score_exact)
    var_learned = weighted_cov(score_learned, score_learned)
    cov = weighted_cov(score_learned, score_exact)
    # components with no exact information (e.g. a fixed SM normalisation) only carry rounding noise
    informative = var_exact > 1e-6 * jnp.max(var_exact)
    safe = jnp.where(informative & (var_learned > 0), var_learned * var_exact, 1.0)
    corr = jnp.where(informative, cov / jnp.sqrt(safe), jnp.nan)
    fisher_learned = jnp.where(
        informative, cov**2 / jnp.where(var_learned > 0, var_learned, 1.0), jnp.nan
    )
    fisher_exact = jnp.where(informative, var_exact, jnp.nan)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        components = [int(i) for i in jnp.flatnonzero(informative)]
        fig, axes = plt.subplots(
            ncols=max(len(components), 1),
            figsize=(5 * max(len(components), 1), 5),
            squeeze=False,
        )
        for ax, i in zip(axes[0], components):
            lim = jnp.quantile(score_exact[:, i], jnp.array([0.002, 0.998]))
            ax.hist2d(
                score_exact[:, i],
                score_learned[:, i],
                bins=80,
                range=[lim, lim],
                weights=weight,
                cmin=1e-12,
            )
            ax.plot(lim, lim, color="grey", linestyle="--")
            ax.set_xlabel(f"Exact score, parameter {i}")
            ax.set_ylabel("Learned score")
            ax.set_title(f"corr {corr[i]:.3f}, information retained {corr[i] ** 2:.3f}")
        fig.tight_layout()
        fig.savefig(output_dir / "score_closure.png")
        plt.close(fig)

    return {
        "score_corr": corr,
        "information_retained": fisher_learned / fisher_exact,
        "fisher_exact": fisher_exact,
        "fisher_learned": fisher_learned,
        "sigma_exact": 1 / jnp.sqrt(fisher_exact),
        "sigma_learned": 1 / jnp.sqrt(fisher_learned),
    }
