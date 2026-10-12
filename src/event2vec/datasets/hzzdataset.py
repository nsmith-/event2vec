"""H -> ZZ -> 4l (2e2mu) events with MELA matrix-element weights, read from a ROOT tree.

Compared with the LHE datasets:

* the events are *weighted*: the stored SM weight ``p_ghz1_1_SMEFT`` is :math:`|M_{SM}|^2 / q_{gen}`
  with a generation density :math:`q_{gen}` that differs between the blocks of the file, so the
  sample is not an unweighted draw at the generation point. :meth:`HZZDataset.weight` includes this
  per-event base weight, so every consumer of ``weight`` (losses, analysis) sees it without change;
* the file is a concatenation of equal-size blocks generated under different hypotheses, so it is
  read *stratified* (the first ``per_block`` entries of each selected block) and shuffled once,
  because :meth:`event2vec.dataset.Dataset.split` takes head and tail;
* one BSM coupling at a time: the stored columns hold SM x BSM and BSM x BSM for single couplings only.

Reading the ROOT file needs ``uproot``, which is not a dependency of event2vec:
``uv run --with uproot e2vrun ...``.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from event2vec.dataset import QuadraticReweightableDataset
from event2vec.nontrainable import QuadraticFormNormalization
from event2vec.shapes import ParamVec
from event2vec.util import EPS, standard_pbar, tril_outer_product

ANGLES = ["costheta1d", "costheta2d", "Phid", "costhetastard", "Phi1d", "MZ1", "MZ2"]
"""The 7 sufficient statistics of the decay: 5 angles and the two Z masses."""
FOURVEC = [f"{c}dau{i}" for i in (1, 2, 3, 4) for c in ("px", "py", "pz", "E")]
"""Lab-frame four-momenta of the four leptons (16 numbers)."""
FEATURE_SETS = ("angles", "fourvec", "invariants")
"""Available observable sets, see :meth:`HZZDataset.from_root`."""
GOOD_BLOCKS = (0, 1, 2, 4, 6, 7, 8, 10, 12)
"""Blocks usable for SM-weighted work (the other four are photon-coupling natives with n_eff/n ~ 1e-4)."""
NUM_BLOCKS = 13
"""Number of equal-size generation blocks in a file."""


def lorentz_invariants(fourvec: np.ndarray) -> np.ndarray:
    r"""The 6 pairwise invariant masses squared and the parity-odd triple product.

    Args:
        fourvec: array (n, 16) of (px, py, pz, E) for the four leptons.

    Returns:
        array (n, 7): :math:`m_{ij}^2` for i < j, then :math:`\epsilon_{\mu\nu\rho\sigma} p_1 p_2 p_3 p_4`.

    These are what a network otherwise has to construct from the raw components before it can
    learn anything about the couplings: a CP-odd interference is a function of the triple product.
    """
    p = [fourvec[:, 4 * i : 4 * i + 4] for i in range(4)]

    def dot(a, b):
        return (
            a[:, 3] * b[:, 3]
            - a[:, 0] * b[:, 0]
            - a[:, 1] * b[:, 1]
            - a[:, 2] * b[:, 2]
        )

    m2 = [dot(p[i] + p[j], p[i] + p[j]) for i in range(4) for j in range(i + 1, 4)]
    # rows (E, px, py, pz): det = eps_{mu nu rho sigma} p1^mu p2^nu p3^rho p4^sigma
    mat = np.stack([np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], -1) for q in p], 1)
    return np.column_stack([*m2, np.linalg.det(mat)])


class HZZDataset(QuadraticReweightableDataset):
    """H -> ZZ -> 2e2mu events with a quadratic dependence on (cSM, g) for one BSM coupling g."""

    observables: jax.Array
    """Event features, shape (num_events, obs_dim)"""
    gen_parameters: jax.Array
    """The SM point (cSM=1, g=0); the quadratic form is stored relative to it"""
    latent_data: jax.Array
    """Quadratic-form coefficients relative to the SM entry, lower-triangular order
    (SM, SM x g interference, g x g), shape (num_events, 3)"""
    base_weight: jax.Array
    """Per-event SM weight relative to the generated sample, normalised to mean 1, shape (num_events,)"""
    normalization: QuadraticFormNormalization
    """The normalization of the events"""
    extended_likelihood: bool = False
    """Whether to use the extended likelihood (i.e. include the overall normalization shift in the weight)"""

    @property
    def observable_dim(self) -> int:
        return self.observables.shape[1]

    @property
    def parameter_dim(self) -> int:
        return self.gen_parameters.shape[0]

    @property
    def quadratic_form(self):
        return self.latent_data

    def likelihood(self, param: jax.Array) -> jax.Array:
        weights = jnp.vecdot(self.latent_data, tril_outer_product(param))
        if self.extended_likelihood:
            return jnp.maximum(weights, EPS)
        return jnp.maximum(weights / self.normalization(param), EPS)

    def weight(self, param: ParamVec) -> Float[Array, " batch"]:
        """Weight of the dataset w.r.t. the given parameter vector.

        The events were not sampled at ``gen_parameters`` but from a generation density, so the
        weight relative to the sample is the base weight times the likelihood ratio to the SM point.
        """
        return self.base_weight * super().weight(param)

    @classmethod
    def from_root(
        cls,
        path: str,
        *,
        key: PRNGKeyArray,
        coupling: str = "ghz4",
        features: str = "invariants",
        per_block: int = 150_000,
        blocks: tuple[int, ...] = GOOD_BLOCKS,
        tree: str = "eventTree",
        extended_likelihood: bool = False,
    ):
        """Load from a ROOT file, block by block (the file is never read whole).

        Args:
            path: ROOT file with the event tree.
            key: random key used to shuffle the events.
            coupling: BSM coupling stem of the MELA columns, e.g. "ghz4" (CP-odd) or "ghz2".
            features: "fourvec" (16 lab-frame lepton components), "invariants"
                (fourvec + 6 m_ij^2 + the triple product) or "angles" (the 7 sufficient statistics).
            per_block: entries read from the start of each selected block.
            blocks: generation blocks to use.
            tree: name of the tree in the file.
            extended_likelihood: see :attr:`extended_likelihood`.
        """
        try:
            import uproot  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "HZZDataset.from_root needs uproot (e.g. `uv run --with uproot ...`)"
            ) from err
        if features not in FEATURE_SETS:
            raise ValueError(
                f"features must be one of {FEATURE_SETS}, got {features!r}"
            )

        p_sm = "p_ghz1_1_SMEFT"
        p_bsm = f"p_{coupling}_1_SMEFT"
        p_mix = f"p_ghz1_1_{coupling}_1_SMEFT"
        flavours = [f"flavdau{i}" for i in (1, 2, 3, 4)]
        columns = ANGLES if features == "angles" else FOURVEC
        observables, coefficients = [], []
        with uproot.open(path) as fin, standard_pbar() as progress:
            events = fin[tree]
            block_size = events.num_entries // NUM_BLOCKS
            for block in progress.track(blocks, description="Loading HZZ blocks..."):
                arrays = events.arrays(
                    columns + [p_sm, p_bsm, p_mix] + flavours,
                    entry_start=block * block_size,
                    entry_stop=block * block_size + min(per_block, block_size),
                    library="np",
                )
                # 2e2mu: the product of the four |pdg ids| is 11*11*13*13
                keep = np.prod([np.abs(arrays[f]) for f in flavours], axis=0) == 20449
                w_sm, w_bsm, w_mix = (
                    np.asarray(arrays[c], np.float64) for c in (p_sm, p_bsm, p_mix)
                )
                keep &= np.isfinite(w_sm) & np.isfinite(w_bsm) & np.isfinite(w_mix)
                keep &= w_sm > 0
                x = np.column_stack([arrays[c][keep] for c in columns]).astype(
                    np.float64
                )
                if features == "invariants":
                    x = np.column_stack([x, lorentz_invariants(x)])
                # (SM, SM x g, g x g) in tril order, normalised per block to mean SM weight 1
                k = np.column_stack(
                    [w_sm[keep], w_mix[keep] - w_sm[keep] - w_bsm[keep], w_bsm[keep]]
                )
                k /= w_sm[keep].mean()
                observables.append(x)
                coefficients.append(k)
        x = np.concatenate(observables)
        k = np.concatenate(coefficients)
        perm = np.asarray(jax.random.permutation(key, len(x)))
        x, k = x[perm], k[perm]
        base_weight = k[:, 0]
        n_eff = base_weight.sum() ** 2 / (base_weight**2).sum()
        print(
            f"HZZ: {len(x):,} 2e2mu events from {len(blocks)} blocks, n_eff/n = {n_eff / len(x):.3f}"
        )
        # sigma(theta)/sigma_SM = E_SM[K/K_SM] = sum_i K_i / sum_i K_i,SM
        rate_coefficients = k.sum(axis=0) / base_weight.sum()
        return cls(
            observables=jnp.asarray(x, jnp.float32),
            gen_parameters=jnp.array([1.0, 0.0]),
            latent_data=jnp.asarray(k / k[:, :1], jnp.float32),
            base_weight=jnp.asarray(base_weight / base_weight.mean(), jnp.float32),
            normalization=QuadraticFormNormalization.from_coefficients(
                jnp.asarray(rate_coefficients, jnp.float32)
            ),
            extended_likelihood=extended_likelihood,
        )


@dataclasses.dataclass
class HZZLoader:
    """Dataset factory for :class:`HZZDataset` (see :class:`event2vec.experiment.DatasetFactory`)."""

    data_path: Path
    coupling: str = "ghz4"
    features: str = "invariants"
    per_block: int = 150_000
    blocks: tuple[int, ...] = GOOD_BLOCKS

    def __call__(self, *, key: PRNGKeyArray) -> HZZDataset:
        return HZZDataset.from_root(
            str(self.data_path),
            key=key,
            coupling=self.coupling,
            features=self.features,
            per_block=self.per_block,
            blocks=self.blocks,
        )
