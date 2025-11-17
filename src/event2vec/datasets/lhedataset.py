import glob

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import pylhe

from event2vec.dataset import QuadraticReweightableDataset, ReweightableDataset
from event2vec.nontrainable import QuadraticFormNormalization
from event2vec.util import EPS, standard_pbar, tril_outer_product


def _to_awkward(path: str) -> ak.Array:
    """Convert an LHE file to an awkward array."""
    if "*" in path:
        with standard_pbar() as progress:
            files = glob.glob(path)
            return ak.concatenate(
                [
                    _to_awkward(p)
                    for p in progress.track(files, description="Loading LHE files...")
                ]
            )
    return pylhe.to_awkward(pylhe.read_lhe_with_attributes(path))


def _lightjets_mask(pid: ak.Array) -> ak.Array:
    """Send all light jets to the same id, since they are more or less indistinguishable."""
    return ak.where(
        (abs(pid) < 4) | (pid == 21),
        1,  # Light jets (u, d, s, g)
        pid,
    )  # type: ignore


class DY1JDataset(ReweightableDataset):
    """A dataset for the Drell-Yan + 1 jet process.

    Only two EFT parameters are supported for now: ced and clj3.
    Only the linear dependence is included in the reweighting basis.
    """

    observables: jax.Array
    """The observed data points, shape (num_events, obs_dim)"""
    gen_parameters: jax.Array
    """The parameters used to sample this event, shape (2,)"""
    latent_data: jax.Array
    """The reweight basis"""
    latent_norm: jax.Array
    """The normalization of the events"""
    extended_likelihood: bool = False
    """Whether to use the extended likelihood (i.e. include the overall normalization shift in the weight)"""

    @property
    def observable_dim(self) -> int:
        return self.observables.shape[1]

    @property
    def parameter_dim(self) -> int:
        return self.latent_data.shape[1]

    @classmethod
    def from_lhe(
        cls,
        path: str,
        extended_likelihood: bool = False,
        add_unobservable: bool = False,
    ):
        """Load a dataset from an LHE file."""
        events = _to_awkward(path)
        items = []
        for i in (2, 3, 4):
            # Skip the first two particles (beam particles)
            items.extend(
                [
                    events.particles[:, i].vector.pt,
                    events.particles[:, i].vector.eta,
                    events.particles[:, i].vector.phi,
                    _lightjets_mask(events.particles[:, i].id),
                ]
            )
        if add_unobservable:
            for i in (0, 1, 2, 3, 4):
                items.extend(
                    [
                        events.particles[:, i].vector.pz,
                        events.particles[:, i].id,
                    ]
                )
            items.append(events.weights["ced_0p1"])
            items.append(events.weights["clj3_0p1"])
            items.append(events.weights["SM"])
        # TODO: add m_ll
        observables = jnp.array(ak.concatenate([i[:, None] for i in items], axis=-1))
        latent_data = jnp.array(
            ak.concatenate(
                [
                    events.weights["ced_0p1"][:, None],
                    events.weights["clj3_0p1"][:, None],
                ],
                axis=-1,
            )
            / events.weights["SM"][:, None],
        )
        starting_point = jnp.array([0.0, 0.0])
        latent_norm = jnp.mean(latent_data, axis=0)
        return cls(
            observables=observables,
            gen_parameters=starting_point,
            latent_data=latent_data,
            latent_norm=latent_norm,
            extended_likelihood=extended_likelihood,
        )

    def likelihood(self, param: jax.Array) -> jax.Array:
        weights = 1.0 + jnp.vecdot(self.latent_data, param)
        if self.extended_likelihood:
            return jnp.maximum(weights, jnp.finfo(jnp.float32).eps)
        norm = 1.0 + param @ self.latent_norm
        # return weights / norm
        # Avoid case where log(0) is called
        return jnp.maximum(weights / norm, jnp.finfo(jnp.float32).eps)


def _decode_weight_name(name: str, expected_wcs: list[str]) -> list[float]:
    """Decode the weight name into the corresponding Wilson coefficient values.

    Expected format: "cHbox_0p1_cHDD_m0p2", etc.
    The special "SM" name corresponds to all coefficients being zero except for cSM=1.
    """
    coefs = [0.0 for _ in expected_wcs]
    coefs[expected_wcs.index("cSM")] = 1.0
    if name == "SM":
        return coefs
    parts = name.split("_")
    for wc, val in zip(parts[0::2], parts[1::2]):
        idx = expected_wcs.index(wc)
        coefs[idx] = float(val.replace("m", "-").replace("p", "."))
    return coefs


def _extract_scaling_coefficients(
    event_weights: ak.Array, expected_wcs: list[str], gen_weights: ak.Array
) -> jax.Array:
    """Extract the scaling coefficients from the event weights.

    Args:
        event_weights: the event weights, with fields corresponding to weight names.
        expected_wcs: expected Wilson coefficient names, e.g. ["cSM", "cHbox", "cHDD", ...]
        gen_weights: the event weights at the generation point (should be all the same value)
    Returns:
        coeffs: array of shape (len(event_weights), num_coeffs), where num_coeffs = p * (p + 1) / 2, for
        p Wilson coefficients.

    Per event, coeffs @ tril_outer_product(gen_parameters) = 1

    The coefficients are extracted by solving a linear system of equations.
    """
    weight_names = list(event_weights.fields)
    points = np.array(
        [_decode_weight_name(name, expected_wcs) for name in weight_names]
    )
    points_quad = np.array([tril_outer_product(p) for p in points])
    weights = jnp.array([event_weights[name] / gen_weights for name in weight_names])
    coeffs, residuals, rank, _ = jnp.linalg.lstsq(points_quad, weights, rcond=None)
    print(
        f"LHE weight fit (rank {rank}) residuals mean: {jnp.mean(residuals)} std: {jnp.std(residuals)}"
    )
    return coeffs.T


class VBFHDataset(QuadraticReweightableDataset):
    """A dataset for a VBF Higgs process

    The Higgs is not decayed in this dataset.
    """

    observables: jax.Array
    """The observed data points, shape (num_events, obs_dim)"""
    gen_parameters: jax.Array
    """The parameters used to sample this event, shape (6,)
    
    Names: cSM, cHbox, cHDD, cHW, cHB, cHWB
    """
    latent_data: jax.Array
    r"""The event weight coefficients, shape (num_events, num_coeffs)
    
    This is $\frac{\theta^T A(z) \theta}{\theta_g^T A(z) \theta_g}$ where $A(z)$ is the matrix of coefficients for event z,
    and $\theta_g$ are the generation parameters.
    """
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

    @classmethod
    def from_lhe(
        cls,
        path: str,
        extended_likelihood: bool = False,
    ):
        """Load a dataset from an LHE file."""
        events = _to_awkward(path)
        h, j1, j2 = (events.particles[:, i] for i in (2, 3, 4))
        jjsystem = j1.vector + j2.vector
        items = [
            np.log(h.vector.pt),
            h.vector.eta,
            h.vector.phi,
            np.log(j1.vector.pt),
            j1.vector.eta,
            j1.vector.deltaphi(h.vector),
            _lightjets_mask(j1.id),
            np.log(j2.vector.pt),
            j2.vector.eta,
            j2.vector.deltaphi(h.vector),
            j2.vector.deltaphi(j1.vector),
            _lightjets_mask(j2.id),
            np.log(jjsystem.mass),
        ]
        observables = jnp.array(ak.concatenate([i[:, None] for i in items], axis=-1))
        latent_data = _extract_scaling_coefficients(
            events.weights,
            ["cSM", "cHbox", "cHDD", "cHW", "cHB", "cHWB"],
            events.eventinfo.weight,
        )
        starting_point = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        latent_norm = QuadraticFormNormalization.from_coefficients(
            jnp.mean(latent_data, axis=0)
        )
        return cls(
            observables=observables,
            gen_parameters=starting_point,
            latent_data=latent_data,
            normalization=latent_norm,
            extended_likelihood=extended_likelihood,
        )

    def likelihood(self, param: jax.Array) -> jax.Array:
        weights = jnp.vecdot(self.latent_data, tril_outer_product(param))
        if self.extended_likelihood:
            return jnp.maximum(weights, EPS)
        norm = self.normalization(param)
        # return weights / norm
        # Avoid case where log(0) is called
        return jnp.maximum(weights / norm, EPS)
