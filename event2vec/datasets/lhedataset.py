import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import pylhe

from event2vec.dataset import ReweightableDataset


def _to_awkward(path: str) -> ak.Array:
    """Convert an LHE file to an awkward array."""
    # workaround for pylhe.to_awkward not supporting weights
    event = next(pylhe.read_lhe_with_attributes(path))
    events = pylhe.to_awkward(pylhe.read_lhe_with_attributes(path))
    assert isinstance(event.weights, dict)
    weights = {w: np.zeros(len(events)) for w in event.weights.keys()}

    for i, event in enumerate(pylhe.read_lhe_with_attributes(path)):
        for name, weight in weights.items():
            weight[i] = event.weights[name]  # type: ignore

    events["weights"] = ak.zip(weights)
    return events


def _lightjets_mask(pid: ak.Array) -> ak.Array:
    """Send all light jets to the same id, since they are more or less indistinguishable."""
    return ak.where(
        (abs(pid) < 4) | (pid == 21),
        1,  # Light jets (u, d, s, g)
        pid,
    )  # type: ignore


class DY1JDataset(ReweightableDataset):
    """A dataset for the Drell-Yan + 1 jet process.

    """

    latent_data: jax.Array
    """The reweight basis"""
    latent_norm: jax.Array
    """The normalization of the events"""
    extended_likelihood: bool = False
    """Whether to use the extended likelihood (i.e. include the overall normalization shift in the weight)"""

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
