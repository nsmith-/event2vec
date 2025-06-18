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
    latent_data: jax.Array
    """The reweight basis"""

    @classmethod
    def from_lhe(cls, path: str):
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
        starting_point = jnp.broadcast_to(jnp.array([0.0, 0.0]), (len(latent_data), 2))
        return cls(
            observables=observables,
            gen_parameters=starting_point,
            latent_data=latent_data,
        )

    def likelihood(self, param: jax.Array) -> jax.Array:
        norm = 1.0 + param @ self.latent_data.mean(axis=0)
        weights = 1.0 + jnp.vecdot(self.latent_data, param)
        return jnp.maximum(weights / norm, jnp.finfo(jnp.float32).eps)
