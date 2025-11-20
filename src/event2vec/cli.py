import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import rich

import event2vec
import event2vec.experiments as experiments
from event2vec.experiment import ExperimentConfig


def _full_leaf_repr(leaf):
    """Convert array leaves to their full repr string.

    This function is used to transform array leaves in the config pytree
    to their full string representations before saving to file.

    Args:
        leaf: A leaf node from the pytree

    Returns:
        For numpy/JAX arrays: their full repr as a string
        For other types: the leaf unchanged
    """
    # Check if it's a JAX array
    try:
        if isinstance(leaf, jax.Array):
            # Convert to numpy for consistent repr
            return repr(np.asarray(leaf))
    except ImportError:
        # JAX not available, skip JAX array handling
        pass

    # Check if it's a numpy array
    if isinstance(leaf, np.ndarray):
        return repr(leaf)

    # Return unchanged for non-array leaves
    return leaf


def _transform_arrays_recursive(obj):
    """Recursively transform arrays in nested structures to their full repr.

    This function traverses nested dataclasses, Equinox modules, and collections
    to find and transform array leaves to their string representations.

    Args:
        obj: The object to transform

    Returns:
        A copy of the object with all arrays replaced by their repr strings
    """
    # Handle arrays directly
    if isinstance(obj, (np.ndarray,)):
        return _full_leaf_repr(obj)

    # Check for JAX arrays if JAX is available
    try:
        if isinstance(obj, jax.Array):
            return _full_leaf_repr(obj)
    except ImportError:
        pass

    # Handle Equinox modules (which are pytrees)
    if isinstance(obj, eqx.Module):
        return jax.tree.map(_full_leaf_repr, obj)

    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        field_values = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            field_values[field.name] = _transform_arrays_recursive(value)
        return type(obj)(**field_values)

    # Handle lists
    if isinstance(obj, list):
        return [_transform_arrays_recursive(item) for item in obj]

    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(_transform_arrays_recursive(item) for item in obj)

    # Handle dicts
    if isinstance(obj, dict):
        return {k: _transform_arrays_recursive(v) for k, v in obj.items()}

    # Return unchanged for other types
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event2vec experiments.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory for the experiment.",
        required=True,
    )
    cmd_root = parser.add_subparsers(dest="command", required=True)
    configs: dict[str, type[ExperimentConfig]] = {}

    for name in experiments.__all__:
        cls = getattr(experiments, name)
        assert issubclass(cls, ExperimentConfig)
        configs[name] = cls
        cls.register_parser(
            cmd_root.add_parser(
                name, help=f"Run the {name} experiment", description=cls.__doc__
            )
        )
    args = parser.parse_args()
    config_cls = configs[args.command]
    config = config_cls.from_args(args)
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.py", "w") as fout:
        print(f"# Date: {datetime.now(timezone.utc)}", file=fout)
        print(f"{event2vec.__version__ = }", file=fout)
        # Transform config to show full array representations
        config_with_full_arrays = _transform_arrays_recursive(config)
        rich.print(config_with_full_arrays, file=fout)
    config.run(output_dir)


if __name__ == "__main__":
    main()
