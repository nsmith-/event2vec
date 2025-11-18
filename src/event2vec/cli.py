import argparse
from datetime import datetime, timezone
from pathlib import Path

import rich

import event2vec
import event2vec.experiments as experiments
from event2vec.experiment import ExperimentConfig


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
        # TODO: equinox modules don't print full reprs of array data
        rich.print(config, file=fout)
    config.run(output_dir)


if __name__ == "__main__":
    main()
