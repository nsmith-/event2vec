# event2vec

## Contributor quick start

1. Clone repo and enter project dir:
   ```sh
   git clone https://github.com/nsmith-/event2vec.git
   cd event2vec
   ```
2. Create virtual environment (with dev dependencies) and activate it.

   - Using `uv` (preferred; uses `uv.lock`):
     ```sh
     uv sync
     source .venv/bin/activate
     ```
   - Using pip (ignores lock file):
     ```sh
     python -m venv .venv --prompt event2vec
     source .venv/bin/activate
     pip install -e . --group dev
     ```

   **Note:** If using `uv`, the remaining commands can be run without activating
   the venv, by prepending each command with `uv run `.

3. Set up the pre-commit git hooks:
   ```sh
   pre-commit install
   ```
4. Try the `e2vrun` script:
   ```sh
   e2vrun --help
   ```

## Gaussian mixture example experiment:

```sh
e2vrun -o gauss GaussianMixture
```

This will create an experiment directory `gauss/` with training logs, model
checkpoints, and analysis plots.
