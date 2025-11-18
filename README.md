# event2vec

Contributor quick start:

```sh
git clone https://github.com/nsmith-/event2vec.git
cd event2vec
python -m venv .venv
source .venv/bin/activate
pip install -e .

e2vrun --help
```

## Gaussian mixture example experiment:

```sh
e2vrun -o gauss GaussianMixture
```

This will create an experiment directory `gauss/` with training logs, model
checkpoints, and analysis plots.
