This submodule implements basic neural network pipelining. This should be
portable across projects. Higher-level constructs like

- less abstract models, losses, datacontents
- config objects and builders
- progress visualizers
- experiment trackers

can be built on top of this submodule.

## Components

- `data.py`

  - Sets the API for declaring a data object as representing a datapoint or a
    dataset. Provides generic classes `DataPoint[DataContentT]` and
    `DataSet[DataContentT]` for losses to interact with.
  - Sets the API for providing meta dataset-attributes and constant
    datapoint-attributes (to be broadcasted over the datapoints).

- ~`model.py`~

  - Fundamentally, a model is just a container for parameters. Right now,
    `Model` is just a type alias for `eqx.Module` defined in `type_aliases.py`.
    Could even be a type alias for `PyTree`?

- `nontrainability.py`

  - This file sets the API for marking certain arrays/modules as "frozen"
    (sidenote: "static" is a loaded term in the `jax` ecosystem:
    [gh-comment-1](https://github.com/patrick-kidger/equinox/issues/798#issuecomment-2284687593),
    [gh-comment-2](https://github.com/patrick-kidger/equinox/issues/1095#issuecomment-3309828519))

- `loss.py`

  - Evaluates `model` using `data`. In principle, all the computations for
    computing the loss (including the `model.__call__(data)` logic) can be
    inside the loss function. Abstract (model, datacontent, loss) combinations
    move some computations from losses into models.
  - This file defines a base loss implementation that handles the vmapping of
    loss functions defined on datapoints to work with datasets. It should cover
    most use-cases.

- `training_and_evaluation.py`
  - Implements performing training steps and computing evaluation metrics.
  - Makes sure that frozen model components and meta/constant data attributes
    are closed over and not traced over in jit.
  - Handles shuffling datapoints, batching them, etc.
  - This file is a layer that separates the files listed above from the
    experiment trackers, visualizers, etc., to improve maintainability. It
    should provide (just) enough functionality to support downstream tasks.

## Parts outside `pipelinax`:

- `{experiment, visualization, logging, tracking, callbacks}`
  - This is the open-ended world of model builders, config objects, progress
    bars, histograms, experiment trackers and loggers, callbacks, etc.
  - It might make sense to plug into an existing open-source tool, say from this
    [list](https://github.com/awesome-mlops/awesome-ml-experiment-management)?
    [Aim](https://github.com/aimhubio/aim) is open-source, limits itself to
    experiment tracking, and can track
    [metrics, distributions, images, etc.](https://aimstack.readthedocs.io/en/latest/quick_start/supported_types.html)
- Less abstract models, losses, datacontents.
