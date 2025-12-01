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
  - Sidenote: "static" is a loaded term in the `jax` ecosystem, hence the term
    frozen:
    [gh-comment-1](https://github.com/patrick-kidger/equinox/issues/798#issuecomment-2284687593),
    [gh-comment-2](https://github.com/patrick-kidger/equinox/issues/1095#issuecomment-3309828519),
    [gh-discussion](https://github.com/jax-ml/jax/discussions/13913)

    - Marking an attribute as a metadata field (e.g., with
      `eqx.field(static=True)` or with
      `jax.tree_util.register_dataclass(..., meta_fields=...)` absorbs the
      attribute into the pytreedef. The attribute will be ignored in vmap, grad,
      jit, etc. `vmap` and `grad`: One can't vmap over or take gradient wrt
      metadata fields. `jit`: Equality of the attribute is checked every time a
      jitted function is called (which could affect performance). If an array is
      marked as a metadata field, things seem to work okay as long as a given
      jitted function is only called with the same value for the metadata field.
      When it is called with a different value, jax raises an error (which is
      good) and recommends against setting arrays as metadata fields.
    - Indicating that an array is static via jit's `static_argnums` or
      `static_argnames` will fail outright (again, good), due to
      non-hashability.
    - One correct way to handle non-trainable parameters is to take them out of
      the first argument to loss (so, jit.grad and optax ignore them
      completely). Additionally, it is more efficient to close over
      non-trainable params for jitting purposes (and ensure that the jitted
      function is not public).
    - There are of course, genuine use-cases for static fields, e.g.,
      [gh-comment](https://github.com/patrick-kidger/equinox/issues/154#issuecomment-1198505287).

- `loss.py`

  - Evaluates `model` using `data`. In principle, all the computations for
    computing the loss (including the `model.__call__(data)` logic) can be
    inside the loss function. Abstract (model, datacontent, loss) combinations
    move some computations from losses into models.
  - This file provides a base loss implementation that handles the vmapping of
    loss functions defined on datapoints to work with datasets. It should cover
    most use-cases.

- `metric.py`

  - Metrics can return numeric objects as well as histograms, images, figures,
    etc.
  - This file will provide a base metric implementation for handling vmapping,
    and combinee jittable and non-jittable portions of the computation

- `training_and_evaluation.py`

  - Implements performing training steps and computing evaluation metrics.
  - Makes sure that frozen model components and meta/constant data attributes
    are closed over and not traced over in jit.
  - Handles shuffling datapoints, batching them, etc.
  - This file is a layer that separates the files listed above from the
    experiment trackers, visualizers, etc., to improve maintainability. It
    should provide (just) enough functionality to support `simple_run.py`.

- `run_experiment.py`

  - Implements a train function that (i) runs a training loop with a progress
    bar, (ii) computes evaluation metrics, and (iii) optionally logs metrics
    into a tracker, if provided.
  - Low priority: implement callback hooks
  - Low priority: Support for multistage experiments? Freeze/unfreeze model
    components, learn psd and then a summary vector, etc.

- `experiment_tracking.py`
  - Provides an experiment tracker API or a plugin to external tool.
  - It might make sense to plug into an existing open-source tool, say from this
    [list](https://github.com/awesome-mlops/awesome-ml-experiment-management)?
    [Aim](https://github.com/aimhubio/aim) is open-source, limits itself to
    experiment tracking, and can track
    [metrics, distributions, images, figures, etc.](https://aimstack.readthedocs.io/en/latest/quick_start/supported_types.html)

## Parts outside `pipelinax`:

- Less abstract trackers, loggers, visualization metrics, callbacks, etc. This
  is the open-ended world of model builders, config objects, histograms,
  callbacks, etc.
- Less abstract models, losses, datacontents.
