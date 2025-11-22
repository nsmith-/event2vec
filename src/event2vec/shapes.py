"""Shapes for various arrays used in event2vec.

Defined using jaxtyping and type aliases for clarity.
"""

from jaxtyping import Array, Float

type ParamVec = Float[Array, " P"]
r"""Parameter point $\theta$ vector"""

type ObsVec = Float[Array, " O"]
"""Observables $x$ vector"""

type LLRScalar = Float[Array, ""]
"""Log-likelihood ratio scalar"""

type LLRVec = Float[Array, " B"]
"""Binwise log-likelihood ratio vector"""

type ProbVec = Float[Array, " B"]
"""Binwise probability vector (sums to 1)"""

type ParamQuadVec = Float[Array, " Q"]
"""Outer product parameter vector (e.g. for quadratic forms), shape Q = P*(P+1)/2
in lower-triangular representation. See `tril_outer_product` to construct this from
P-dimensional parameter vector.
"""

type PSDMatrix = Float[Array, " P P"]
"""Positive semi-definite matrix of shape P x P, for P-dimensional parameter space."""
