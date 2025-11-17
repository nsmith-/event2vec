"""Shapes for various arrays used in event2vec.

Defined using jaxtyping and type aliases for clarity.
"""

from typing import TypeAlias
from jaxtyping import Float, Array

ParamVec: TypeAlias = Float[Array, " P"]
r"""Parameter point $\theta$ vector"""

ObsVec: TypeAlias = Float[Array, " O"]
"""Observables $x$ vector"""

LLRScalar: TypeAlias = Float[Array, ""]
"""Log-likelihood ratio scalar"""

LLRVec: TypeAlias = Float[Array, " B"]
"""Binwise log-likelihood ratio vector"""

ProbVec: TypeAlias = Float[Array, " B"]
"""Binwise probability vector (sums to 1)"""

ParamQuadVec: TypeAlias = Float[Array, " Q"]
"""Outer product parameter vector (e.g. for quadratic forms), shape Q = P*(P+1)/2
in lower-triangular representation. See `tril_outer_product` to construct this from
P-dimensional parameter vector.
"""

PSDMatrix: TypeAlias = Float[Array, " P P"]
"""Positive semi-definite matrix of shape P x P, for P-dimensional parameter space."""
