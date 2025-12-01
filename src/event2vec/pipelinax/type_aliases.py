from typing import Literal

import equinox as eqx

type Model = eqx.Module

# To support different behaviors during training and evaluation (e.g., dropout on/off)
type ModeStr = Literal["training", "evaluation"]
