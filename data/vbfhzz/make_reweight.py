from itertools import product

import numpy as np

fWc = np.array(
    [
        0.0032308741124999997,
        0.0017678811482655726,
        -0.009692622337499995,
        -0.003330850311085803,
    ]
)
fWWc = np.array([-0.006461748225, 0.0, 0.0, 0.0])
coefs = ["cHW", "cHWB", "cHbox", "cH"]


def rw(fW: float, fWW: float):
    val = fW * fWc + fWW * fWWc
    name = f"fW_{fW:.1f}_fWW_{fWW:.1f}".replace(".", "p").replace("-", "m")
    out = f"launch --rwgt_name={name}\n"
    for c, v in zip(coefs, val):
        out += f"set {c} {v:.10e}\n"
    return out


pts = [-10, -1, -0.1, 0, 0.1, 1, 10]
for fW, fWW in product(pts, pts):
    print(rw(fW, fWW))
