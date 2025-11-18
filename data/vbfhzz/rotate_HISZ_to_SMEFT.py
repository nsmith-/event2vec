"""Convert MadMiner example process to Warsaw basis

This is the VBF Higgs to ZZ example as used in https://arxiv.org/pdf/1805.00020 Eqn. 9

Can be run after installing Rosetta with:
pip install git+https://github.com/nsmith-/Rosetta.git@modernize

The output should be:

In HISZ basis:
Lam 1000.0
fW 1.0
Rosetta will be performing the translation:
    HISZ -> HiggsBasis -> WarsawBasis
In Warsaw basis:
cHW 0.0032308741124999997
cHWB 0.0017678811482655726
cHbx -0.009692622337499995
cH -0.003330850311085803
In HISZ basis:
Lam 1000.0
fWW 1.0
Rosetta will be performing the translation:
    HISZ -> HiggsBasis -> WarsawBasis
In Warsaw basis:
cHW -0.006461748225
"""

from rosetta import HISZ


def rotate(input_dat: str):
    hisz = HISZ.HISZ(input_dat)

    print("In HISZ basis:")
    for coef in hisz.all_coeffs:
        val = hisz[coef]
        if abs(val) > 1e-10:
            print(coef, val)

    warsaw = hisz.translate("warsaw")
    print("In Warsaw basis:")
    for coef in warsaw.all_coeffs:
        val = warsaw[coef]
        if abs(val) > 1e-10:
            print(coef, val)


if __name__ == "__main__":
    rotate("hisz_fW1.dat")
    rotate("hisz_fWW1.dat")
