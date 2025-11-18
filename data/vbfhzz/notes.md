The smeftsim_VBFH-\*.lhe.gz are produced using

```
import model SMEFTsim_topU3l_MwScheme_UFO-masslessHVonly
define q = u c d s u~ c~ d~ s~
generate p p > h q q QED=3 QCD=0 SMHLOOP=0 NP=1
```

where masslessHVonly is a restrict card with only cH, cHbox, cHDD, cHW, cHB,
cHWB set to non-zero.

The events are generated at the standard model, i.e. the param_card.dat is

```
BLOCK SMEFT #
      3 0.000000e-00 # ch
      4 0.000000e-00 # chbox
      5 0.000000e-00 # cHDD
      7 0.000000e-00 # chw
      8 0.000000e-00 # chb
      9 0.000000e-00 # chwb
```

The reweight points span the quadratic space of these parameters except cH since
it doesn't appear in any diagrams for this process, i.e. the reweight_card.dat
is

```
change rwgt_dir rwgt
launch --rwgt_name=SM
launch --rwgt_name=cHB_0p1
set cHB 0.1
launch --rwgt_name=cHB_0p2
set cHB 0.2
launch --rwgt_name=cHDD_0p1
set cHDD 0.1
launch --rwgt_name=cHDD_0p2
set cHDD 0.2
launch --rwgt_name=cHW_0p1
set cHW 0.1
launch --rwgt_name=cHW_0p2
set cHW 0.2
launch --rwgt_name=cHWB_0p1
set cHWB 0.1
launch --rwgt_name=cHWB_0p2
set cHWB 0.2
launch --rwgt_name=cHbox_0p1
set cHbox 0.1
launch --rwgt_name=cHbox_0p2
set cHbox 0.2
launch --rwgt_name=cHB_0p1_cHDD_0p1
set cHB 0.1
set cHDD 0.1
launch --rwgt_name=cHB_0p1_cHW_0p1
set cHB 0.1
set cHW 0.1
launch --rwgt_name=cHB_0p1_cHWB_0p1
set cHB 0.1
set cHWB 0.1
launch --rwgt_name=cHB_0p1_cHbox_0p1
set cHB 0.1
set cHbox 0.1
launch --rwgt_name=cHDD_0p1_cHW_0p1
set cHDD 0.1
set cHW 0.1
launch --rwgt_name=cHDD_0p1_cHWB_0p1
set cHDD 0.1
set cHWB 0.1
launch --rwgt_name=cHDD_0p1_cHbox_0p1
set cHDD 0.1
set cHbox 0.1
launch --rwgt_name=cHW_0p1_cHWB_0p1
set cHW 0.1
set cHWB 0.1
launch --rwgt_name=cHW_0p1_cHbox_0p1
set cHW 0.1
set cHbox 0.1
launch --rwgt_name=cHWB_0p1_cHbox_0p1
set cHWB 0.1
set cHbox 0.1
```
