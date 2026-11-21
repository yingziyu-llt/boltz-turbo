import os

# Ada's 99 KiB shared-memory limit requires pointer loads and smaller MMA tiles.
CONFIGS = {
    "ada64": dict(BI=2, BJ=4, BK=32, BZ=16, GROUP_M=8, num_warps=4, num_stages=2, AT=True),
    "ada128": dict(BI=4, BJ=4, BK=32, BZ=16, GROUP_M=8, num_warps=4, num_stages=2, AT=True),
    "ada64p": dict(BI=2, BJ=4, BK=32, BZ=16, GROUP_M=8, num_warps=4, num_stages=2, AT=True, PM=1),
}


def install():
    from opt_core.ops import msa_opm

    msa_opm.CFG_VARIANTS.update(CONFIGS)
    msa_opm._CFG_BY_CC[(8, 9)] = {("mask_norm", 128): CONFIGS[os.environ.get("SM89_OPM_CFG", "ada128")]}
