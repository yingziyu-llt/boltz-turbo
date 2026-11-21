import os
from boltz2_opt import modes
from boltz2_opt.__main__ import main

# L40 不在原套件的精确副本验证范围；保留原树，按硬件明确裁剪。
exact_drops = {
    "exactln": {"reason": "cc_unproven:sm_89", "env": ("BOLTZ_EXACTLN",), "attach": "exactln"},
    "dit_smx": {"reason": "unproven_cc:sm_89"},
    "dit_glue": {"reason": "unproven_cc:sm_89"},
    "msa_pwa_exact": {"reason": "pin:cc89"},
    "msa_trans2_exact": {"reason": "pin:cc89"},
}
modes.CARD_DROPS["8.9"] = {
    "exact": exact_drops,
    "fast": {"exactln": exact_drops["exactln"],
             "fpf_opm": {"reason": "OutOfResources:sm_89_default_and_v2"}},
    "big": {"exactln": exact_drops["exactln"],
            "fpf_opm": {"reason": "OutOfResources:sm_89_default_and_v2"},
            "dit_tf32": {"reason": "unvouched_transition_cell:sm_89"}},
}
# 原套件 v2：指针加载、小 tile，不使用 Hopper TMA。
os.environ["FPF_OPM_CFG"] = "v2"
for mode in ("fast", "big"):
    modes.MODES[mode]["env"]["FPF_OPM_CFG"] = "v2"
raise SystemExit(main())
