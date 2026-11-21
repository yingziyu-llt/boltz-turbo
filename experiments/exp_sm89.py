import os
from boltz2_opt import modes
from boltz2_opt.__main__ import main

exact_drops = {
    "exactln": {"reason": "cc_unproven:sm_89", "env": ("BOLTZ_EXACTLN",), "attach": "exactln"},
    "dit_smx": {"reason": "unproven_cc:sm_89"},
    "dit_glue": {"reason": "unproven_cc:sm_89"},
    "msa_pwa_exact": {"reason": "pin:cc89"},
    "msa_trans2_exact": {"reason": "pin:cc89"},
}
modes.CARD_DROPS["8.9"] = {
    "exact": exact_drops,
    "fast": {"exactln": exact_drops["exactln"]},
    "big": {"exactln": exact_drops["exactln"], "dit_tf32": {"reason": "unvouched_transition_cell:sm_89"}},
}
if os.environ.get("SM89_FULL") == "1":
    modes.CARD_DROPS["8.9"] = {
        "exact": {},
        "fast": {},
        "big": {"dit_tf32": {"reason": "unvouched_transition_cell:sm_89"}},
    }
for mode in ("fast", "big"):
    modes.MODES[mode]["env"]["BOLTZ_FPF_MSA_OPM_CFG"] = os.environ.get("SM89_OPM_CFG", "ada128")
raise SystemExit(main())
