def install():
    from opt_core.kernels import ln
    from boltz2_opt import ditexact, msa2
    from functools import partial
    from opt_core.ops import msa_pwa2

    # Keep each adapter's real-input bit comparison; a failed class remains refused.
    if "8.9" not in ln.EXACTLN_PROVEN_CC:
        ln.EXACTLN_PROVEN_CC += ("8.9",)
    dx = ditexact._dx()
    dx.PROVEN_CC["smx"].add((8, 9))
    dx.PROVEN_CC["glue"].add((8, 9))
    msa2.EXACT_PIN["cc"]["trans2x"] += ((8, 9),)
    if (8, 9) not in msa2.EXACT_PIN["cc"]["pwa2x"]:
        msa2.EXACT_PIN["cc"]["pwa2x"] += ((8, 9),)
        msa_pwa2.pwa_forward = partial(msa_pwa2.pwa_forward, BI=32, BS=4, num_warps=4, num_stages=1)
