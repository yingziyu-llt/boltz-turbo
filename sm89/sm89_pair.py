def install():
    from opt_core.attn import pair_fused as pf

    specs = [
        ("prologue", [128, 4, 32], "v3", dict(BI=8, BJ=8, num_warps=4, num_stages=1)),
        ("epilogue", [128, 4, 32], "v2", dict(KVER=2, BI=8, BJ=4, num_warps=4, num_stages=1, EXP="libdevice")),
        ("prologue", [64, 4, 32], "v3", dict(BI=8, BJ=8, num_warps=4, num_stages=1)),
        ("epilogue", [64, 4, 32], "v2", dict(KVER=2, BI=8, BJ=4, num_warps=4, num_stages=1, EXP="libdevice")),
        ("transition", [128, 512], "v1", dict(BM=32, BH=32, num_warps=4, num_stages=1, IL=0)),
        ("transition", [128, 256], "v1", dict(BM=32, BH=32, num_warps=4, num_stages=1, IL=0)),
        ("transition", [64, 128], "v1", dict(BM=32, BH=32, num_warps=4, num_stages=1, IL=0)),
    ]
    rows = pf.cells()["rows"]
    for piece, key, variant, cfg in specs:
        ident = "sm89:" + piece + ":" + "x".join(map(str, key))
        if not any(row["id"] == ident for row in rows):
            rows.append(dict(id=ident, impl="fpf", piece=piece, key=key, cc="8.9", triton="3.7", variant=variant,
                             cfg=cfg, status="certified", evidence="smoke_sm89_pair.py on L40; explicit low-shared-memory launch"))
