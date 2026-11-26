import statistics
from types import MethodType

import torch
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.transition import Transition

from boltz_ops_opt import _opm_forward_opt, _pwa_forward_opt


def bench_cuda(fn, warmup: int = 3, iters: int = 10):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        stop.record()
        stop.synchronize()
        times_ms.append(start.elapsed_time(stop))
    return out, statistics.median(times_ms)


def check_error(name: str, ref: torch.Tensor, opt: torch.Tensor, max_tol: float):
    assert torch.isfinite(opt).all(), f"{name}: non-finite output!"
    ref_f = ref.float()
    opt_f = opt.float()
    diff = (ref_f - opt_f).abs()
    max_abs = float(diff.max().item())
    rel_l2 = float(torch.linalg.norm(diff) / (torch.linalg.norm(ref_f) + 1e-8))
    print(f"[EXACT] {name}: max_abs={max_abs:.6f} rel_l2={rel_l2:.6f} (tol={max_tol})")
    assert max_abs <= max_tol, f"{name}: max_abs={max_abs} > tol={max_tol}"
    return max_abs, rel_l2


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    bsz, n_seq, n_tok, c_m, c_z = 1, 8192, 263, 64, 128

    m = torch.randn(bsz, n_seq, n_tok, c_m, device=device, dtype=torch.bfloat16)
    z = torch.randn(bsz, n_tok, n_tok, c_z, device=device, dtype=torch.bfloat16)
    msa_mask = (torch.rand(bsz, n_seq, n_tok, device=device) > 0.1).float()
    pair_mask = torch.ones(bsz, n_tok, n_tok, device=device, dtype=torch.float32)

    # 1. OuterProductMean
    opm = OuterProductMean(c_in=c_m, c_hidden=32, c_out=c_z).to(device).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_opm, ms_opm_ref = bench_cuda(lambda: opm(m, msa_mask, chunk_size=None))
        opm._orig_forward = opm.forward
        opm.forward = MethodType(_opm_forward_opt, opm)
        opt_opm, ms_opm_opt = bench_cuda(lambda: opm(m, msa_mask, chunk_size=None))
    check_error("OuterProductMean", ref_opm, opt_opm, max_tol=5e-2)
    print(f"[BENCH] OuterProductMean: stock={ms_opm_ref:.2f}ms opt={ms_opm_opt:.2f}ms ({ms_opm_ref/ms_opm_opt:.2f}x)")

    # 2. PairWeightedAveraging
    pwa = PairWeightedAveraging(c_m=c_m, c_z=c_z, c_h=32, num_heads=8).to(device).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_pwa, ms_pwa_ref = bench_cuda(lambda: pwa(m, z, pair_mask, chunk_heads=False))
        pwa._orig_forward = pwa.forward
        pwa.forward = MethodType(_pwa_forward_opt, pwa)
        opt_pwa, ms_pwa_opt = bench_cuda(lambda: pwa(m, z, pair_mask, chunk_heads=False))
    check_error("PairWeightedAveraging", ref_pwa, opt_pwa, max_tol=5e-2)
    print(f"[BENCH] PairWeightedAveraging: stock={ms_pwa_ref:.2f}ms opt={ms_pwa_opt:.2f}ms ({ms_pwa_ref/ms_pwa_opt:.2f}x)")

    # 3. Transition (eager vs torch.compile)
    for label, x in (
        ("Transition_MSA", m),
        ("Transition_Pair", z),
    ):
        dim = x.shape[-1]
        tr = Transition(dim=dim, hidden=dim * 4).to(device).eval()
        tr_c = torch.compile(tr, mode="default")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref_tr, ms_tr_ref = bench_cuda(lambda: tr(x))
            opt_tr, ms_tr_opt = bench_cuda(lambda: tr_c(x))
        check_error(label, ref_tr, opt_tr, max_tol=5e-2)
        print(f"[BENCH] {label}: stock={ms_tr_ref:.2f}ms compile={ms_tr_opt:.2f}ms ({ms_tr_ref/ms_tr_opt:.2f}x)")


if __name__ == "__main__":
    main()
