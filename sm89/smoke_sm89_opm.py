import json
import torch
import triton
from boltz.model.layers.outer_product_mean import OuterProductMean
from opt_core.ops import msa_opm
from sm89_opm import CONFIGS, install

install()
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.get_device_capability() == (8, 9)
rows = []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for n, s in [(199, 128), (397, 128), (397, 1024)]:
        module = OuterProductMean(64, 32, 128).cuda().eval()
        torch.nn.init.normal_(module.proj_o.weight, std=0.03)
        torch.nn.init.normal_(module.proj_o.bias, std=0.02)
        m = torch.randn(1, s, n, 64, device="cuda")
        mask = (torch.rand(1, s, n, device="cuda") > 0.1).float()
        chunk = 4 if n > 384 else None
        ref = module(m, mask, chunk)
        stock_ms = triton.testing.do_bench(lambda: module(m, mask, chunk), warmup=100, rep=300)
        for name, cfg in CONFIGS.items():
            msa_opm._CFG_BY_CC[(8, 9)][("mask_norm", 128)] = cfg
            out = msa_opm.forward_mask_norm(module, m, mask, chunk)
            error = (out.float() - ref.float())
            relative_rms = (error.square().mean() / ref.float().square().mean()).sqrt().item()
            assert torch.isfinite(out).all()
            assert relative_rms < 0.015, (n, s, name, relative_rms)
            assert torch.equal(out, msa_opm.forward_mask_norm(module, m, mask, chunk))
            ms = triton.testing.do_bench(lambda: msa_opm.forward_mask_norm(module, m, mask, chunk), warmup=100, rep=300)
            row = dict(n=n, s=s, cfg=name, stock_ms=stock_ms, fused_ms=ms, speedup=stock_ms/ms,
                       relative_rms=relative_rms, max_abs=error.abs().max().item())
            rows.append(row)
            print(json.dumps(row), flush=True)
print(json.dumps({"rows": rows, "kernel_resources": {str(k): v for k, v in msa_opm._KSTATS.items()}}), flush=True)
