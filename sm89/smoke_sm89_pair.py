import json
import torch
import triton
from opt_core.attn import pair_fused as pf
from boltz.model.layers.triangular_attention.attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
from boltz2_opt.pairblock import weights_of, _cueq_core
from boltz.model.layers.transition import Transition
from sm89_pair import install

install()
torch.manual_seed(0)
assert torch.cuda.get_device_capability() == (8, 9)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for c in (64, 128):
        for n in (199, 397):
            for cls in (TriangleAttentionStartingNode, TriangleAttentionEndingNode):
                module = cls(c, 32, 4).cuda().eval()
                for p in module.parameters():
                    torch.nn.init.normal_(p, std=0.03)
                z = torch.randn(1, n, n, c, device="cuda")
                mask = torch.ones(1, n, n, device="cuda")
                w = pf.pack_triattn_weights(**weights_of(module), device=z.device)
                ref = module(z, mask, use_kernels=True)
                def fused():
                    return pf.tri_attn_block(z, w, mask, ending=cls is TriangleAttentionEndingNode, residual=False,
                                             impl="fpf", core=_cueq_core, ln="stock", x_ln=module.layer_norm(z.transpose(-2, -3) if cls is TriangleAttentionEndingNode else z).to(torch.bfloat16).contiguous())
                out = fused()
                delta = out.float()-ref.float()
                rms = (delta.square().mean()/ref.float().square().mean()).sqrt().item()
                assert torch.isfinite(out).all()
                assert rms < 0.02, (c, n, cls.__name__, rms)
                stock_ms = triton.testing.do_bench(lambda: module(z, mask, use_kernels=True), warmup=100, rep=200)
                fused_ms = triton.testing.do_bench(fused, warmup=100, rep=200)
                print(json.dumps(dict(c=c,n=n,kind=cls.__name__,relative_rms=rms,max_abs=delta.abs().max().item(),
                                      equal=torch.equal(out, ref),stock_ms=stock_ms,fused_ms=fused_ms,speedup=stock_ms/fused_ms)),flush=True)
    for c, hidden in ((64, 128), (128, 256), (128, 512)):
        module = Transition(c, hidden).cuda().eval()
        torch.nn.init.normal_(module.fc3.weight, std=0.03)
        x = torch.randn(1, 397, 397, c, device="cuda")
        weights = pf.pack_transition_weights(w_out=module.fc3.weight, w_a=module.fc1.weight,
                                             w_b=module.fc2.weight, ln_w=module.norm.weight, ln_b=module.norm.bias)
        ref = module(x)
        def fused_transition():
            return pf.transition(x, weights, ln="stock", x_ln=module.norm(x).to(torch.bfloat16))
        out = fused_transition()
        error = out.float() - ref.float()
        rms = (error.square().mean()/ref.float().square().mean()).sqrt().item()
        assert torch.isfinite(out).all() and rms < 0.02, (c, hidden, rms)
        stock_ms = triton.testing.do_bench(lambda: module(x), warmup=100, rep=200)
        fused_ms = triton.testing.do_bench(fused_transition, warmup=100, rep=200)
        print(json.dumps(dict(kind="transition", c=c, hidden=hidden, relative_rms=rms,
                              stock_ms=stock_ms, fused_ms=fused_ms, speedup=stock_ms/fused_ms)), flush=True)
