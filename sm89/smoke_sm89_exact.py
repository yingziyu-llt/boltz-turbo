import json
import torch
from opt_core.kernels import ln
from boltz2_opt import ditexact
from sm89_exact import install

install()
torch.manual_seed(0)
assert torch.cuda.get_device_capability() == (8, 9)
e = ln.carried_module("exactln")
with torch.inference_mode():
    for c in (64, 128, 256, 384, 512, 768):
        x = torch.randn(4096, c, device="cuda")
        w = torch.randn(c, device="cuda")
        b = torch.randn(c, device="cuda")
        for affine in (False, True):
            args = (w, b) if affine else (None, None)
            out = e.layer_norm(x, (c,), *args)
            ref = torch.nn.functional.layer_norm(x, (c,), *args)
            assert torch.equal(out, ref), (c, affine, (out-ref).abs().max().item())
        print(json.dumps(dict(kind="layer_norm", c=c, equal=True)), flush=True)
    dx = ditexact._dx()
    for n in (199, 397):
        for masked in (False, True):
            a = torch.randn(1, 4, n, n, device="cuda")
            b = torch.randn_like(a)
            mask = torch.zeros(1, 1, 1, n, device="cuda") if masked else None
            if masked:
                mask[..., -5:] = -1e9
            expected = a / (48**0.5) + b
            if mask is not None:
                expected += mask
            expected = expected.softmax(-1)
            out = dx.softmax_scale_bias_mask(a.clone(), 48**0.5, b, mask, ("smx", "sba"))
            assert torch.equal(out, expected)
            print(json.dumps(dict(kind="softmax", n=n, masked=masked, status=dx.STATS["smx_bitcmp"])), flush=True)
    a = torch.randn(1, 397, 768, device="cuda")
    g = torch.randn_like(a)
    o = torch.randn_like(a)
    for name, out, ref in (
        ("affine", dx.glue_affine(g, a, o, ("glue",)), g*a+o),
        ("gres", dx.glue_gres(a,g,o,("glue",)), a+g*o),
    ):
        assert torch.equal(out, ref), name
    print(json.dumps(dict(kind="glue", stats=dx.STATS["glue_bitcmp"], ln=e.facts())), flush=True)
