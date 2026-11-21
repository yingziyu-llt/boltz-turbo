import json
import torch
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from opt_core.ops import msa_pwa2

torch.manual_seed(0)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for n, s in ((199, 128), (397, 128)):
        mod = PairWeightedAveraging(64, 128, 32, 8).cuda().eval()
        torch.nn.init.normal_(mod.proj_o.weight, std=0.03)
        m = torch.randn(1, s, n, 64, device="cuda")
        z = torch.randn(1, n, n, 128, device="cuda")
        mask = torch.ones(1, n, n, device="cuda")
        chunk = n > 384
        ref = mod(m, z, mask, chunk)
        for km, residue in msa_pwa2.distinct_candidates(n):
            out = msa_pwa2.pwa_forward(mod, m, z, mask, chunk, km, residue, BI=32, BS=4, num_warps=4, num_stages=1)
            error = out.float()-ref.float()
            print(json.dumps(dict(n=n, s=s, kmode=km, residue=residue, equal=torch.equal(out, ref),
                                  max_abs=error.abs().max().item(), relative_rms=(error.square().mean()/ref.float().square().mean()).sqrt().item())), flush=True)
