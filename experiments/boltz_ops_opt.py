import functools
from types import MethodType

import torch
import torch.nn.functional as F
from boltz.data import const
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.triangular_attention import primitives as tri_primitives
from boltz.model.modules.trunkv2 import MSAModule


def _opm_forward_opt(self, m: torch.Tensor, mask: torch.Tensor, chunk_size: int = None) -> torch.Tensor:
    if chunk_size is not None or self.training:
        return self._orig_forward(m, mask, chunk_size=chunk_size)

    # Mathematically equivalent to sum_s mask[b,s,i] * mask[b,s,j] without a 5D broadcast tensor.
    mask_2d = mask.to(dtype=torch.float32)
    num_mask = torch.matmul(mask_2d.transpose(1, 2), mask_2d).clamp(min=1.0).unsqueeze(-1)

    m_norm = self.norm(m)
    mask_e = mask.unsqueeze(-1).to(m_norm)
    a = self.proj_a(m_norm) * mask_e
    b = self.proj_b(m_norm) * mask_e

    bsz, n_seq, n_tok, c_hid = a.shape
    a_flat = a.reshape(bsz, n_seq, n_tok * c_hid)
    b_flat = b.reshape(bsz, n_seq, n_tok * c_hid)
    out = torch.matmul(a_flat.transpose(1, 2), b_flat)

    z = out.view(bsz, n_tok, c_hid, n_tok, c_hid).permute(0, 1, 3, 2, 4).reshape(bsz, n_tok, n_tok, c_hid * c_hid)
    z = (z.float() / num_mask).to(m_norm.dtype)
    return self.proj_o(z)


def _pwa_forward_opt(
    self, m: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, chunk_heads: bool = False
) -> torch.Tensor:
    if chunk_heads or self.training:
        return self._orig_forward(m, z, mask, chunk_heads=chunk_heads)

    z_n = self.norm_z(z)
    b = self.proj_z(z_n).permute(0, 3, 1, 2)
    b = b + (1 - mask[:, None]) * -self.inf
    w = torch.softmax(b, dim=-1)

    bsz, n_seq, n_tok, _ = m.shape
    n_heads = self.num_heads
    c_h = self.c_h
    w_bh = w.reshape(bsz * n_heads, n_tok, n_tok)

    w_mg = getattr(self, "_cached_mg_weight", None)
    if w_mg is None or w_mg.device != self.proj_m.weight.device or w_mg.dtype != self.proj_m.weight.dtype:
        w_mg = torch.cat([self.proj_m.weight, self.proj_g.weight], dim=0).contiguous()
        self._cached_mg_weight = w_mg

    s_chunk = 1024 if n_seq > 1024 else n_seq
    out = None
    for s0 in range(0, n_seq, s_chunk):
        s1 = min(s0 + s_chunk, n_seq)
        sc = s1 - s0
        mc = self.norm_m(m[:, s0:s1])
        mg = F.linear(mc, w_mg)
        vc, gc = mg.chunk(2, dim=-1)
        gc = gc.sigmoid()

        vc_bh = vc.view(bsz, sc, n_tok, n_heads, c_h).permute(0, 3, 2, 1, 4).reshape(bsz * n_heads, n_tok, sc * c_h)
        oc_bh = torch.bmm(w_bh.to(vc_bh.dtype), vc_bh)
        oc = oc_bh.view(bsz, n_heads, n_tok, sc, c_h).permute(0, 3, 2, 1, 4).reshape(bsz, sc, n_tok, n_heads * c_h)
        out_c = self.proj_o(gc * oc)
        if s_chunk == n_seq:
            return out_c
        if out is None:
            out = torch.empty((bsz, n_seq, n_tok, out_c.shape[-1]), device=out_c.device, dtype=out_c.dtype)
        out[:, s0:s1] = out_c
    return out


def _msa_module_forward_opt(
    self,
    z: torch.Tensor,
    emb: torch.Tensor,
    feats: dict[str, torch.Tensor],
    use_kernels: bool = False,
) -> torch.Tensor:
    if self.training or self.subsample_msa:
        return self._orig_forward(z, emb, feats, use_kernels=use_kernels)

    if z.shape[1] > const.chunk_size_threshold:
        chunk_heads_pwa = True
        chunk_size_transition_z = 64
        chunk_size_transition_msa = 32
        chunk_size_outer_product = 4
        chunk_size_tri_attn = 128
    else:
        chunk_heads_pwa = False
        chunk_size_transition_z = None
        chunk_size_transition_msa = None
        chunk_size_outer_product = None
        chunk_size_tri_attn = 512

    msa_tensor = feats["msa"]
    cache_key = (msa_tensor.data_ptr(), emb.data_ptr(), tuple(msa_tensor.shape), tuple(emb.shape))
    cached = getattr(self, "_boltz_ops_msa_cache", None)
    if cached is not None and cached[0] == cache_key:
        _, m, msa_mask, token_mask = cached
    else:
        msa_oh = torch.nn.functional.one_hot(msa_tensor, num_classes=const.num_tokens)
        has_deletion = feats["has_deletion"].unsqueeze(-1)
        deletion_value = feats["deletion_value"].unsqueeze(-1)
        is_paired = feats["msa_paired"].unsqueeze(-1)
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"].float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        if self.use_paired_feature:
            m_feat = torch.cat([msa_oh, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m_feat = torch.cat([msa_oh, has_deletion, deletion_value], dim=-1)

        m = self.msa_proj(m_feat) + self.s_proj(emb).unsqueeze(1)
        self._boltz_ops_msa_cache = (cache_key, m, msa_mask, token_mask)

    for layer in self.layers:
        z, m = layer(
            z,
            m,
            token_mask,
            msa_mask,
            chunk_heads_pwa,
            chunk_size_transition_z,
            chunk_size_transition_msa,
            chunk_size_outer_product,
            chunk_size_tri_attn,
            use_kernels,
        )
    return z


def install_ops_opt(model: torch.nn.Module):
    """Install reversible lossless operator optimizations on a Boltz2 model."""
    patched_modules = []

    for mod in model.modules():
        if isinstance(mod, OuterProductMean):
            mod._orig_forward = mod.forward
            mod.forward = MethodType(_opm_forward_opt, mod)
            patched_modules.append(mod)
        elif isinstance(mod, PairWeightedAveraging):
            mod._orig_forward = mod.forward
            mod.forward = MethodType(_pwa_forward_opt, mod)
            patched_modules.append(mod)
        elif isinstance(mod, MSAModule):
            mod._orig_forward = mod.forward
            mod.forward = MethodType(_msa_module_forward_opt, mod)
            patched_modules.append(mod)

    orig_kernel_tri_attn = tri_primitives.kernel_triangular_attn

    @functools.wraps(orig_kernel_tri_attn)
    def kernel_triangular_attn_fp32_bias(q, k, v, tri_bias, mask, scale):
        if tri_bias.dtype != torch.float32:
            tri_bias = tri_bias.float()
        return orig_kernel_tri_attn(q, k, v, tri_bias=tri_bias, mask=mask, scale=scale)

    tri_primitives.kernel_triangular_attn = kernel_triangular_attn_fp32_bias

    def restore():
        tri_primitives.kernel_triangular_attn = orig_kernel_tri_attn
        for mod in patched_modules:
            if hasattr(mod, "_orig_forward"):
                mod.forward = mod._orig_forward
                delattr(mod, "_orig_forward")
            if hasattr(mod, "_boltz_ops_msa_cache"):
                delattr(mod, "_boltz_ops_msa_cache")
            if hasattr(mod, "_cached_mg_weight"):
                delattr(mod, "_cached_mg_weight")

    return restore
