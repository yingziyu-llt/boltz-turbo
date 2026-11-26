"""Reusable Affinity-only Trunk-TeaCache optimization for Boltz-2."""

from __future__ import annotations

import torch


def _rel_l1(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().abs().sum().item()
    if denom <= 1e-12:
        return 0.0
    return float(diff.float().abs().sum().item() / denom)


def install_affinity_teacache(
    model,
    msa_cache_after: int = 1,
    pf_adaptive_thresh: float = 0.05,
):
    """Install Affinity-only MSA delta cache + Pairformer adaptive exit on `model`.

    Leaves the structure prediction stage 100% untouched (returns a no-op restorer
    when ``getattr(model, "affinity_prediction", False)`` is False).
    Returns ``(restore_fn, stats)``.
    """
    stats: dict[str, list[int]] = {"msa_recycles": [], "pf_recycles": []}
    if not getattr(model, "affinity_prediction", False):
        return (lambda: None), stats

    orig_model_fwd = model.forward
    msa_mod = getattr(model.msa_module, "_orig_mod", model.msa_module)
    pf_mod = getattr(model.pairformer_module, "_orig_mod", model.pairformer_module)

    def wrapped_forward(feats, *args, **kwargs):
        tok_valid = feats["token_pad_mask"][0].bool()
        mol_type = feats["mol_type"][0]
        prot_idx = torch.where((mol_type == 0) & tok_valid)[0]
        lig_idx = torch.where((mol_type == 3) & tok_valid)[0]

        orig_msa_fwd = msa_mod.forward
        orig_pf_fwd = pf_mod.forward
        state = {
            "msa_r": 0,
            "pf_r": 0,
            "msa_exec": 0,
            "pf_exec": 0,
            "msa_delta": None,
            "prev_z": None,
            "pf_converged": False,
            "pf_conv_s": None,
            "pf_conv_z": None,
        }

        def msa_hook(z, emb, f, *a, **kw):
            r = state["msa_r"]
            state["msa_r"] += 1
            if state["pf_converged"]:
                return z
            if r > msa_cache_after and state["msa_delta"] is not None:
                return z + state["msa_delta"]
            state["msa_exec"] += 1
            z_out = orig_msa_fwd(z, emb, f, *a, **kw)
            if r == msa_cache_after:
                state["msa_delta"] = (z_out - z).detach()
            return z_out

        def pf_hook(s, z, *a, **kw):
            r = state["pf_r"]
            state["pf_r"] += 1
            if state["pf_converged"]:
                return state["pf_conv_s"], state["pf_conv_z"]
            state["pf_exec"] += 1
            s_out, z_out = orig_pf_fwd(s, z, *a, **kw)
            if r >= 1 and state["prev_z"] is not None:
                z_pl_cur = z_out[:, prot_idx][:, :, lig_idx]
                z_pl_prev = state["prev_z"][:, prot_idx][:, :, lig_idx]
                if _rel_l1(z_pl_cur - z_pl_prev, z_pl_prev) < pf_adaptive_thresh:
                    state["pf_converged"] = True
                    state["pf_conv_s"] = s_out.detach()
                    state["pf_conv_z"] = z_out.detach()
            state["prev_z"] = z_out.detach()
            return s_out, z_out

        msa_mod.forward = msa_hook
        pf_mod.forward = pf_hook
        try:
            out = orig_model_fwd(feats, *args, **kwargs)
        finally:
            msa_mod.forward = orig_msa_fwd
            pf_mod.forward = orig_pf_fwd

        stats["msa_recycles"].append(state["msa_exec"])
        stats["pf_recycles"].append(state["pf_exec"])
        return out

    model.forward = wrapped_forward

    def restore():
        model.forward = orig_model_fwd

    return restore, stats
