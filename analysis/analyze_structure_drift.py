import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

root = Path(__file__).resolve().parent.parent / "results/anthropic"
parser = MMCIFParser(QUIET=True)
reference_path = root / "off_1/boltz_results_smoke/predictions/smoke/smoke_model_0.cif"
reference = list(parser.get_structure("stock", str(reference_path)).get_atoms())


def identity(atom):
    residue = atom.get_parent()
    return residue.get_parent().id, residue.id, atom.name, atom.altloc


def fit(moving, fixed, weights):
    weights = weights / weights.sum()
    a = (moving * weights[:, None]).sum(0)
    b = (fixed * weights[:, None]).sum(0)
    u, _, vt = np.linalg.svd(((moving - a) * weights[:, None]).T @ (fixed - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return a, b, u @ correction @ vt


keys = [identity(a) for a in reference]
xyz = np.array([a.coord for a in reference], dtype=np.float64)
confidence = np.array([a.bfactor for a in reference])
protein = np.array([a.get_parent().id[0] == " " for a in reference])
ca = protein & np.array([a.name == "CA" for a in reference])
backbone = protein & np.array([a.name in ("N", "CA", "C", "O") for a in reference])
ligand = ~protein
core = ca & (confidence >= 90)
pocket = protein & (np.linalg.norm(xyz[:, None] - xyz[ligand][None], axis=-1).min(1) <= 5)
pocket_residues = {keys[i][:2] for i in np.flatnonzero(pocket)}
pocket = np.array([key[:2] in pocket_residues for key in keys])
output = Path(__file__).resolve().parent.parent / "results/structure_drift"
output.mkdir(parents=True, exist_ok=True)
reports = []
for name in ("sm89_baseline_1", "sm89_opm_1", "sm89_full_0", "sm89_exact_0"):
    atoms = list(parser.get_structure(name, str(root / name / "by_seed/smoke/s0/smoke_model_0.cif")).get_atoms())
    assert [identity(a) for a in atoms] == keys
    moving = np.array([a.coord for a in atoms], dtype=np.float64)
    report = {"name": name, "alignments": {}}
    for align_name, weights in (("all_atom", np.ones(len(atoms))),
                                ("protein_ca", ca.astype(float)),
                                ("stock_plddt_weighted_ca", ca * confidence / 100),
                                ("stock_plddt90_ca", core.astype(float))):
        a, b, rotation = fit(moving, xyz, weights)
        delta = (moving - a) @ rotation + b - xyz
        squared = (delta * delta).sum(1)
        groups = {}
        for group, mask in (("all", np.ones(len(atoms), dtype=bool)), ("protein", protein),
                            ("protein_ca", ca), ("protein_backbone", backbone), ("ligand", ligand),
                            ("pocket_protein", pocket), ("pocket_ca", pocket & ca),
                            ("protein_plddt_lt50", protein & (confidence < 50)),
                            ("protein_plddt50_70", protein & (confidence >= 50) & (confidence < 70)),
                            ("protein_plddt70_90", protein & (confidence >= 70) & (confidence < 90)),
                            ("protein_plddt_ge90", protein & (confidence >= 90)),
                            ("protein_ca_plddt_ge90", core)):
            if mask.any():
                groups[group] = dict(atoms=int(mask.sum()), rmsd=float(np.sqrt(squared[mask].mean())),
                                     sse_fraction=float(squared[mask].sum()/squared.sum()) if squared.sum() else 0)
        groups["protein_plddt_weighted_rmsd"] = float(np.sqrt(np.average(squared[protein], weights=confidence[protein])))
        report["alignments"][align_name] = groups
        if align_name == "stock_plddt90_ca":
            residues = []
            for chain, residue_id in dict.fromkeys(key[:2] for key in keys):
                mask = np.array([key[:2] == (chain, residue_id) for key in keys])
                ca_mask = mask & ca
                residues.append(dict(chain=chain, residue=residue_id[1], name=reference[np.flatnonzero(mask)[0]].get_parent().resname,
                                     stock_plddt=float(confidence[mask].mean()), candidate_plddt=float(np.mean([atoms[i].bfactor for i in np.flatnonzero(mask)])),
                                     rmsd=float(np.sqrt(squared[mask].mean())), ca_displacement=float(np.sqrt(squared[ca_mask][0])) if ca_mask.any() else None,
                                     sse=float(squared[mask].sum()), pocket=bool((mask & pocket).any())))
            with (output / (name + "_residues.csv")).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(residues[0]))
                writer.writeheader()
                writer.writerows(residues)
            report["largest_residue_sse"] = sorted(residues, key=lambda r: r["sse"], reverse=True)[:15]
            report["pocket_residues"] = [r for r in residues if r["pocket"]]
    a, b, rotation = fit(moving[ligand], xyz[ligand], np.ones(int(ligand.sum())))
    report["ligand_self_aligned_rmsd"] = float(np.sqrt(np.mean(np.sum(((moving[ligand]-a) @ rotation + b - xyz[ligand])**2, axis=1))))
    reports.append(report)
summary = dict(reference="off_1", confidence="stock CIF B_iso_or_equiv, pLDDT 0-100; protein per residue, ligand per atom", core_ca_count=int(core.sum()), reports=reports)
(output / "comparison.json").write_text(json.dumps(summary, indent=2))
for report in reports:
    print(json.dumps({"name": report["name"], "all_atom_fit_rmsd": report["alignments"]["all_atom"]["all"]["rmsd"],
                      "core_fit": report["alignments"]["stock_plddt90_ca"], "weighted_ca_fit": report["alignments"]["stock_plddt_weighted_ca"],
                      "largest_residue_sse": report["largest_residue_sse"], "ligand_self_aligned_rmsd": report["ligand_self_aligned_rmsd"]}), flush=True)
