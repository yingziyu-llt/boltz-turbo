import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIF2Dict

root = Path(__file__).resolve().parent.parent / "results/training_free"
reference = root / "stock_karras_200_s0"


def load_cif(folder):
    cif = MMCIF2Dict.MMCIF2Dict(str(folder / "boltz_results_smoke/predictions/smoke/smoke_model_0.cif"))
    xyz = np.column_stack([np.asarray(cif["_atom_site.Cartn_" + axis], dtype=float) for axis in "xyz"])
    keys = list(zip(cif["_atom_site.label_asym_id"], cif["_atom_site.label_seq_id"], cif["_atom_site.label_atom_id"]))
    chains = np.asarray(cif["_atom_site.label_asym_id"])
    elements = np.asarray(cif["_atom_site.type_symbol"])
    return xyz, keys, chains, elements


def align(mobile, target):
    x, y = mobile - mobile.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


baseline = json.loads((reference / "summary.json").read_text())
ref, ref_keys, chains, elements = load_cif(reference)
protein, ligand = chains == "A", chains == "B"
ca = protein & np.asarray([key[2] == "CA" for key in ref_keys])
rows = []
for path in sorted(root.glob("*/summary.json")):
    report = json.loads(path.read_text())
    coords, keys, candidate_chains, _ = load_cif(path.parent)
    assert keys == ref_keys
    rotation = align(coords[ca], ref[ca])
    fitted = (coords - coords[ca].mean(0)) @ rotation + ref[ca].mean(0)
    ligand_rotation = align(coords[ligand], ref[ligand])
    ligand_delta = (coords[ligand] - coords[ligand].mean(0)) @ ligand_rotation - (ref[ligand] - ref[ligand].mean(0))
    distances = np.linalg.norm(coords[ligand, None] - coords[None, protein], axis=-1)
    row = {"configuration": path.parent.name, "method": report["method"], "steps": report["steps"],
           "sample_seconds": {s["stage"]: s["warm_sample_seconds"] for s in report["stages"]},
           "denoiser_calls": {s["stage"]: s["denoiser_calls_first"] for s in report["stages"]},
           "sample_equivalent_nfe": {s["stage"]: s["sample_equivalent_nfe_first"] for s in report["stages"]},
           "scores": report["scores"],
           "score_abs_error": {k: abs(v - baseline["scores"][k]) for k, v in report["scores"].items()},
           "protein_ca_rmsd_angstrom": float(np.sqrt(np.mean(np.sum((fitted[ca] - ref[ca])**2, axis=-1)))),
           "ligand_protein_aligned_rmsd_angstrom": float(np.sqrt(np.mean(np.sum((fitted[ligand] - ref[ligand])**2, axis=-1)))),
           "ligand_self_aligned_rmsd_angstrom": float(np.sqrt(np.mean(np.sum(ligand_delta**2, axis=-1)))),
           "ligand_protein_pairs_below_1_2_angstrom": int((distances < 1.2).sum()),
           "peak_allocated_gib": {s["stage"]: s["peak_allocated_gib"] for s in report["stages"]}}
    row["combined_warm_sample_seconds"] = sum(row["sample_seconds"].values())
    rows.append(row)
result = {"reference": reference.name, "results": rows,
          "limitations": ["One unlabeled input and one seed; no enrichment or general quality conclusion.",
                          "Warm sampling timings exclude trunk, confidence, checkpoint load and I/O.",
                          "CLI and Trainer timings include an extra warm sampling repeat.",
                          "Geometry differences are against stock predictions, not experimental truth; atom naming has no symmetry correction.",
                          "Sub-1.2 A pair counts are a coarse overlap diagnostic, not PoseBusters validity."]}
(root / "comparison.json").write_text(json.dumps(result, indent=2))
for row in rows:
    print(json.dumps(row), flush=True)
