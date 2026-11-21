import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

root = Path(__file__).resolve().parent.parent / "results/training_free"
parser = MMCIFParser(QUIET=True)
stock_dir = root / "stock_karras_200_s0"
stock_atoms = list(parser.get_structure("stock", str(stock_dir / "boltz_results_smoke/predictions/smoke/smoke_model_0.cif")).get_atoms())


def identity(atom):
    residue = atom.get_parent()
    return residue.get_parent().id, residue.id, atom.name


keys = [identity(a) for a in stock_atoms]
stock = np.array([a.coord for a in stock_atoms], dtype=float)
confidence = np.array([a.bfactor for a in stock_atoms])
protein = np.array([a.get_parent().id[0] == " " for a in stock_atoms])
ca = protein & np.array([a.name == "CA" for a in stock_atoms])
ligand = ~protein
core = ca & (confidence >= 90)
pocket_atoms = protein & (np.linalg.norm(stock[:, None] - stock[ligand][None], axis=-1).min(1) <= 5)
pocket_residues = {keys[i][:2] for i in np.flatnonzero(pocket_atoms)}
pocket = protein & np.array([key[:2] in pocket_residues for key in keys])


def metrics(moving, reference):
    x, y = moving[core], reference[core]
    a, b = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - a).T @ (y - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    fitted = (moving - a) @ (u @ correction @ vt) + b
    sq = np.sum((fitted - reference)**2, axis=-1)
    groups = {"all": np.ones(len(keys), dtype=bool), "high_confidence_all_atom": protein & (confidence >= 90),
              "high_confidence_ca": core, "plddt70_90": protein & (confidence >= 70) & (confidence < 90),
              "plddt_lt70": protein & (confidence < 70), "pocket": pocket, "ligand": ligand}
    return {name: {"atoms": int(mask.sum()), "rmsd_A": float(np.sqrt(sq[mask].mean())),
                   "p95_displacement_A": float(np.percentile(np.sqrt(sq[mask]), 95))}
            for name, mask in groups.items() if mask.any()}


reports = []
for folder in sorted(root.glob("dpm2m_*")):
    path = folder / "summary.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    if "dpm_parameters" not in report:
        continue
    atoms = list(parser.get_structure("candidate", str(folder / "boltz_results_smoke/predictions/smoke/smoke_model_0.cif")).get_atoms())
    assert [identity(a) for a in atoms] == keys
    item = {"configuration": folder.name, "seed": report["seed"],
            "versus_stock": metrics(np.array([a.coord for a in atoms], dtype=float), stock)}
    reference_path = folder / "structure_heun512.npy"
    if reference_path.exists():
        mask = np.load(folder / "structure_coords.npz")["mask"].astype(bool)
        reference = np.load(reference_path)[0, mask]
        assert reference.shape == stock.shape
        item["versus_same_condition_heun512"] = {}
        for candidate in sorted(folder.glob("structure_*.npy")):
            coordinates = np.load(candidate)[0, mask]
            item["versus_same_condition_heun512"][candidate.stem.removeprefix("structure_")] = metrics(coordinates, reference)
    reports.append(item)
result = {"confidence": "Fixed stock CIF pLDDT, threshold >=90; Kabsch fit on stock high-confidence protein CA",
          "core_ca_count": int(core.sum()), "reports": reports,
          "limitations": ["Structure stage only; affinity has different cropped atom identities and is not mapped here.",
                          "Same-condition reference is numerical Heun512, not experimental truth.",
                          "Stock-confidence labels are fixed regions, not confidence predictions of Heun512.",
                          "One complex; stock seed 0 defines masks and the stock comparison for both candidate seeds.",
                          "Trajectory atom order is taken from Boltz output order after padding removal; only atom-count agreement is checked for trajectory arrays."]}
(root / "dpm_confidence.json").write_text(json.dumps(result, indent=2))
for row in reports:
    print(json.dumps(row))
