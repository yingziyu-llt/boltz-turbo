import itertools
import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, MMCIF2Dict
from rdkit import Chem

repo_root = Path(__file__).resolve().parent.parent
root = repo_root / "results/stability"
manifest = {x["name"]: x for x in json.loads((repo_root / "data/stability/manifest.json").read_text())}
parser = MMCIFParser(QUIET=True)


def load(folder, system):
    path = folder / f"boltz_results_{system}/predictions/{system}/{system}_model_0.cif"
    atoms = list(parser.get_structure("prediction", path).get_atoms())
    keys = [(a.get_parent().get_parent().id, a.get_parent().id[1], a.name) for a in atoms]
    return np.array([a.coord for a in atoms], dtype=float), keys, np.array([a.bfactor for a in atoms])


def fitted(moving, reference, mask):
    a, b = moving[mask].mean(0), reference[mask].mean(0)
    u, _, vt = np.linalg.svd((moving[mask] - a).T @ (reference[mask] - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return (moving - a) @ (u @ correction @ vt) + b


def rms(delta):
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=-1))))


rows = []
coverage = []
pairwise = []
for system, info in manifest.items():
    folders = sorted((root / system).glob("*/summary.json"))
    records = []
    for path in folders:
        report = json.loads(path.read_text())
        if "scores" not in report:
            continue
        records.append((path.parent, report))
    coverage.append({"system": system, "completed": len(records), "expected": 15})
    baseline = {r["seed"]: f for f, r in records if r["method"] == "stock"}
    if 0 not in baseline:
        continue
    anchor, keys, plddt = load(baseline[0], system)
    protein = np.array([k[0] == "A" for k in keys])
    ligand = np.array([k[0] == "B" for k in keys])
    ca = protein & np.array([k[2] == "CA" for k in keys])
    core = ca & (plddt >= 90)
    pocket_atoms = protein & (np.linalg.norm(anchor[:, None] - anchor[ligand][None], axis=-1).min(1) <= 5)
    residues = {keys[i][:2] for i in np.flatnonzero(pocket_atoms)}
    pocket = protein & np.array([k[:2] in residues for k in keys])
    groups = {"high_confidence_all_atom": protein & (plddt >= 90), "high_confidence_ca": core,
              "pocket": pocket, "ligand": ligand, "plddt_lt70": protein & (plddt < 70)}
    coordinate_rows = []
    for folder, report in records:
        xyz, candidate_keys, confidence = load(folder, system)
        assert candidate_keys == keys
        method = "stock200" if report["method"] == "stock" else f"dpm{report['steps']}"
        row = {"system": system, "seed": report["seed"], "method": method,
               "core_ca_count": int(core.sum()),
               "scores": report["scores"],
               "cli_seconds": report["cli_seconds_including_warm_repeat"],
               "sampling_seconds": sum(s["sample_seconds"] for s in report["stages"]),
               "peak_allocated_gib": max(s["peak_allocated_gib"] for s in report["stages"]),
               "protein_mean_plddt": float(confidence[protein].mean()),
               "ligand_mean_plddt": float(confidence[ligand].mean())}
        if core.sum() >= 3:
            reference, _, _ = load(baseline[report["seed"]], system)
            aligned = fitted(xyz, reference, core)
            row["paired_stock_rmsd_A"] = {name: rms((aligned - reference)[mask])
                                            for name, mask in groups.items() if mask.any()}
            coordinate_rows.append((method, report["seed"], xyz))
        else:
            row["paired_stock_rmsd_A"] = None
        distances = np.linalg.norm(xyz[ligand, None] - xyz[None, protein], axis=-1)
        row["ligand_protein_pairs_below_1_2A"] = int((distances < 1.2).sum())
        if "pdb" in info:
            reference_atoms = {(int(i), name): np.array(coord) for i, name, coord in info["protein_atoms"]}
            ref_ligand = {name: np.array(coord) for name, coord in info["ligand_atoms"]}
            reference = xyz.copy()
            common_ca = np.array([k[0] == "A" and k[2] == "CA" and (k[1], k[2]) in reference_atoms for k in keys])
            for i, key in enumerate(keys):
                if key[0] == "A" and (key[1], key[2]) in reference_atoms:
                    reference[i] = reference_atoms[key[1], key[2]]
                elif key[0] == "B" and key[2] in ref_ligand:
                    reference[i] = ref_ligand[key[2]]
            aligned = fitted(xyz, reference, common_ca)
            common_ligand = ligand & np.array([k[2] in ref_ligand for k in keys])
            assert common_ligand.sum() == len(info["ligand_atoms"])
            row["experimental_protein_ca_rmsd_A"] = rms((aligned - reference)[common_ca])
            row["experimental_ligand_rmsd_A"] = rms((aligned - reference)[common_ligand])
            self_fit = fitted(xyz, reference, common_ligand)
            row["experimental_ligand_self_rmsd_A"] = rms((self_fit - reference)[common_ligand])
            common_protein = protein & np.array([(k[1], k[2]) in reference_atoms for k in keys])
            distances_to_ligand = np.linalg.norm(reference[:, None] - reference[common_ligand][None], axis=-1).min(1)
            exp_pocket_residues = {keys[i][:2] for i in np.flatnonzero(common_protein & (distances_to_ligand <= 5))}
            exp_pocket = common_protein & np.array([k[:2] in exp_pocket_residues for k in keys])
            row["experimental_pocket_rmsd_A"] = rms((aligned - reference)[exp_pocket])
            row["experimental_high_confidence_ca_rmsd_A"] = rms((aligned - reference)[common_ca & (plddt >= 90)])
            ccd = MMCIF2Dict.MMCIF2Dict(str(repo_root / "data/stability" / f"{info['ligand']}.cif"))
            index = {key[2]: i for i, key in enumerate(keys) if key[0] == "B"}
            bonds = [(a, b) for a, b in zip(ccd["_chem_comp_bond.atom_id_1"], ccd["_chem_comp_bond.atom_id_2"])
                     if a in index and b in index and a in ref_ligand and b in ref_ligand]
            errors = [abs(np.linalg.norm(xyz[index[a]] - xyz[index[b]]) - np.linalg.norm(ref_ligand[a] - ref_ligand[b])) for a, b in bonds]
            row["experimental_ligand_bond_mae_A"] = float(np.mean(errors))
            row["experimental_ligand_bond_max_error_A"] = float(max(errors))
            symbols = dict(zip(ccd["_chem_comp_atom.atom_id"], ccd["_chem_comp_atom.type_symbol"]))
            charges = dict(zip(ccd["_chem_comp_atom.atom_id"], ccd["_chem_comp_atom.charge"]))
            names = list(ref_ligand)
            graph = Chem.RWMol()
            for atom_name in names:
                atom = Chem.Atom(symbols[atom_name])
                atom.SetFormalCharge(int(charges[atom_name]))
                graph.AddAtom(atom)
            bond_types = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
                          "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC}
            for a, b, order in zip(ccd["_chem_comp_bond.atom_id_1"], ccd["_chem_comp_bond.atom_id_2"], ccd["_chem_comp_bond.value_order"]):
                if a in names and b in names:
                    graph.AddBond(names.index(a), names.index(b), bond_types[order])
            graph = graph.GetMol()
            Chem.SanitizeMol(graph)
            permutations = graph.GetSubstructMatches(graph, uniquify=False, maxMatches=10000)
            moving_ligand = np.array([aligned[index[name]] for name in names])
            experimental_ligand = np.array([ref_ligand[name] for name in names])
            row["experimental_ligand_symmetry_rmsd_A"] = min(rms(moving_ligand - experimental_ligand[list(p)]) for p in permutations)
            row["ligand_symmetry_permutations"] = len(permutations)
        rows.append(row)
    if core.sum() >= 3:
        for method in ("stock200", "dpm16", "dpm32"):
            selected = [x for x in coordinate_rows if x[0] == method]
            for (_, seed_a, a), (_, seed_b, b) in itertools.combinations(selected, 2):
                delta = fitted(a, b, core) - b
                pairwise.append({"system": system, "method": method, "seeds": [seed_a, seed_b],
                                 "rmsd_A": {name: rms(delta[mask]) for name, mask in groups.items() if mask.any()}})
aggregates = []
for system, method in itertools.product(manifest, ("stock200", "dpm16", "dpm32")):
    subset = [r for r in rows if r["system"] == system and r["method"] == method]
    if not subset:
        continue
    record = {"system": system, "method": method, "n": len(subset)}
    values = {"probability": [r["scores"]["affinity_probability_binary"] for r in subset],
              "cli_seconds": [r["cli_seconds"] for r in subset],
              "sampling_seconds": [r["sampling_seconds"] for r in subset]}
    for key in ("experimental_ligand_rmsd_A", "experimental_ligand_symmetry_rmsd_A", "experimental_ligand_bond_mae_A",
                "experimental_pocket_rmsd_A", "experimental_high_confidence_ca_rmsd_A"):
        if key in subset[0]:
            values[key] = [r[key] for r in subset]
    for group in groups:
        v = [r["paired_stock_rmsd_A"][group] for r in subset if r["paired_stock_rmsd_A"] is not None and group in r["paired_stock_rmsd_A"]]
        if v:
            values["paired_stock_" + group] = v
    for key, value in values.items():
        record[key] = {"median": float(np.median(value)), "min": float(min(value)), "max": float(max(value)),
                       "std": float(np.std(value, ddof=1)) if len(value) > 1 else None}
    aggregates.append(record)
result = {"coverage": coverage, "aggregates": aggregates, "rows": rows, "cross_seed_pairs": pairwise,
          "limitations": ["Three historical PDB complexes may overlap model training; not a held-out generalization benchmark.",
                          "New systems use observed monomer sequences and empty MSA; native oligomers, waters and non-binder cofactors are omitted.",
                          "Experimental symmetry RMSD minimizes over CCD graph automorphisms; direct-name RMSD is also retained. No stereo-aware atom permutation constraints are applied.",
                          "Fixed confidence and pocket masks come from each system's stock seed0, never the candidate.",
                          "No core alignment metric is reported when fewer than three pLDDT>=90 CA exist.",
                          "Bond deviations and close-contact counts are diagnostics, not PoseBusters validity.",
                          "No active/inactive labels, early enrichment or screening recall measured.",
                          "CLI includes checkpoint loading; these are single-process cold/warm-cache latencies, not resident throughput."]}
(root / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps({"coverage": coverage, "aggregates": aggregates}, indent=2))
