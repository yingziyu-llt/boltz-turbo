import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import yaml
from Bio.PDB import MMCIFParser, MMCIF2Dict
from Bio.SeqUtils import seq1

repo_root = Path(__file__).resolve().parent.parent
root = repo_root / "data/stability"
root.mkdir(parents=True, exist_ok=True)
manifest = []
for pdb, ligand_code in [("3PTB", "BEN"), ("1IEP", "STI"), ("1STP", "BTN")]:
    path = root / f"{pdb}.cif"
    path.write_bytes(urlopen(f"https://files.rcsb.org/download/{pdb}.cif", timeout=90).read())
    structure = MMCIFParser(QUIET=True).get_structure(pdb, path)
    candidates = []
    for chain in structure[0]:
        proteins = [r for r in chain if r.id[0] == " " and "CA" in r]
        for ligand in chain:
            if ligand.resname == ligand_code:
                candidates.append((chain.id, proteins, ligand))
    chain_id, proteins, ligand = candidates[0]
    assert proteins
    sequence = "".join(seq1(r.resname) for r in proteins)
    ccd_path = root / f"{ligand_code}.cif"
    ccd_path.write_bytes(urlopen(f"https://files.rcsb.org/ligands/download/{ligand_code}.cif", timeout=90).read())
    ccd = MMCIF2Dict.MMCIF2Dict(str(ccd_path))
    rows = zip(ccd["_pdbx_chem_comp_descriptor.type"], ccd["_pdbx_chem_comp_descriptor.program"], ccd["_pdbx_chem_comp_descriptor.descriptor"])
    smiles = [value for kind, program, value in rows if kind == "SMILES_CANONICAL" and program == "OpenEye OEToolkits"][0]
    # Observed experimental chain; no template or pocket information enters inference.
    payload = {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
        {"ligand": {"id": "B", "ccd": ligand_code}}],
        "properties": [{"affinity": {"binder": "B"}}]}
    (root / f"{pdb}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    protein_atoms = [(i + 1, a.name, a.coord.tolist()) for i, r in enumerate(proteins) for a in r if a.element != "H"]
    ligand_atoms = [(a.name, a.coord.tolist()) for a in ligand if a.element != "H"]
    manifest.append({"name": pdb, "pdb": pdb, "chain": chain_id, "ligand": ligand_code,
                     "sequence": sequence, "smiles": smiles, "protein_atoms": protein_atoms,
                     "ligand_atoms": ligand_atoms, "msa": "empty",
                     "reference_url": f"https://www.rcsb.org/structure/{pdb}"})
base = yaml.safe_load((repo_root / "data/smoke.yaml").read_text())
(root / "smoke.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
manifest.append({"name": "smoke", "msa": "precomputed", "experimental_reference": False})
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps([{k: v for k, v in m.items() if k not in ("protein_atoms", "ligand_atoms", "sequence")} for m in manifest], indent=2))
