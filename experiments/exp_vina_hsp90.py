import argparse
import csv
import json
import math
import os
import re
import resource
import socket
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, process_time

import meeko
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = Path('/lustre3/lhlai_pkuhpc/llt/hsp90_affinity/hsp90_labels_primary.csv')
STRUCTURE_URL = 'https://files.rcsb.org/download/4YKR.pdb'
BOX_SIZE = (20.0, 20.0, 20.0)
EXHAUSTIVENESS = 8
SEED = 0
SINGLE_CORE_SAMPLE = 12


def prepare_receptor(source, out_dir):
    from openmm import Platform
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    coords, protein = [], []
    for line in source.read_text().splitlines():
        if line[21:22] != 'A' or line[16] not in (' ', 'A') or line[76:78].strip() in ('H', 'D'):
            continue
        if line.startswith('ATOM  '):
            protein.append(line[:16] + ' ' + line[17:])
        elif line.startswith('HETATM') and line[17:27] == '4EP A 301 ':
            coords.append([float(line[i:i + 8]) for i in (30, 38, 46)])
    assert len(coords) == 21 and protein, 'Expected the 21 heavy atoms of 4YKR chain A 4EP:301 and chain A protein'
    center = np.mean(coords, axis=0).tolist()
    chain = out_dir / '4YKR_chain_A.pdb'
    chain.write_text('\n'.join(protein) + '\nTER\nEND\n')

    # Repair unresolved side-chain heavy atoms and OXT without building missing residues.
    fixer = PDBFixer(filename=str(chain), platform=Platform.getPlatformByName('Reference'))
    fixer.findMissingResidues()
    fixer.missingResidues.clear()
    fixer.findMissingAtoms()
    repaired = {}
    for residue, atoms in [*fixer.missingAtoms.items(), *fixer.missingTerminals.items()]:
        key = f'{residue.chain.id}:{residue.id}:{residue.name}'
        repaired.setdefault(key, []).extend(a if isinstance(a, str) else a.name for a in atoms)
    fixer.addMissingAtoms(seed=1)  # OpenMM uses zero to request an automatic seed.
    fixed = out_dir / '4YKR_chain_A_fixed.pdb'
    with fixed.open('w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    receptor = out_dir / '4YKR_chain_A.pdbqt'
    subprocess.run(['mk_prepare_receptor.py', '--read_pdb', str(fixed), '-p', str(receptor)], check=True)
    assert receptor.stat().st_size > 0
    return receptor, center, repaired


def dock_one(ligand_id, smiles, vina_args):
    t0, cpu0 = perf_counter(), process_time()
    children0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    stage = 'smiles'
    record = {'ligand_id': ligand_id, 'vina_score': None, 'vina_effective_seed': None}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'RDKit could not parse SMILES: {smiles}')
        mol = Chem.AddHs(mol)
        stage = 'etkdg'
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        params.numThreads = 1
        if AllChem.EmbedMolecule(mol, params) < 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) < 0:
                raise ValueError('ETKDGv3 failed with both distance geometry and random coordinates')
        record['etkdg_random_coords'] = params.useRandomCoords
        stage = 'force_field'
        if AllChem.MMFFHasAllMoleculeParams(mol):
            record['force_field'] = 'MMFF94'
            opt = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        else:
            record['force_field'] = 'UFF'
            opt = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        if opt != 0:
            raise ValueError(f'{record["force_field"]} did not converge: {opt}')
        stage = 'ligand_pdbqt'
        setup, = meeko.MoleculePreparation().prepare(mol)
        pdbqt, ok, error = meeko.PDBQTWriterLegacy.write_string(setup)
        if not ok:
            raise ValueError(error)

        with tempfile.TemporaryDirectory(prefix='hsp90-vina-') as tmp:
            ligand = Path(tmp) / 'ligand.pdbqt'
            poses = Path(tmp) / 'poses.pdbqt'
            ligand.write_text(pdbqt)
            stage = 'vina'
            proc = subprocess.run([*vina_args, '--ligand', str(ligand), '--out', str(poses)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f'Vina exit {proc.returncode}:\n{proc.stdout}')
            seed = re.search(r'random seed:\s*(-?\d+)', proc.stdout, re.IGNORECASE)
            if seed is None:
                raise RuntimeError(f'Vina did not report its effective seed:\n{proc.stdout}')
            record['vina_effective_seed'] = int(seed[1])
            stage = 'vina_score'
            scores = [float(line.split()[3]) for line in poses.read_text().splitlines()
                      if line.startswith('REMARK VINA RESULT:')]
            if not scores or not all(map(math.isfinite, scores)):
                raise ValueError(f'Missing or non-finite Vina scores: {scores}\n{proc.stdout}')
            record.update(status='ok', vina_score=min(scores))
    except Exception:
        record.update(status='failed', error_stage=stage, error=traceback.format_exc())
    children1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = process_time() - cpu0 + children1.ru_utime - children0.ru_utime + children1.ru_stime - children0.ru_stime
    return record | {'wall_seconds': perf_counter() - t0, 'cpu_core_seconds': cpu}


def rate(seconds, n):
    return {'seconds_per_mol': seconds / n if n else None, 'mol_per_second': n / seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-csv', type=Path, default=LABELS_CSV)
    parser.add_argument('--dir', type=Path, default=REPO_ROOT / 'results/hsp90_vina')
    parser.add_argument('--receptor-source', type=Path)
    args = parser.parse_args()
    job_id = int(os.environ['SLURM_JOB_ID'])
    cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    node = socket.gethostname()
    assert not node.startswith('login'), f'Docking requires a Slurm compute node: {node}'
    with args.labels_csv.open(newline='') as f:
        rows = [(r['ligand_id'], r['smiles']) for r in csv.DictReader(f)]
    assert len(rows) == len({k for k, _ in rows}) == 180

    out_dir = args.dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = args.receptor_source or out_dir / 'inputs/4YKR.pdb'
    vina_version = subprocess.check_output(['vina', '--version'], text=True).strip()
    print(f'JOB_ID={job_id} NODE={node} CORES={cores} {vina_version}', flush=True)
    t0 = perf_counter()
    receptor, center, repaired = prepare_receptor(source, out_dir)
    receptor_wall = perf_counter() - t0
    print(f'4YKR chain A, 4EP heavy-atom center={center}, box={BOX_SIZE}; repaired={repaired}', flush=True)
    vina_args = ['vina', '--receptor', str(receptor), '--cpu', '1', '--exhaustiveness', str(EXHAUSTIVENESS), '--seed', str(SEED)]
    for name, values in (('center', center), ('size', BOX_SIZE)):
        for axis, value in zip('xyz', values):
            vina_args.extend([f'--{name}_{axis}', str(value)])

    records = []
    steady_start = perf_counter()
    with (out_dir / 'scores.jsonl').open('w', buffering=1) as stream, ProcessPoolExecutor(max_workers=cores) as pool:
        jobs = [pool.submit(dock_one, k, smiles, vina_args) for k, smiles in rows]
        for done, future in enumerate(as_completed(jobs), 1):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if record['status'] != 'ok':
                print(f"FAILED {record['ligand_id']} ({record['error_stage']}): {record['error']}", file=sys.stderr, flush=True)
            if done % 20 == 0:
                print(f'{done}/{len(rows)} finished in {perf_counter() - steady_start:.1f}s', flush=True)
    steady_wall = perf_counter() - steady_start
    wall = perf_counter() - t0

    # Measure serial throughput with no parallel jobs on the allocation, on an evenly spaced subset.
    print(f'Running {SINGLE_CORE_SAMPLE} extra ligands serially for single-core throughput', flush=True)
    sample_t0 = perf_counter()
    sample = [dock_one(*rows[int(i)], vina_args)
              for i in np.linspace(0, len(rows) - 1, SINGLE_CORE_SAMPLE, dtype=int)]
    sample_wall = perf_counter() - sample_t0
    for rec in sample:
        if rec['status'] != 'ok':
            print(f"FAILED single-core timing {rec['ligand_id']} ({rec['error_stage']}): {rec['error']}", file=sys.stderr, flush=True)
    scored = sum(r['status'] == 'ok' for r in records)
    sample_scored = sum(r['status'] == 'ok' for r in sample)
    cpu = sum(r['cpu_core_seconds'] for r in records)
    seeds = {r['vina_effective_seed'] for r in records if r['vina_effective_seed'] is not None}
    summary = {
        'job_id': job_id, 'node': node, 'partition': os.environ['SLURM_JOB_PARTITION'],
        'cpu_cores': cores, 'vina_version': vina_version, 'environment': sys.prefix,
        'rdkit_version': rdBase.rdkitVersion, 'meeko_version': meeko.__version__,
        'receptor_pdb': '4YKR', 'receptor_source': str(source), 'receptor_source_url': STRUCTURE_URL,
        'box_ligand': '4EP A:301 (21 heavy atoms; centroid only)', 'box_center': center, 'box_size': BOX_SIZE,
        'receptor_preparation': 'Protein chain A only; blank/A altloc; strip input hydrogen, water and all HETATM. PDBFixer repairs missing heavy atoms/OXT without adding residues; Meeko adds H and assigns protonation/atom types/charges from residue templates (no explicit pH titration).',
        'receptor_repaired_atoms': repaired,
        'ligand_preparation': 'CSV SMILES only; preserve input formal charges and tautomer, add H with RDKit, ETKDGv3, MMFF94 (UFF if MMFF94 parameters unavailable), Meeko PDBQT.',
        'exhaustiveness': EXHAUSTIVENESS, 'seed': SEED,
        'seed_note': 'RDKit seed is fixed at 0. Vina receives --seed 0, which requests automatic seeds; the effective seeds are recorded in scores.jsonl.',
        'distinct_vina_effective_seeds': len(seeds),
        'total_ligands': len(rows), 'failed_count': len(rows) - scored, 'coverage': scored / len(rows),
        'wall_seconds': wall, 'receptor_wall_seconds': receptor_wall, 'steady_wall_seconds': steady_wall,
        'cpu_core_seconds_per_mol': rate(cpu, scored)['seconds_per_mol'],
        'wall_seconds_per_mol': rate(wall, scored)['seconds_per_mol'],
        'throughput': {
            'steady_multicore': rate(steady_wall, scored), 'end_to_end_multicore': rate(wall, scored),
            'single_core_measured': rate(sample_wall, sample_scored), 'cpu_core_work': rate(cpu, scored),
        },
        'single_core_sample_wall_seconds': sample_wall,
        'single_core_sample_ids': [r['ligand_id'] for r in sample],
        'single_core_sample_failed_count': len(sample) - sample_scored,
        'single_core_sample_failures': [r for r in sample if r['status'] != 'ok'],
        'throughput_notes': 'Steady multicore includes ligand preparation, Vina process launches and worker pool overhead, excludes one-time receptor preparation; end-to-end adds receptor preparation. Single-core measured on an evenly spaced CSV subset, run serially after multicore. CPU-core work sums worker and Vina CPU time; its inverse is molecules per CPU-core-second. Timings exclude queue, evaluation and staging, and multicore timings exclude the extra serial benchmark.',
        'remote_results': str(out_dir),
        'information_asymmetry': 'Vina uses a 4YKR holo crystal receptor and a known binding-site center from its co-crystallized ligand. Boltz-2 predicts from sequence and ligand SMILES without the crystal structure or pocket coordinates. No ligand pose coordinates were used as Vina input.',
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Vina coverage: {scored}/{len(rows)}, failed={len(rows) - scored}; multicore wall={wall:.3f}s, serial sample wall={sample_wall:.3f}s', flush=True)


if __name__ == '__main__':
    main()
