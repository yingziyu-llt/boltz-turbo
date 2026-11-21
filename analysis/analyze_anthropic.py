import json
import math
from pathlib import Path
import re

import numpy as np
from Bio.PDB import MMCIFParser

root = Path(__file__).resolve().parent.parent / 'results/anthropic'
parser = MMCIFParser(QUIET=True)
reference_dir = root / 'off_1/boltz_results_smoke/predictions/smoke'
reference_scores = json.loads((reference_dir / 'affinity_smoke.json').read_text())
reference_cif = reference_dir / 'smoke_model_0.cif'
reference_xyz = np.array([a.coord for a in parser.get_structure('reference', str(reference_cif)).get_atoms()])
rows = []
for name in ['off_0', 'off_1', 'exact_0', 'exact_1', 'fast_0', 'fast_1', 'big_0', 'big_1', 'big_x2'] + [f'adapt_{m}_{r}' for m in ('exact', 'fast', 'big', 'big_x2') for r in (0, 1)] + [f'sm89_{m}_{r}' for m in ('baseline', 'opm', 'full', 'exact') for r in (0, 1)]:
    log_path = root / (name + '.log')
    if not log_path.exists():
        continue
    text = log_path.read_text()
    row = {'name': name, 'wall_seconds': float(re.findall(r'^real (\S+)', text, re.M)[-1]),
           'refusals': [x for x in text.splitlines() if 'NOT ACTIVE:' in x and 'mode off:' not in x],
           'phases': [x for x in text.splitlines() if 'PHASE item=' in x],
           'levers': [x for x in text.splitlines() if '[boltz2-opt] LEVER name=' in x],
           'exit': [x for x in text.splitlines() if '[boltz2-opt] EXIT ' in x]}
    prediction = root / name / ('boltz_results_smoke/predictions/smoke' if name.startswith('off') else 'by_seed/smoke/s0')
    affinity = prediction / 'affinity_smoke.json'
    cif = prediction / 'smoke_model_0.cif'
    row['complete_outputs'] = affinity.exists() and cif.exists()
    if row['complete_outputs']:
        scores = json.loads(affinity.read_text())
        assert all(math.isfinite(x) for x in scores.values())
        assert 0 <= scores['affinity_probability_binary'] <= 1
        row['scores'] = scores
        row['score_abs_errors'] = {k: abs(v - reference_scores[k]) for k, v in scores.items()}
        row['cif_byte_identical'] = cif.read_bytes() == reference_cif.read_bytes()
        xyz = np.array([a.coord for a in parser.get_structure(name, str(cif)).get_atoms()])
        assert xyz.shape == reference_xyz.shape
        row['raw_coordinate_rms_error_angstrom'] = float(np.sqrt(np.mean((xyz-reference_xyz)**2)))
        # 刚体对齐后比较同序原子，避免全局旋转掩盖内部结构一致性。
        a = xyz.astype(np.float64) - xyz.mean(0)
        b = reference_xyz.astype(np.float64) - reference_xyz.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        d = np.eye(3)
        d[-1, -1] = np.linalg.det(u @ vt)
        row['aligned_all_atom_rmsd_angstrom'] = float(np.sqrt(np.mean(np.sum((a @ u @ d @ vt - b)**2, axis=1))))
    rows.append(row)
report = {'reference': 'off_1', 'kit_commit': 'f4f62fa6592ae4938d49b1757bea0cfeff9f468e', 'rows': rows}
(root / 'comparison.json').write_text(json.dumps(report, indent=2))
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k not in ('phases','levers','scores','refusals')}, ensure_ascii=False))
    if row['refusals']:
        print('REFUSAL', *row['refusals'], sep='\n')
