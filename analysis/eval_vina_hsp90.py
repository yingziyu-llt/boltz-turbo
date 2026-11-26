import argparse
import json
from pathlib import Path

import numpy as np

from eval_hsp90 import (
    auroc,
    average_precision,
    enrichment_and_recall,
    kendall_tau_b,
    load_hsp90_labels,
    pearson,
    spearman,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = Path('/lustre3/lhlai_pkuhpc/llt/hsp90_affinity/hsp90_labels_primary.csv')
QUANT_FIELDS = ('pearson_r', 'spearman_rho', 'kendall_tau', 'crmse_kcal', 'raw_rmse_kcal')
SCREEN_FIELDS = ('auroc', 'ap', 'ef10', 'recall10')


def cohort_metrics(ids, labels, scores):
    # Match eval_hsp90's sorted ligand order, including stable ties in AP and top 10%.
    ids = sorted(ids)
    cohort = [labels[k] for k in ids]
    eq = np.array([r['relation'] == '=' for r in cohort], dtype=bool)
    log10 = np.array([r['boltz_log10_IC50_uM'] for r in cohort])
    quant = eq & np.isfinite(log10)
    pred = np.array([scores[k] for k in ids])
    ic50 = np.array([r['ic50_uM'] for r in cohort])
    exp_dg = 1.364 * log10[quant] - 8.184
    pred_dg = pred[quant]
    r, rho, tau = (fn(pred_dg, exp_dg) for fn in (pearson, spearman, kendall_tau_b))
    if pred_dg.size:
        crmse = float(np.sqrt(np.mean(((pred_dg - pred_dg.mean()) - (exp_dg - exp_dg.mean())) ** 2)))
        rmse = float(np.sqrt(np.mean((pred_dg - exp_dg) ** 2)))
    else:
        crmse = rmse = None
    metrics = dict(zip(QUANT_FIELDS, (r, rho, tau, crmse, rmse)))
    metrics.update(total_ligands=len(ids), quant_ligands=int(quant.sum()))
    for key, threshold in (('screen_1uM', 1.0), ('screen_100nM', 0.1)):
        binary = (eq & (ic50 <= threshold)).astype(int)
        ef, recall = enrichment_and_recall(-pred, binary, 0.10)
        metrics[key] = dict(zip(SCREEN_FIELDS, (auroc(-pred, binary), average_precision(-pred, binary), ef, recall)))
        metrics[key]['actives'] = int(binary.sum())
    per_chemotype = {}
    chemotypes = np.array([row['chemotype'] for row in cohort], dtype=str)[quant]
    for name in sorted(set(chemotypes)):
        mask = chemotypes == name
        if mask.sum() >= 3:
            per_chemotype[name] = {
                'n_quant': int(mask.sum()),
                'pearson_r': pearson(pred_dg[mask], exp_dg[mask]),
                'spearman_rho': spearman(pred_dg[mask], exp_dg[mask]),
            }
    return metrics, per_chemotype


def fmt(value):
    return 'n/a' if value is None or not np.isfinite(value) else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default=REPO_ROOT / 'results/hsp90_vina')
    parser.add_argument('--labels-csv', type=Path, default=LABELS_CSV)
    parser.add_argument('--boltz-dir', type=Path, default=REPO_ROOT / 'results')
    args = parser.parse_args()
    labels = load_hsp90_labels(args.labels_csv)
    summary = json.loads((args.dir / 'summary.json').read_text())
    records = [json.loads(line) for line in (args.dir / 'scores.jsonl').read_text().splitlines()]
    assert len(records) == len(labels) == 180 and {r['ligand_id'] for r in records} == labels.keys()
    scores = {r['ligand_id']: r['vina_score'] for r in records if r['status'] == 'ok'}
    assert all(np.isfinite(v) for v in scores.values()), 'Non-finite Vina score'
    assert len(labels) - len(scores) == summary['failed_count']

    boltz = {suite: json.loads((args.boltz_dir / f'hsp90_a800_{suite}_status.json').read_text())
             for suite in ('fair', 'decoupled')}
    assert set(boltz['fair']['metadata']['ligand_names']) == labels.keys()
    subset = set(boltz['decoupled']['metadata']['ligand_names'])
    assert len(subset) == 60 and subset <= labels.keys()
    metrics, per_chemotype = cohort_metrics(scores, labels, scores)
    matched, matched_per_chemotype = cohort_metrics(subset & scores.keys(), labels, scores)
    if len(scores) == 180:
        assert (metrics['quant_ligands'], metrics['screen_1uM']['actives'], metrics['screen_100nM']['actives']) == (147, 73, 41)
    if matched['total_ligands'] == 60:
        assert (matched['quant_ligands'], matched['screen_1uM']['actives'], matched['screen_100nM']['actives']) == (48, 25, 14)

    comparisons = [{'method': 'Vina', 'cohort': 'fair', 'metrics': metrics,
                    'per_chemotype': per_chemotype, 'screening_score': '-vina_score',
                    'steady_s_per_mol': summary['throughput']['steady_multicore']['seconds_per_mol'],
                    'steady_mol_per_s': summary['throughput']['steady_multicore']['mol_per_second']},
                   {'method': 'Vina', 'cohort': 'decoupled', 'metrics': matched,
                    'per_chemotype': matched_per_chemotype, 'screening_score': '-vina_score',
                    'steady_s_per_mol': None, 'steady_mol_per_s': None}]
    for suite, configs in (('fair', ('stock_200', 'dpm2m_16')),
                           ('decoupled', ('stock_200', 'dpm2m_16', 'dpm2m_16_affrec1'))):
        by_config = {row['config']: row for row in boltz[suite]['rows']}
        for config in configs:
            row = by_config[config]
            comparisons.append({
                'method': config, 'cohort': suite, 'screening_score': '-affinity_pred_value',
                'metrics': {key: row[f'quant_{key}'] for key in QUANT_FIELDS} | {
                    'total_ligands': row['total_ligands'], 'quant_ligands': row['quant_ligands'],
                    **{f'screen_{name}': row[f'screen_{name}_by_neg_val'] for name in ('1uM', '100nM')},
                },
                'screening_by_probability': {name: row[f'screen_{name}_by_prob'] for name in ('1uM', '100nM')},
                'per_chemotype': row['per_chemotype'],
                'steady_s_per_mol': row['median_gpu_seconds_per_molecule'],
                'steady_mol_per_s': row['median_molecules_per_second'],
            })
    comparisons.sort(key=lambda row: row['cohort'] != 'fair')

    print(f"Vina: {len(scores)}/{len(labels)} scored, failed={summary['failed_count']}, coverage={summary['coverage']:.2%}")
    if summary['failed_count']:
        print('Vina metrics use only scored ligands; affected cohorts do not exactly match the Boltz-2 cohorts.')
    print('\n| Cohort | Method | n | Quant n | r | ρ | τ | cRMSE | raw RMSE | steady s/mol | steady mol/s |')
    print('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in comparisons:
        m = row['metrics']
        values = [m[k] for k in QUANT_FIELDS] + [row['steady_s_per_mol'], row['steady_mol_per_s']]
        print(f"| {row['cohort']} | {row['method']} | {m['total_ligands']} | {m['quant_ligands']} | "
              + ' | '.join(map(fmt, values)) + ' |')
    print('\nScreening by descending -Vina score / -Boltz-2 predicted affinity; Boltz-2 probability shown separately.')
    for key, threshold in (('1uM', '≤1 µM'), ('100nM', '≤100 nM')):
        print(f'\n| {threshold} cohort | Method | Rank by | Actives | AUROC | AP | EF@10% | Recall@10% |')
        print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |')
        for row in comparisons:
            screens = [(row['screening_score'], row['metrics'][f'screen_{key}'])]
            if 'screening_by_probability' in row:
                screens.append(('probability', row['screening_by_probability'][key]))
            for rank_by, screen in screens:
                print(f"| {row['cohort']} | {row['method']} | {rank_by} | {screen['actives']} | "
                      + ' | '.join(fmt(screen[k]) for k in SCREEN_FIELDS) + ' |')
    print('\n| Cohort | Method | Chemotype | Quant n | r | ρ |')
    print('| --- | --- | --- | ---: | ---: | ---: |')
    for row in comparisons:
        for name, m in row['per_chemotype'].items():
            print(f"| {row['cohort']} | {row['method']} | {name} | {m['n_quant']} | "
                  f"{fmt(m['pearson_r'])} | {fmt(m['spearman_rho'])} |")
    print('\n| Vina throughput | s/mol | mol/s |')
    print('| --- | ---: | ---: |')
    for mode, rate in summary['throughput'].items():
        print(f"| {mode} | {fmt(rate['seconds_per_mol'])} | {fmt(rate['mol_per_second'])} |")
    print(summary['throughput_notes'])
    print(summary['information_asymmetry'])

    out = summary | {'metrics': metrics, 'per_chemotype': per_chemotype, 'comparisons': comparisons}
    # JSON null denotes an undefined metric, never a substitute for a failed score.
    out_path = args.dir / 'vina_eval.json'
    def undefined_to_null(value):
        if isinstance(value, dict):
            return {k: undefined_to_null(v) for k, v in value.items()}
        if isinstance(value, list):
            return [undefined_to_null(v) for v in value]
        return None if isinstance(value, float) and np.isnan(value) else value

    out_path.write_text(json.dumps(undefined_to_null(out), indent=2, allow_nan=False) + '\n')
    print(f'\nSaved evaluation to {out_path}')


if __name__ == '__main__':
    main()
