#!/usr/bin/env bash
#SBATCH --job-name=vina-hsp90
#SBATCH --partition=cn-short
#SBATCH --account=lhlai_g1
#SBATCH --qos=lhlaicns
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --chdir=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate
#SBATCH --output=/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate/results/vina-hsp90-%j.log
set -euo pipefail
export PATH=/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/fap-docking/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
# CPU-only repair uses OpenMM's built-in Reference platform; loading the environment's plugins crashes on cn-short.
export OPENMM_PLUGIN_DIR=/dev/null

OUT_DIR=results/hsp90_vina
python experiments/exp_vina_hsp90.py --dir "$OUT_DIR"
python analysis/eval_vina_hsp90.py --dir "$OUT_DIR" --boltz-dir "$OUT_DIR/inputs"
