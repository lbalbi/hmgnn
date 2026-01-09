#!/bin/bash
#SBATCH --job-name=huri_perPPI_hgcn_noNegsCL
#SBATCH --array=1-1
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-t2
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/huri_perPPI_hgcn_noNegsCL"
OUTDIR="${LOG_DIR}/output_huri_perPPI_hgcn_noNegsCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_huri_perPPI_hgcn_noNegsCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/huri_data_withLC" \
  --task "huri" \
  --model "hgcn" \
  --use_nstatement_sampler \
  --output_dir "huri_perPPI_hgcn_noNegsCL/output_huri_perPPI_hgcn_noNegsCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1