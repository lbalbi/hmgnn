#!/bin/bash
#SBATCH --job-name=human_sgnn_perProt_NEW
#SBATCH --array=2-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_sgnn_perProt_NEW"
OUTDIR="${LOG_DIR}/output_human_sgnn_perProt_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_sgnn_perProt_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --model "sgnn" \
  --protein_splits \
  --no_contrastive \
  --output_dir "human_sgnn_perProt_NEW/output_human_sgnn_perProt_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1