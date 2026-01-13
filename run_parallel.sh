#!/bin/bash
#SBATCH --job-name=human_sgat_perProt_NEW
#SBATCH --array=11-11
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_sgat_perProt_NEW"
OUTDIR="${LOG_DIR}/human_sgat_perProt_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/human_sgat_perProt_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --no_contrastive \
  --model "sgat" \
  --patience 15 \
  --protein_splits \
  --finaltrain_only \
  --batch_size 3024 \
  --final_lr 0.01 \
  --final_epochs 40 \
  --output_dir "human_sgat_perProt_NEW/human_sgat_perProt_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1