#!/bin/bash
#SBATCH --job-name=wikidata_sgat_NEW
#SBATCH --array=5-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=45:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_sgat_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_sgat_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_sgat_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "sgat" \
  --batch_size 3024 \
  --no_contrastive \
  --finaltrain_only \
  --final_lr 0.005 \
  --final_epochs 35 \
  --output_dir "wikidata_sgat_NEW/output_wikidata_sgat_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
