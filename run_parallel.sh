#!/bin/bash
#SBATCH --job-name=wikidata_rahgcn_protCL
#SBATCH --array=4-4
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_protCL_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_protCL_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_protCL_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_hgcn" \
  --batch_size 3024 \
  --finaltrain_only \
  --final_lr 0.05 \
  --final_epochs 50 \
  --output_dir "wikidata_protCL_NEW/output_wikidata_protCL_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1