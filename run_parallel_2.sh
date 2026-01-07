#!/bin/bash
#SBATCH --job-name=wikidata_rahgcn_noNegsCL_NEW
#SBATCH --array=2-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=45:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail
RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rahgcn_noNegsCL_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_rahgcn_noNegsCL_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rahgcn_noNegsCL_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_hgcn" \
  --use_nstatementsampler \
  --batch_size 3024 \
  --finaltrain_only \
  --final_lr 0.005 \
  --final_epochs 38 \
  --output_dir "wikidata_rahgcn_noNegsCL_NEW/output_wikidata_rahgcn_noNegsCL_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1