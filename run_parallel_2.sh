#!/bin/bash
#SBATCH --job-name=wikidata_rahgat_noCL_NEW
#SBATCH --array=1-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rahgat_noCL_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_rahgat_noCL_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rahgat_noCL_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_hgat" \
  --no_contrastive \
  --batch_size 3024 \
  --output_dir "wikidata_rahgat_noCL_NEW/output_wikidata_rahgat_noCL_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1