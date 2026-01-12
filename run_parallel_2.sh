#!/bin/bash
#SBATCH --job-name=wikidata_gcn_NEW
#SBATCH --array=3-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=45:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_gcn_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_gcn_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_gcn_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "gcn" \
  --batch_size 3024 \
  --no_contrastive \
  --test_only \
  --output_dir "wikidata_gcn_NEW/output_wikidata_gcn_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
