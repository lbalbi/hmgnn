#!/bin/bash
#SBATCH --job-name=wikidata_sgat_noCL_wn_linkloader
#SBATCH --array=2-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_sgat_noCL_wn_linkloader"
OUTDIR="${LOG_DIR}/output_wikidata_sgat_noCL_wn_linkloader_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_sgat_noCL_wn_linkloader_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "sgat" \
  --batch_size 12460 \
  --no_contrastive \
  --output_dir "wikidata_sgat_noCL_wn_linkloader/output_wikidata_sgat_noCL_wn_linkloader_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1