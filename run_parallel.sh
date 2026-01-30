#!/bin/bash
#SBATCH --job-name=wikidata_test_sgnn_noCL
#SBATCH --array=1-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_test_sgnn_noCL"
OUTDIR="${LOG_DIR}/output_wikidata_test_sgnn_noCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_test_sgnn_noCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "sgat" \
  --batch_size 10640 \
  --no_contrastive \
  --parallel_grid \
  --parallel_grid_workers 2 \
  --parallel_grid_loader_workers 0 \
  --output_dir "wikidata_test_sgnn_noCL/output_wikidata_test_sgnn_noCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1