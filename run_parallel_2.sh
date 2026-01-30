#!/bin/bash
#SBATCH --job-name=wikidata_test_rargcn_noCL
#SBATCH --array=1-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=gpu_un

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_test_rargcn_noCL"
OUTDIR="${LOG_DIR}/output_wikidata_test_rargcn_noCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_test_rargcn_noCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 18640 \
  --no_contrastive \
  --parallel_grid \
  --parallel_grid_workers 2 \
  --parallel_grid_loader_workers 0 \
  --output_dir "wikidata_test_rargcn_noCL/output_wikidata_test_rargcn_noCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1