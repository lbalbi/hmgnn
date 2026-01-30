#!/bin/bash
#SBATCH --job-name=wikidata_test_gcn_noCL
#SBATCH --array=1-2
#SBATCH --output=slurm_log2.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --gpus=1
#SBATCH --partition=gpu_hi
#SBATCH --chdir=/home/lbalbi/datasets/hmgnn

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_test_gcn_noCL"
OUTDIR="${LOG_DIR}/output_wikidata_test_gcn_noCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_test_gcn_noCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "gcn" \
  --batch_size 18640 \
  --no_contrastive \
  --parallel_grid \
  --parallel_grid_workers 2 \
  --parallel_grid_loader_workers 0 \
  --output_dir "wikidata_test_gcn_noCL/output_wikidata_test_gcn_noCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1