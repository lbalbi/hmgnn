#!/bin/bash
#SBATCH --job-name=wikidata_test_graphstats
#SBATCH --array=1-1
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=gpu_hi
#SBATCH --chdir=/home/lbalbi/datasets/hmgnn

# ulimit -n
# ulimit -Hn
ulimit -n 65535

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_test_graphstats"
OUTDIR="${LOG_DIR}/output_wikidata_test_graphstats_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_test_graphstats_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u /home/lbalbi/datasets/hmgnn/main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 18640 \
  --contrastive_weight 0.2 \
  --parallel_grid \
  --parallel_grid_workers 2 \
  --parallel_grid_loader_workers 0 \
  --finaltrain_only \
  --final_lr 0.005 \
  --final_epochs 15 \
  --output_dir "wikidata_test_graphstats/output_wikidata_test_graphstats_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
