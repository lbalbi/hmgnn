#!/bin/bash
#SBATCH --job-name=wikidata_test_rargcn_OntoCL
#SBATCH --array=2-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=gpu_hi
#SBATCH --chdir=/home/lbalbi/datasets/hmgnn

# ulimit -n
# ulimit -Hn
ulimit -n 65535

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_test_rargcn_OntoCL"
OUTDIR="${LOG_DIR}/output_wikidata_test_rargcn_OntoCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_test_rargcn_OntoCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u /home/lbalbi/datasets/hmgnn/main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 18640 \
  --use_negativeentity_sampler \
  --parallel_grid \
  --parallel_grid_workers 2 \
  --parallel_grid_loader_workers 0 \
  --output_dir "wikidata_test_rargcn_OntoCL/output_wikidata_test_rargcn_OntoCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
