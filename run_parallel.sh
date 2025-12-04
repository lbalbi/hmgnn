#!/bin/bash
#SBATCH --job-name=human_test_shgcn_noCL
#SBATCH --array=4-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_test_shgcn_noCL"
OUTDIR="${LOG_DIR}/output_human_test_shgcn_noCL${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_test_shgcn_noCL${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --model "shgcn" \
  --no_contrastive \
  --output_dir "human_test_shgcn_noCL/output_human_test_shgcn_noCL${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1