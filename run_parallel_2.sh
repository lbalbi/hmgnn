#!/bin/bash
#SBATCH --job-name=human_test_shgcn_noCL_perPPI
#SBATCH --array=1-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_test_shgcn_noCL_perPPI"
OUTDIR="${LOG_DIR}/output_human_test_shgcn_noCL_perPPI_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_test_shgcn_noCL_perPPI_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --no_contrastive \
  --model shgcn \
  --output_dir "human_test_shgcn_noCL_perPPI/output_human_test_shgcn_noCL_perPPI_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
