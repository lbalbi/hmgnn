#!/bin/bash
#SBATCH --job-name=human_perPPI_gcn
#SBATCH --array=3-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-t2
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_perPPI_gcn"
OUTDIR="${LOG_DIR}/output_human_perPPI_gcn_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_perPPI_gcn_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --no_contrastive \
  --model "gcn" \
  --patience 25 \
  --output_dir "human_perPPI_gcn/output_human_perPPI_gcn_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1