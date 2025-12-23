#!/bin/bash
#SBATCH --job-name=human_perprot_noNegsCL
#SBATCH --array=1-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_perprot_noNegsCL"
OUTDIR="${LOG_DIR}/output_human_perprot_noNegsCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_perprot_noNegsCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --use_nstatement_sampler \
  --protein_splits \
  --output_dir "human_perprot_noNegsCL/output_human_perprot_noNegsCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1