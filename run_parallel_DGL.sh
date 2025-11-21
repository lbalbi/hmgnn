#!/bin/bash
#SBATCH --job-name=human_DGL_withLC
#SBATCH --array=1-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_DGL_withLC"
OUTDIR="${LOG_DIR}/output_human_DGL_withLC_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_DGL_withLC_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "human_data" \
  --task "human" \
  --output_dir "human_DGL_withLC/output_human_DGL_withLC_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
