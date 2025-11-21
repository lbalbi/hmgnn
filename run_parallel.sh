#!/bin/bash
#SBATCH --job-name=cerevisae_withLC
#SBATCH --array=1-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/cerevisae_withLC"
OUTDIR="${LOG_DIR}/output_cerevisae_withLC_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_cerevisae_withLC_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "cerevisae_data_withLC" \
  --task "cerevisae" \
  --output_dir "cerevisae_withLC/output_cerevisae_withLC_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1