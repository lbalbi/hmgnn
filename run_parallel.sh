#!/bin/bash
#SBATCH --job-name=human_noLC
#SBATCH --array=1-10
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3


echo "Job running on node: $(hostname)"
set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_noLC"
OUTDIR="${LOG_DIR}/output_human_noLC_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_noLC_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py --output_dir "human_noLC/output_human_noLC_${RUN_TAG}/" >> "${LOGFILE}" 2>&1