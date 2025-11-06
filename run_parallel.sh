#!/bin/bash
#SBATCH --job-name=human_noLC
#SBATCH --array=1-10
#SBATCH --output=slurm-%x_%A_%a.txt
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

echo "LOG_DIR = ${LOG_DIR}"
echo "OUTDIR  = ${OUTDIR}"
echo "LOGFILE = ${LOGFILE}"

python -u main.py --output_dir "human_noLC/output_human_noLC_${RUN_TAG}/" >> "${LOGFILE}" 2>&1