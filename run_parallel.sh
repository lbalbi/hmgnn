#!/bin/bash
#SBATCH --job-name=melanogaster_withLC
#SBATCH --array=1-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

echo "Job running on node: $(hostname)"
set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/melanogaster_withLC"
OUTDIR="${LOG_DIR}/output_melanogaster_withLC_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_melanogaster_withLC_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py --path "melanogaster_data" --task "melanogaster" --output_dir "melanogaster_withLC/output_melanogaster_withLC_${RUN_TAG}/" >> "${LOGFILE}" 2>&1