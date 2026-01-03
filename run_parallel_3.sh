#!/bin/bash
#SBATCH --job-name=human_perPPI_randomNegs_hgcn
#SBATCH --array=5-10
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_perPPI_randomNegs_hgcn"
OUTDIR="${LOG_DIR}/output_human_perPPI_randomNegs_hgcn_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_perPPI_randomNegs_hgcn_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --use_rstatement_sampler \
  --patience 30 \
  --output_dir "human_perPPI_randomNegs_hgcn/output_human_perPPI_randomNegs_hgcn_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1