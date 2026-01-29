#!/bin/bash
#SBATCH --job-name=human_cgcn_randomNegs_noNegs_NEW
#SBATCH --array=1-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_cgcn_randomNegs_noNegs_NEW"
OUTDIR="${LOG_DIR}/output_human_cgcn_randomNegs_noNegs_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/human_cgcn_randomNegs_noNegs_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --model "gcn" \
  --batch_size 3048 \
  --protein_splits \
  --use_rstatement_sampler \
  --use_nstatement_sampler \
  --output_dir "human_cgcn_randomNegs_noNegs_NEW/output_human_cgcn_randomNegs_noNegs_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1