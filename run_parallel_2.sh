#!/bin/bash
#SBATCH --job-name=human_hgat_protCL_NEW
#SBATCH --array=3-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_perProt_hgat_protCL_NEW"
OUTDIR="${LOG_DIR}/output_human_perProt_hgat_protCL_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_perProt_hgat_protCL_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --model "hgat" \
  --batch_size 3048 \
  --protein_splits \
  --output_dir "human_perProt_hgat_protCL_NEW/output_human_perProt_hgat_protCL_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1