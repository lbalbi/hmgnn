#!/bin/bash
#SBATCH --job-name=human_perProt_gcn_protCL
#SBATCH --array=1-1
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-t2
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_perProt_gcn_protCL"
OUTDIR="${LOG_DIR}/output_human_perProt_gcn_protCL_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_perProt_gcn_protCL_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --model "gcn" \
  --protein_splits \
  --output_dir "human_perProt_gcn_protCL/output_human_perProt_gcn_protCL_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1