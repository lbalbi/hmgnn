#!/bin/bash
#SBATCH --job-name=human_shgcn_randomNegs_perProt
#SBATCH --array=6-10
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_shgcn_randomNegs_perProt"
OUTDIR="${LOG_DIR}/output_human_shgcn_randomNegs_perProt_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_shgcn_randomNegs_perProt_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --use_protein_contrastive \
  --use_rstatement_sampler \
  --protein_splits \
  --model shgcn \
  --output_dir "human_shgcn_randomNegs_perProt/output_human_shgcn_randomNegs_perProt_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1