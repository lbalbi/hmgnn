#!/bin/bash
#SBATCH --job-name=human_hgcn_noNegs_noCL_perProt_NEW
#SBATCH --array=1-3
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_hgcn_noNegs_noCL_perProt_NEW"
OUTDIR="${LOG_DIR}/human_hgcn_noNegs_noCL_perProt_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/human_hgcn_noNegs_noCL_perProt_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --model "hgcn" \
  --protein_splits \
  --use_nstatement_sampler \
  --no_contrastive \
  --batch_size 3024 \
  --output_dir "human_hgcn_noNegs_noCL_perProt_NEW/human_hgcn_noNegs_noCL_perProt_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
  #   --patience 15 \