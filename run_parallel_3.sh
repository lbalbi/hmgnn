#!/bin/bash
#SBATCH --job-name=human_test_shgcn_protCL_perprot_noNegsCL_v3
#SBATCH --array=1-5
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/human_test_shgcn_protCL_perprot_noNegsCL_v3"
OUTDIR="${LOG_DIR}/output_human_test_shgcn_protCL_perprot_noNegsCL_v3_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_human_test_shgcn_protCL_perprot_noNegsCL_v3_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/human_data_withLC" \
  --task "human" \
  --use_protein_contrastive \
  --use_nstatement_sampler \
  --protein_splits \
  --model shgcn \
  --output_dir "human_test_shgcn_protCL_perprot_noNegsCL_v3/output_human_test_shgcn_protCL_perprot_noNegsCL_v3_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
