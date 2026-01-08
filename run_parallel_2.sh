#!/bin/bash
#SBATCH --job-name=wikidata_srahgcn_RandomNegs_NEW
#SBATCH --array=1-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=45:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_srahgcn_RandomNegs_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_srahgcn_RandomNegs_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_srahgcn_RandomNegs_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "sra_hgcn" \
  --batch_size 3024 \
  --use_rstatement_sampler \
  --output_dir "wikidata_srahgcn_RandomNegs_NEW/output_wikidata_srahgcn_RandomNegs_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1