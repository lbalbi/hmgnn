#!/bin/bash
#SBATCH --job-name=wikidata_rahgcn_randomNegs_NEW
#SBATCH --array=1-1
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=45:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=tier3

set -euo pipefail
RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rahgcn_randomNegs_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_rahgcn_randomNegs_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rahgcn_randomNegs_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_hgcn" \
  --use_rstatement_sampler \
  --output_dir "wikidata_rahgcn_randomNegs_NEW/output_wikidata_rahgcn_randomNegs_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1