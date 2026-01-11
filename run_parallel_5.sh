#!/bin/bash
#SBATCH --job-name=human_rahgcn_noNegs_RandomNegs_NEW
#SBATCH --array=6-6
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rahgcn_noNegs_RandomNegs_NEW"
OUTDIR="${LOG_DIR}/output_wikidata_rahgcn_noNegs_RandomNegs_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rahgcn_noNegs_RandomNegs_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_hgcn" \
  --batch_size 3048 \
  --use_rstatement_sampler \
  --use_nstatementsampler \
  --test_only \
  --output_dir "wikidata_rahgcn_noNegs_RandomNegs_NEW/output_wikidata_rahgcn_noNegs_RandomNegs_NEW_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1