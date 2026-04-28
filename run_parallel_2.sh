#!/bin/bash
#SBATCH --job-name=wikidata_rargcn_typesampler
#SBATCH --array=2-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-05
#SBATCH --partition=gpu_hi

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rargcn_typesampler"
OUTDIR="${LOG_DIR}/output_wikidata_rargcn_typesampler_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rargcn_typesampler_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main_NEW.py \
  --path "data/wikidata_data" \
  --use_retrieved \
  --print_sampler_stats \
  --task "wikidata" \
  --model "ra_rgcn" \
  --use_tstatement_sampler \
  --batch_size 12460 \
  --output_dir "wikidata_rargcn_typesampler/output_wikidata_rargcn_typesampler_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1