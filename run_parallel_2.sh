#!/bin/bash
#SBATCH --job-name=wikidata_rargcn_protCL_1802_test
#SBATCH --array=1-1
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=gpu_hi

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rargcn_protCL_1802_test"
OUTDIR="${LOG_DIR}/output_wikidata_rargcn_protCL_1802_test_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rargcn_protCL_1802_test_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main_NEW.py \
  --path "data/wikidata_data" \
  --use_retrieved \
  --print_sampler_stats \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 12460 \
  --output_dir "wikidata_rargcn_protCL_1802_test/output_wikidata_rargcn_protCL_1802_test_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1