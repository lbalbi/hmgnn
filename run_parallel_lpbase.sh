#!/bin/bash
#SBATCH --job-name=LP_RGCN_NEW
#SBATCH --array=1-2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=gpu_un

set -ex pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/LP_RGCN_NEW"
OUTDIR="${LOG_DIR}/output_LP_RGCN_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_LP_RGCN_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python3 -u main_NEW.py \
  --path "data/wikidata_data" \
  --print_sampler_stats \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 12264 \
  --cv_val_ratio 0.002 \
  --val_lp_eval_every 5 \
  --no_contrastive \
  --output_dir "LP/output_LP_RGCN_NEW_${RUN_TAG}/" \
  > "${LOGFILE}" 2>&1
#   --use_retrieved \
