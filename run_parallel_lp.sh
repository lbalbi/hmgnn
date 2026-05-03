#!/bin/bash
#SBATCH --job-name=LP_NegKRGCN_NEW_
#SBATCH --array=1-2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=gpu_hi

set -ex pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/LP_NegKRGCN_NEW"
OUTDIR="${LOG_DIR}/output_LP_NegKRGCN_NEW_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_LP_NegKRGCN_NEW_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python3 -u main_NEW.py \
  --path "data/wikidata_data" \
  --print_sampler_stats \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 17564 \
  --cv_val_ratio 0.003 \
  --val_lp_eval_every 5 \
  --max_contrastive_anchors 3072 \
  --use_retrieved \
  --output_dir "LP/output_LP_NegKRGCN_NEW_${RUN_TAG}/" \
  > "${LOGFILE}" 2>&1
  #  --no_contrastive \
