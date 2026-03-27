#!/bin/bash
#SBATCH --job-name=NBFNet
#SBATCH --array=2-2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=gpu_hi

# 11201_1
set -ex pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/NBFNet"
OUTDIR="${LOG_DIR}/output_NBFNet_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_NBFNet_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python3 -u main_NEW.py \
  --path "data/wikidata_data" \
  --use_retrieved \
  --print_sampler_stats \
  --task "wikidata" \
  --model "nbfnet" \
  --batch_size 4648 \
  --no_contrastive \
  --output_dir "NBFNet/output_NBFNet_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1