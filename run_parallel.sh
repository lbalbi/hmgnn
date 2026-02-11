#!/bin/bash
#SBATCH --job-name=wikidata_rargcn_protCL_0402
#SBATCH --array=1-1
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=gpu_hi

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_get_graph_stats"
OUTDIR="${LOG_DIR}/output_wikidata_get_graph_stats_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_get_graph_stats_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main_NEW.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 18460 \
  --finaltrain_only \
  --final_lr 0.005 \
  --final_epochs 15 \
  --output_dir "wikidata_get_graph_stats/output_wikidata_get_graph_stats_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1