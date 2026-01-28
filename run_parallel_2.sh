#!/bin/bash
#SBATCH --job-name=wikidata_gcn_noNegs_RandomNegs_NS_wn_linkloader
#SBATCH --array=1-2
#SBATCH --output=slurm_log.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-03
#SBATCH --partition=tier3

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_gcn_noNegs_RandomNegs_NS_wn_linkloader"
OUTDIR="${LOG_DIR}/output_wikidata_gcn_noNegs_RandomNegs_NS_wn_linkloader_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_gcn_noNegs_RandomNegs_NS_wn_linkloader_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python -u main_NEW.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "gcn" \
  --batch_size 12460 \
  --use_nstatementsampler \
  --use_rstatement_sampler \
  --output_dir "wikidata_gcn_noNegs_RandomNegs_NS_wn_linkloader/output_wikidata_gcn_noNegs_RandomNegs_NS_wn_linkloader_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1