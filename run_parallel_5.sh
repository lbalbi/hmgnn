#!/bin/bash
#SBATCH --job-name=wikidata_rargcn_noCL_0402
#SBATCH --array=1-2
#SBATCH --output=slurm_log2.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --nodelist=liseda-01
#SBATCH --partition=gpu_hi

set -euo pipefail

RUN_TAG="run_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="output/wikidata_rargcn_noCL_0402"
OUTDIR="${LOG_DIR}/output_wikidata_rargcn_noCL_0402_${RUN_TAG}"
LOGFILE="${LOG_DIR}/output_wikidata_rargcn_noCL_0402_${RUN_TAG}.txt"
mkdir -p "${OUTDIR}"

python - <<'PY' >> "${LOGFILE}" 2>&1
try:
    import torch
    print("[CUDA] available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[CUDA] device_count:", torch.cuda.device_count())
        print("[CUDA] device_0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("[CUDA] check failed:", e)
PY

python -u main_NEW.py \
  --path "data/wikidata_data" \
  --task "wikidata" \
  --model "ra_rgcn" \
  --batch_size 18460 \
  --no_contrastive \
  --output_dir "wikidata_rargcn_noCL_0402/output_wikidata_rargcn_noCL_0402_${RUN_TAG}/" \
  >> "${LOGFILE}" 2>&1
