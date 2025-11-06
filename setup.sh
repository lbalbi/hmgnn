#!/usr/bin/env bash
#SBATCH --job-name=setup_env
#SBATCH --output=./setup.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00


set -euo pipefail

ENV_NAME="${ENV_NAME:-py311-cu121}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0+cu121}"
PYG_VERSION="${PYG_VERSION:-2.6.1}"
RDFLIB_VERSION="${RDFLIB_VERSION:-7.1.3}"
TORCH_SCATTER_VERSION="${TORCH_SCATTER_VERSION:-2.1.2+pt24cu121}"
TORCH_SPARSE_VERSION="${TORCH_SPARSE_VERSION:-0.6.18+pt24cu121}"
TORCH_CLUSTER_VERSION="${TORCH_CLUSTER_VERSION:-1.6.3+pt24cu121}"
TORCH_SPLINE_CONV_VERSION="${TORCH_SPLINE_CONV_VERSION:-1.2.2+pt24cu121}"

die() { echo "Error: $*" >&2; exit 1; }
need_cmd() {command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found. Please install it and retry."
}
need_cmd curl
need_cmd python
need_cmd conda

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
else conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==${TORCH_VERSION}"
python -m pip install \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  "torch-geometric==${PYG_VERSION}" \
  "torch-scatter==${TORCH_SCATTER_VERSION}" \
  "torch-sparse==${TORCH_SPARSE_VERSION}" \
  "torch-cluster==${TORCH_CLUSTER_VERSION}" \
  "torch-spline-conv==${TORCH_SPLINE_CONV_VERSION}"
python -m pip install "rdflib==${RDFLIB_VERSION}"

python - <<'PY'
import sys
import torch
import rdflib
import torch_geometric
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("rdflib:", rdflib.__version__)
print("torch_geometric:", torch_geometric.__version__)
PY

echo ">>> Downloading data archive"
DATA_URL="${DATA_URL:-https://zenodo.org/records/17542163/files/data.zip}"
DATA_ZIP="${DATA_ZIP:-data.zip}"
curl -L "$DATA_URL" -o "$DATA_ZIP"
unzip -o -q "$DATA_ZIP"
rm -f "$DATA_ZIP"
echo "To activate environment: conda activate $ENV_NAME"