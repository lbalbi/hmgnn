#!/usr/bin/env bash
#SBATCH --job-name=setup_env
#SBATCH --nodelist=opel
#SBATCH --output=./output/setup.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --partition=compute


ENV_DIR="${1:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
PY_BIN="${PY_BIN:-python3.11}"

echo "Creating virtual environment at: $ENV_DIR (using $PY_BIN)"
"$PY_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: $REQ_FILE not found in $(pwd)."
  exit 1
fi

pip install -r "$REQ_FILE"
