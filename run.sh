#!/bin/bash
#SBATCH --job-name=hmgnn_hgcn
#SBATCH --nodelist=opel
#SBATCH --output=./output/output_hmgnn_hgcn.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

python main.py --model hgcn