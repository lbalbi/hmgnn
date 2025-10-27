#!/bin/bash
#SBATCH --job-name=gnn_pyg
#SBATCH --nodelist=opel
#SBATCH --output=./gnn_pyg.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

python gnn_pyg.py -model GCN
