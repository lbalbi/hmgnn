#!/bin/bash
#SBATCH --job-name=GCN
#SBATCH --nodelist=opel
#SBATCH --output=./GCN_melanogaster.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

python gnn_dgl.py -model GCN