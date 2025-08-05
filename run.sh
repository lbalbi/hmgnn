#!/bin/bash
#SBATCH --job-name=hgcn_ALL_noCLoss
#SBATCH --nodelist=opel
#SBATCH --output=./output/output_hgcn_ALL_noCLoss.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

# python main.py --model hgcn --output_dir hgcn_ALL_noCLoss/ --use_nstatement_sampler --task ppi --path ppi_data