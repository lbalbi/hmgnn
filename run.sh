#!/bin/bash
#SBATCH --job-name=hmgnn_hgcn_randomneg_CLoss
#SBATCH --nodelist=opel
#SBATCH --output=./output/output_hmgnn_hgcn_randomneg_CLoss.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

python main.py --model hgcn --output_dir hgcn_randomneg_CLoss/ --use_rstatement_sampler
# python main.py --model hgcn --output_dir hgcn_GDA_all_CLoss/  --gda_data --task gda