#!/bin/bash
#SBATCH --job-name=human_noLC
#SBATCH --output=./output/human_noLC/output_human_noLC.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00

echo "Job running on node: $(hostname)"

## example commands
# python main.py --model gae --output_dir hgcn_gae_noneg_noloss_melanogaster_orthogonal/ --use_nstatement_sampler --no_contrastive --task melanogaster --path melanogaster_data
python main.py --output_dir /human_noLC