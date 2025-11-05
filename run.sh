#!/bin/bash
#SBATCH --job-name=output_dummy
#SBATCH --nodelist=opel
#SBATCH --output=./output/output_dummy.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

## example commands
# python main.py --model gae --output_dir hgcn_gae_noneg_noloss_melanogaster_orthogonal/ --use_nstatement_sampler --no_contrastive --task melanogaster --path melanogaster_data
python main.py --CV_epochs 1 --epochs 1