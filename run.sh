#!/bin/bash
#SBATCH --job-name=gae_melanogaster
#SBATCH --nodelist=opel
#SBATCH --output=./output/output_gae_melanogaster_orthogonal.txt
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --partition=compute

echo "Job running on node: $(hostname)"

python main.py --model gae --output_dir hgcn_gae_noneg_noloss_melanogaster_orthogonal/ --use_nstatement_sampler --no_contrastive --task melanogaster --path melanogaster_data