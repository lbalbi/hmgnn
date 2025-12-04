Updated implementation of Contrastive Relational GNN approach.

####

This implementation introduces a novel mechanism for leveraging verified negative evidence from scientific KGs to train a classification model in a contrastive setting. The contrastive loss pulls apart nodes' positive views from their negative neighbors (statement objects) and closens them to the positive neighbors.
The model is trained for a final objective that results from combining the composed contrastive loss with a task-specific classification loss (BCELoss). The contrastive loss has a learnable trade-off coefficient associated to it that defines its weight in the final loss.


To setup a conda environment with the needed dependencies and download the benchmarks and experimental settings for the C-RGCN paper run the following in-line command:

```
sbatch setup.sh ENV_NAME=myenv
```

To run 10 experiments in parallel for the Contrastive Relational (asuming a SLURM environment) execute the following in-line command:

```
sbatch run_parallel.sh
```

Within the .sh files you can pass specific arguments to change the experimental settings and benchmarks; see the example .sh provided.

This pipeline is extensible to other GNN configurations and datasets by adding their parameters and metadata in a config.json file.
