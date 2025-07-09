Original implementation of Contrastive Heterophilic GNN approach for negation-based conflict -aware KGRL.

####

This implementation introduces a novel mechanism for sampling negatives from negative KG statements that builds dual entity representations to train a model in a contrastive setting. The contrastive loss pulls apart nodes from their negative neighbors (statement objects) and closens them to the positive neighbors.
The model is trained on a final loss that results from combining the contrastive loss with a task-specific classification loss (BCELoss). The contrastive loss has an alpha coefficient associated to it that defines the contrastive weight in the final loss, with a default value of 0.1.

To run the full pipeline for the Contrastive Heterophilic GNN execute the following in-line command:

```
sbatch run.sh
```

If you wish to run a different architecture for the CHGNN you must pass it as a flag within the run.sh file, e.g. for CHGAT:

```
python main.py --model hgat
```

This pipeline is extensible to other GNNs and datasets by adding their parameters and metadata to the config.json file.