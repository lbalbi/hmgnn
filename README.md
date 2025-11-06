Original implementation of Contrastive Relational GNN approach for the paper submission "Integrating Negative Scientific Knowledge into Relational Graph Learning".

####

This implementation introduces a novel mechanism for leveraging verified negative evidence from scientific KGs to train a classification model in a contrastive setting. The contrastive loss pulls apart nodes from their negative neighbors (statement objects) and closens them to the positive neighbors.
The model is trained for a final objective that results from combining the contrastive loss with a task-specific classification loss (BCELoss). The contrastive loss has a trade-off coefficient associated to it that defines its weight in the final loss.

The main branch contains the original DGL implementation.
The hmgnn_pyg branch contains the novel PyG implementation, that is from now on the only one receiving updates.

To run the full pipeline for the Contrastive Relational GNN execute the following in-line command:

```
sbatch run.sh
```

This pipeline is extensible to other GNNs and datasets by adding their parameters and metadata to the config.json file.
