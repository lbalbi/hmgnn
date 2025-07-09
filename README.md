Implementation of Hybrid Modeling GNN approach for training of positive and negative entity representations in tandem for negation-based conflict -aware KGRL.

To run the full pipeline for the Contrastive Heterophilic GNN execute the following in-line command:

```
sbatch run.sh
```

If you wish to run a different architecture for the CHGNN you must pass it as a flag within the run.sh file, e.g. for CHGAT:

```
python main.py --model hgat
```
