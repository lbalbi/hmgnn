import torch

class ContrastiveLoss():
    """
    Euclidian distance-based contrastive loss function.
    This loss function is used to train models to differentiate between similar and dissimilar pairs of samples.
    """

    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, positive, negative):
        """
        Args: positive (torch.Tensor) - Positive sample embeddings
              negative (torch.Tensor) - Negative sample embeddings.
        Returns: torch.Tensor - Contrastive loss score
        """
        pos_dist = torch.sum((positive - negative) ** 2, dim=1) # Euclidean distance between positives and negatives
        loss = torch.mean(pos_dist) + self.margin * torch.mean(torch.clamp(self.margin - pos_dist, min=0)) # mean of the distances
        # between pos + neg with margin value
        return loss