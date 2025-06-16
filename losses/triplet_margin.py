class TripletLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        """
        :param anchor: Embedding of the anchor sample.
        :param positive: Embedding of the positive sample.
        :param negative: Embedding of the negative sample.
        :return: Computed triplet margin loss.
        """
        pos_dist = (anchor - positive).norm(p=2, dim=-1)
        neg_dist = (anchor - negative).norm(p=2, dim=-1)
        loss = (pos_dist - neg_dist + self.margin).clamp(min=0.0)
        return loss.mean()
    