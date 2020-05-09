import torch as t
from torch.nn import functional as F


def cosine_mse_loss(o1, o2, y, reduction):
    assert o1.shape == o2.shape
    sim = F.cosine_similarity(o1, o2)
    return F.mse_loss(sim, y, reduction)


class CosineMSELoss(t.nn.Module):
    r"""
    Compute the mse based on the cosine similarity, just as:
      L = mse_loss(cosine_similarity(o1, o2), y)
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, o1, o2, y):
        return cosine_mse_loss(o1, o2, y, self.reduction)
