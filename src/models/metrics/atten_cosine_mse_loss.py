import torch as t

from src.models.metrics.cosine_mse_loss import CosineMSELoss


class AttenCosineMSELoss(t.nn.Module):
    r"""
    Compute the mix-loss by combining the mse loss on cosine similarity
    and the penalty of attention metric.

    ..math
        L(o1, o2, y, A)_F = \sum_{i=0}^n (cosine_sim(o1, o2) - y)^2 +
          \alpha * \vert A A.T \vert _F

    This is first introduced by Z. Lin, et al 2017 and is adopted by
    :class SAFE to construct the structural loss.

    Reference:
        @article{lin2017structured,
          title={A structured self-attentive sentence embedding},
          author={Lin, Zhouhan and Feng, Minwei and Santos, Cicero Nogueira dos
          and Yu, Mo and Xiang, Bing and Zhou, Bowen and Bengio, Yoshua},
          journal={arXiv preprint arXiv:1703.03130},
          year={2017}
        }
    """

    def __init__(self, alpha=1.0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.loss_fn = CosineMSELoss('sum')

    def forward(self, o1, o2, y):
        """
        NOTE!!: This depends on the model to yield the pred and the attention
          matrix at the same time as the ouput. Consider to reconstruct this
        """
        (o1, A), (o2, _) = o1, o2
        base_loss = self.loss_fn(o1, o2, y)
        I = t.eye(A.shape[0], dtype=A.dtype, device=A.device)
        A_loss = t.norm(t.matmul(A, A.T) - I)
        return base_loss + A_loss * self.alpha
