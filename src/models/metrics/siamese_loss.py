import torch as t


class SiameseLoss(t.nn.Module):
    """
    A loss class for wrapping the output of a model in siamese architecture
    by computing a similarity first. Then the similarity and the label
    will be passed into the :param loss_fn to compute the final loss.
    """

    def __init__(self, sim_fn, loss_fn):
        super().__init__()
        self.sim_fn = sim_fn
        self.loss_fn = loss_fn

    def forward(self, o, y):
        return self.loss_fn(self.sim_fn(o[0], o[1]), y)
