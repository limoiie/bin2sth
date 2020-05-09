import torch as t


class SiameseLoss(t.nn.Module):
    def __init__(self, sim_fn, loss_fn):
        super().__init__()
        self.sim_fn = sim_fn
        self.loss_fn = loss_fn

    def forward(self, o, y):
        return self.loss_fn(self.sim_fn(o[0], o[1]), y)
