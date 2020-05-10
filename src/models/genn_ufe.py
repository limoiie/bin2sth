import torch as t
import torch.nn.functional as F

from torch.nn import Parameter


class GENNBasedModel(t.nn.Module):
    def __init__(self, n_hidden, n_emb, l, T):
        super().__init__()
        self.T = T
        self.n_hidden = n_hidden
        self.f_state()
        self.layers = [
            t.nn.Linear(n_hidden, n_hidden) for _ in range(l)
        ]
        self.W1 = Parameter(t.empty(1, n_emb, n_hidden))  # 1 for batch dim
        self.W2 = Parameter(t.empty(1, n_hidden, n_hidden))

    def forward(self, vs, adj_m):
        """
        Embedding a function by using the provided block-level feature and
        its adjacence matrix
        :param vs: block-level feature vectors, has a shape of (batch, n_b, n_e)
        :param adj_m: adjacence matrix, has a shape of (batch, n_b, n_b)
        :return:
        """
        us_shape = vs.shape[0], vs.shape[1], self.n_hidden
        us = t.randn(us_shape, dtype=t.float32).to(self.W1.device)
        # -> batch * n_b * n_h

        for _ in self.T:  # T times
            # compute context by accumulating adjs for each block
            adj_accum = adj_m.bmm(us)
            # -> batch * n_blks * n_hidden

            # DNN on context
            for layer in self.layers:
                # NOTE: original paper did not include relu for the last layer
                adj_accum = F.relu(layer(adj_accum.view(-1, self.n_hidden)))
            # -> batch * n_blks * n_hidden

            # update each block state vectors
            us = F.tanh(vs.bmm(self.W1) + adj_accum)
            # -> batch * n_blks * n_hidden

        # final graph embedding
        return self.W2.bmm(us.sum(dim=1))
