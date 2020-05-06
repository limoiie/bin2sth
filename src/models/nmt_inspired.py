"""
This model is slightly modified from the repository at the next link:
https://github.com/nmt4binaries/nmt4binaries.github.io/tree/master/download
which is the official implement of zuo2019neural.
@inproceedings{zuo2019neural,
    title={Neural Machine Translation Inspired Binary Code Similarity
      Comparison beyond Function Pairs},
    author={Zuo, Fei and Li, Xiaopeng and Young, Patrick and Luo, Lannan and
      Zeng, Qiang and Zhang, Zhexin},
    booktitle={Proceedings of the 2019 Network and Distributed Systems
      Security Symposium (NDSS)},
    year={2019} }
"""

import torch as t

from src.models.model import layer_trainable


class NMTInspiredModel(t.nn.Module):
    def __init__(self, vocab_size, n_emb, w_embedding,
                 max_seq_len, n_lstm_hidden):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_lstm_hidden = n_lstm_hidden

        # batch * seq * emb -> batch * seq * emb
        self.encoder = t.nn.Embedding(
            vocab_size, n_emb, _weight=t.tensor(w_embedding))
        layer_trainable(self.encoder, False)

        # batch * seq * emb -> batch * seq * unites_2nd_layer
        self.shared_lstm = t.nn.LSTM(
            n_emb, n_lstm_hidden, batch_first=True, num_layers=2)

        # batch * unites_2nd_layer -> batch * 1
        self.decider = t.nn.Linear(n_lstm_hidden, 1)

    def forward(self, l_input, r_input):
        # -> batch * seq * emb
        l_encoded = self.encoder(l_input)
        r_encoded = self.encoder(r_input)
        # -> batch * seq * emb
        l_output, _ = self.shared_lstm(l_encoded)
        r_output, _ = self.shared_lstm(r_encoded)
        # -> batch * seq * units_2nd_layer
        dis = t.exp(-t.sum(t.abs(l_output - r_output), dim=1, keepdim=False))
        # -> batch * units_2nd_layer
        return self.decider(dis).squeeze()
