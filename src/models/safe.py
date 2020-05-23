# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This model is slightly modified from the repository at the following link:
https://github.com/facebookresearch/SAFEtorch
which is the official implement of massarelli2019safe.
@inproceedings{massarelli2019safe,
    title={Safe: Self-attentive function embeddings for binary similarity},
    author={Massarelli, Luca and Di Luna, Giuseppe Antonio and Petroni, Fabio
    and Baldoni, Roberto and Querzoni, Leonardo},
    booktitle={International Conference on Detection of Intrusions and Malware,
    and Vulnerability Assessment},
    pages={309--329},
    year={2019},
    organization={Springer}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn.parameter import Parameter

from src.models.builder import ModelBuilder
from src.models.model import layer_trainable
from src.models.modules.word2vec import Word2Vec, Word2VecRecipe
from src.preprocesses.vocab import AsmVocab
from src.training.args.safe_args import SAFEArgs


@ModelBuilder.register_cls
class SAFE(nn.Module):
    vocab: AsmVocab
    w2v: Word2Vec

    def __init__(self, conf, vocab=None, w2v=None, vocab_size=0):
        super(SAFE, self).__init__()

        self.conf: SAFEArgs = conf
        self.instructions_embeddings: Embedding
        self.bidirectional_rnn: torch.nn.GRU
        self.WS1: Parameter
        self.WS2: Parameter
        self.dense_1: torch.nn.Linear
        self.dense_2: torch.nn.Linear

        vocab_size = vocab_size or vocab.size
        weight = None
        if w2v is not None:
            weight = w2v.idx2hdn.weight.clone().detach()
        self.instructions_embeddings = torch.nn.Embedding(
            vocab_size, self.conf.n_emb, _weight=weight)
        layer_trainable(self.instructions_embeddings, w2v is None)

        self.bidirectional_rnn = torch.nn.GRU(
            input_size=self.conf.n_emb,
            hidden_size=self.conf.n_rnn_state,
            num_layers=self.conf.rnn_depth,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

        self.WS1 = Parameter(
            torch.empty(self.conf.atten_depth, 2 * self.conf.n_rnn_state)
        )
        self.WS2 = Parameter(
            torch.empty(self.conf.atten_hops, self.conf.atten_depth)
        )

        self.dense_1 = torch.nn.Linear(
            2 * self.conf.atten_hops * self.conf.n_rnn_state,
            self.conf.n_dense_layer,
            bias=True,
        )
        self.dense_2 = torch.nn.Linear(
            self.conf.n_dense_layer, self.conf.n_emb, bias=True
        )

    def forward(self, instructions):
        # each functions is a list of embeddings id
        # (an id is an index in the embedding matrix)
        # with this we transform it in a list of embeddings vectors.
        instructions_vectors = self.instructions_embeddings(instructions)
        # -> batch_size * seq * n_emb

        # We create the GRU RNN
        H, h_n = self.bidirectional_rnn(instructions_vectors)
        # -> H: batch_size * seq * (2 * rnn_state_size)
        #      <==> batch_size * seq * n_emb

        # We do a tile to account for training batches
        ws1_tiled = self.WS1.unsqueeze(0)  # batch_size * attn_depth * n_emb
        ws2_tiled = self.WS2.unsqueeze(0)  # batch_size * attn_hops * attn_depth

        # we compute the matrix A
        A = torch.softmax(
            ws2_tiled.matmul(
                torch.tanh(
                    ws1_tiled.matmul(
                        H.transpose(1, 2)  # -> batch_size * n_emb * seq
                    )  # -> batch_size * attn_depth * seq
                )  # -> batch_size * attn_depth * seq
            ),  # -> batch_size * attn_hops * seq
            2
        )  # -> batch_size * attn_hops * seq

        # embedding matrix M
        M = A.matmul(H)
        # -> batch_size * attn_hops * n_emb

        # we create the flattened version of M
        flattened_M = M.view(
            -1, 2 * self.conf.atten_hops * self.conf.n_rnn_state
        )
        # -> batch_size * (attn_hops * n_emb), where n_emb == 2 * rnn_state_size

        dense_1_out = F.relu(self.dense_1(flattened_M))
        function_embedding = F.normalize(self.dense_2(dense_1_out), dim=1, p=2)

        return function_embedding, A.mean(dim=0)


def out_transform_for_safe(_x, y, o):
    """
    Unwrap each output of :class SAFE under the siamese architecture so
    that it is acceptable for siamese metric
    """
    (o1, _), (o2, _) = o
    return (o1, o2), y
