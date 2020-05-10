# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.model import layer_trainable


class SAFE(nn.Module):
    def __init__(self, config, vocab_size, embedding):
        super(SAFE, self).__init__()

        self.conf = config

        # self.instructions_embeddings = torch.nn.Embedding(
        #     self.conf.num_embeddings, self.conf.embedding_size
        # )
        self.instructions_embeddings = torch.nn.Embedding(
            vocab_size, self.conf.embedding_size, 
            _weight=embedding.clone().detach())
        layer_trainable(self.instructions_embeddings, False)

        self.bidirectional_rnn = torch.nn.GRU(
            input_size=self.conf.embedding_size,
            hidden_size=self.conf.rnn_state_size,
            num_layers=self.conf.rnn_depth,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

        self.WS1 = Parameter(
            torch.empty(self.conf.attention_depth, 2 * self.conf.rnn_state_size)
        )
        self.WS2 = Parameter(
            torch.empty(self.conf.attention_hops, self.conf.attention_depth)
        )

        self.dense_1 = torch.nn.Linear(
            2 * self.conf.attention_hops * self.conf.rnn_state_size,
            self.conf.dense_layer_size,
            bias=True,
        )
        self.dense_2 = torch.nn.Linear(
            self.conf.dense_layer_size, self.conf.embedding_size, bias=True
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
            -1, 2 * self.conf.attention_hops * self.conf.rnn_state_size
        )
        # -> batch_size * (attn_hops * n_emb), where n_emb == 2 * rnn_state_size

        dense_1_out = F.relu(self.dense_1(flattened_M))
        function_embedding = F.normalize(self.dense_2(dense_1_out), dim=1, p=2)

        return function_embedding, A.mean(dim=0)

    # def forward(self, instructions, lengths):
    #     # for now assume a batch size of 1
    #     batch_size = 1
    #
    #     # check valid input
    #     if lengths[0] <= 0:
    #         return torch.zeros(batch_size, self.conf.embedding_size)
    #
    #     # each functions is a list of embeddings id
    #     # (an id is an index in the embedding matrix)
    #     # with this we transform it in a list of embeddings vectors.
    #     instructions_vectors = self.instructions_embeddings(instructions)
    #
    #     # consider only valid instructions (defdined by lengths)
    #     valid_instructions = \
    #         torch.split(instructions_vectors, lengths[0], 0)[0].unsqueeze(0)
    #     # -> batch_size * seq * n_emb
    #
    #     # We create the GRU RNN
    #     output, h_n = self.bidirectional_rnn(valid_instructions)
    #     # -> output: batch_size * seq * (2 * rnn_state_size)
    #     #      <==> batch_size * seq * n_emb
    #
    #     pad = torch.zeros(
    #       1, self.conf.max_instructions - lengths[0], self.conf.embedding_size
    #     )
    #     # batch_size * (max_ins - seq) * n_emb
    #
    #     # We create the matrix H
    #     H = torch.cat((output, pad), 1)
    #     # batch_size * max_ins * n_emb
    #
    #     # We do a tile to account for training batches
    #     ws1_tiled = self.WS1.unsqueeze(0)  # batch_size * attn_depth * n_emb
    #   ws2_tiled = self.WS2.unsqueeze(0)  # batch_size * attn_hops * attn_depth
    #
    #     # we compute the matrix A
    #     A = torch.softmax(
    #         ws2_tiled.matmul(
    #             torch.tanh(
    #                 ws1_tiled.matmul(
    #                     H.transpose(1, 2)  # -> batch_size * n_emb * max_ins
    #                 )  # -> batch_size * attn_depth * max_ins
    #             )  # -> batch_size * attn_depth * max_ins
    #         ),  # -> batch_size * attn_hops * max_ins
    #         2
    #     )  # -> batch_size * attn_hops * max_ins
    #
    #     # embedding matrix M
    #     M = A.matmul(H)
    #     # -> batch_size * attn_hops * n_emb
    #
    #     # we create the flattened version of M
    #     flattened_M = M.view(
    #       batch_size, 2 * self.conf.attention_hops * self.conf.rnn_state_size
    #     )
    #   # -> batch_size * (attn_hops * n_emb), where n_emb == 2 * rnn_state_size
    #
    #     dense_1_out = F.relu(self.dense_1(flattened_M))
    #   function_embedding = F.normalize(self.dense_2(dense_1_out), dim=1, p=2)
    #
    #     return function_embedding


class Config:
    def __init__(self):
        self.num_embeddings = 527683
        self.embedding_size = 100  # dimension of the function embedding

        # RNN PARAMETERS, these parameters are only used for RNN model.
        self.rnn_state_size = 50  # dimesion of the rnn state
        self.rnn_depth = 1  # depth of the rnn
        self.max_instructions = 150  # number of instructions

        # ATTENTION PARAMETERS
        self.attention_hops = 10
        self.attention_depth = 250

        # RNN SINGLE PARAMETER
        self.dense_layer_size = 2000
