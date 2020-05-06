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

import keras.backend as kb
import torch as t

from keras.layers import Input, Embedding, LSTM, Lambda
from keras.layers.merge import Subtract
from keras.models import Model

from src.models.model import layer_trainable

max_seq_length = 101

n_units_2nd_layer = 50
n_units_1st_layer = 64


class NMTInspiredModel:
    def __init__(self, embeddings, embedding_dim):
        lh_input = Input(shape=(max_seq_length,), dtype='int32')
        rh_input = Input(shape=(max_seq_length,), dtype='int32')

        embedding_layer = Embedding(
            len(embeddings), embedding_dim, weights=[embeddings],
            input_length=max_seq_length, trainable=False)

        # Embedded version of the inputs
        encoded_lh = embedding_layer(lh_input)
        encoded_rh = embedding_layer(rh_input)

        # The 1st hidden layer
        shared_lstm_01 = LSTM(n_units_1st_layer, return_sequences=True)
        # The 2nd hidden layer
        shared_lstm_02 = LSTM(n_units_2nd_layer, activation='relu')

        lh_output = shared_lstm_02(shared_lstm_01(encoded_lh))
        rh_output = shared_lstm_02(shared_lstm_01(encoded_rh))

        def exponent_neg_manhattan_distance(diff):
            """
            Helper function for the similarity estimate of the LSTMs outputs
            """
            return kb.exp(-kb.sum(kb.abs(diff), axis=1, keepdims=True))

        my_distance = Subtract()([lh_output, rh_output])
        my_distance = Lambda(
            exponent_neg_manhattan_distance,
            output_shape=lambda x: (x[0], 1)
        )(my_distance)

        # Pack it all up into a model
        self.smodel = Model([lh_input, rh_input], [my_distance])

    def __call__(self, lhv, rhv):
        return self.smodel.predict(lhv, rhv)

    def load_weights(self, *args, **kwargs):
        return self.smodel.load_weights(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.smodel.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.smodel.fit(*args, **kwargs)

    def compile(self, *args, **kwargs):
        return self.smodel.compile(*args, **kwargs)


class NMTInspiredModel2(t.nn.Module):
    def __init__(self, vocab_size, n_emb, w_embedding, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # batch * seq * emb -> batch * seq * emb
        self.encoder = t.nn.Embedding(
            vocab_size, n_emb, _weight=t.tensor(w_embedding))
        layer_trainable(self.encoder, False)

        # batch * seq * emb -> batch * seq * unites_2nd_layer
        self.shared_lstm = t.nn.LSTM(
            n_emb, n_units_2nd_layer, batch_first=True, num_layers=2)

        # batch * unites_2nd_layer -> batch * 1
        self.decider = t.nn.Linear(n_units_2nd_layer, 1)

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
        return self.decider(dis)
