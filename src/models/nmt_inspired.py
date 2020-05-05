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

from keras.layers import Input, Embedding, LSTM, Lambda
from keras.layers.merge import Subtract
from keras.models import Model

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
