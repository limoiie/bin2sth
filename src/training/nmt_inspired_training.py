# noinspection DuplicatedCode
"""
This model is modified from the repository at the next link:
https://github.com/nmt4binaries/nmt4binaries.github.io/tree/master/download
which is the official implement of zuo2019neural.
@inproceedings{zuo2019neural,
    title={Neural Machine Translation Inspired Binary Code Similarity
      Comparison beyond Function Pairs},
    author={Zuo, Fei and Li, Xiaopeng and Young, Patrick and Luo,Lannan and
      Zeng,Qiang and Zhang, Zhexin},
    booktitle={Proceedings of the 2019 Network and Distributed Systems
      Security Symposium (NDSS)},
    year={2019} }
"""

import fire
import numpy as np
import torch as t
from gensim import models
from torch.optim import Adam

from src.database.database import get_database_client, load_nmt_data_end
from src.dataset import NMTInspiredDataset
from src.evaluating.evaluate import evaluate_auc
from src.models.nmt_inspired import NMTInspiredModel
from src.training.pvdm_args import parse_eval_file, ModelArgs
from src.training.training import train_one_epoch_supervised
from src.utils.logger import get_logger

logger = get_logger('training')

tmp_folder = 'src/training/.tmp/nmt_inspired'

# TRAIN_CSV = f'{tmp_folder}/train_set_O2.csv'
# TEST_CSV = f'{tmp_folder}/test_set_O2.csv'

# saved_weights = f'{tmp_folder}/siamese_model_100DW2V_2HL_50HU_O2.ourown.hdf5'
embedding_weights = \
    f'{tmp_folder}/100D_MinWordCount0_downSample1e-5_trained100epoch_L.w2v'

# Inpute size
max_seq_length = 101
n_lstm_hidden = 64

questions_cols = ['x86_bb', 'arm_bb']
label_col = 'eq'


def train(training):
    # TODO: implement query
    work = do_training if training else do_training

    def proxy(arg_file, **model_args):
        client = get_database_client()
        db = client.test_database
        rt = ModelArgs(**model_args)
        work(arg_file, db, rt)
        client.close()
    return proxy


def do_training(data_args, db, rt):
    vocab_args, train_corpus, query_corpus = parse_eval_file(data_args)
    data_end = load_nmt_data_end(
        db, vocab_args, train_corpus, query_corpus, 101)

    # Load a trained w2v model
    w2v = models.Word2Vec.load(embedding_weights)
    # this is used to map word in text into one-hot encoding
    embeddings = make_embedding(data_end.vocab.tkn2idx, w2v)

    logger.info(f'Shape of embedding: {embeddings.shape}')

    data = t.tensor(data_end.data, dtype=t.long)
    label = t.tensor(data_end.label, dtype=t.float32)

    X_train, X_valid, X_test = \
        data[:90000], data[90000:100821], data[:100821]
    Y_train, Y_valid, Y_test = \
        label[:90000], label[90000:100821], label[:100821]

    logger.info(f'Shape of test dataset: {X_train.shape}')

    model = NMTInspiredModel(data_end.vocab.size, rt.n_emb,
                             embeddings, max_seq_length, n_lstm_hidden)
    optim = Adam(model.parameters(), lr=rt.init_lr)
    loss = t.nn.MSELoss()

    for epoch in range(1, rt.epochs + 1):
        ds = NMTInspiredDataset(X_train, Y_train)
        ds_val = NMTInspiredDataset(X_valid, Y_valid)
        _epoch_loos = train_one_epoch_supervised(
            epoch, model, ds, ds_val, optim, loss, rt.n_batch)

    pred = model(X_test[:, 0], X_test[:, 1])
    evaluate_auc(Y_test, pred.detach())


def make_embedding(vocabulary, model):
    n_emb = model.wv.shape[1]
    # This will be the embedding matrix
    embeddings = np.random.randn(len(vocabulary), n_emb)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix, please refer to the meeting slides for
    # more detailed explanation
    for word, index in vocabulary.items():
        if word in model.wv:
            embeddings[index] = model.wv[word]
    return embeddings.astype(np.float32)


if __name__ == '__main__':
    fire.Fire({
        'train': train(True),
        'query': train(False)
    })
