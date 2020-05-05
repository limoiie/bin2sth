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

import itertools
from IPython import embed

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import models
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, auc

from src.database.database import get_database_client, load_nmt_data_end, \
    load_inst_vocab, fix_vocab
from src.models.nmt_inspired import NMTInspiredModel
from src.training.pvdm_args import parse_eval_file
from src.utils.logger import get_logger
from src.vocab import AsmVocab

logger = get_logger('training')

tmp_folder = '.tmp/nmt_inspired'

TRAIN_CSV = f'{tmp_folder}/train_set_O2.csv'
TEST_CSV = f'{tmp_folder}/test_set_O2.csv'

saved_weights = f'{tmp_folder}/siamese_model_100DW2V_2HL_50HU_O2.hdf5'
embedding_weights = \
    f'{tmp_folder}/100D_MinWordCount0_downSample1e-5_trained100epoch_L.w2v'

# Inpute size
w2v_dim = 100
embedding_dim = w2v_dim

max_seq_length = 101

questions_cols = ['x86_bb', 'arm_bb']
label_col = 'eq'


def train(arg_file, **model_args):
    client = get_database_client()
    db = client.test_database
    # rt = ModelArgs(**model_args)
    rt = None
    do_training(arg_file, db, rt)
    client.close()


def do_training(data_args, db, rt):
    vocab_args, train_corpus, query_corpus = parse_eval_file(data_args)
    data_end = load_nmt_data_end(
        db, vocab_args, train_corpus, query_corpus, 101)

    logger.info(f'Size of vocabulary: {data_end.vocab.size}')

    # Load a trained w2v model
    model = models.Word2Vec.load(embedding_weights)
    # this is used to map word in text into one-hot encoding
    embeddings = make_embedding(data_end.vocab.tkn2idx, model)

    logger.info(f'Shape of embedding: {embeddings.shape}')

    X_test = np.array(data_end.data[100821:])
    Y_test = np.array(data_end.label[100821:])

    logger.info(f'Shape of test dataset: {X_test.shape}')

    smodel = NMTInspiredModel(embeddings, embedding_dim)
    smodel.load_weights(saved_weights)

    pred = smodel.predict([X_test[:, 0], X_test[:, 1]])

    evaluate_auc(Y_test, pred)


def make_embedding(vocabulary, model):
    # This will be the embedding matrix
    embeddings = np.random.randn(len(vocabulary), embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix, please refer to the meeting slides for
    # more detailed explanation
    for word, index in vocabulary.items():
        if word in model.wv:
            embeddings[index] = model.wv[word]
    return embeddings


def evaluate_auc(Y_test, pred):
    fpr, tpr, _ = roc_curve(Y_test, pred, pos_label=1)
    roc_auc = auc(fpr, tpr) * 100

    plt.figure()
    plt.plot(fpr, tpr, color='red', linewidth=1.2,
             label='Siamese Model (AUC = %0.2f%%)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='silver', linestyle=':', linewidth=1.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    fire.Fire(train)
