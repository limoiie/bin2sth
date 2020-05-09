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
import torch as t
from gensim import models
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import Loss, RunningAverage
from torch.optim import Adam

from src.database.database import get_database_client, load_nmt_data_end
from src.dataset import get_data_loaders
from src.models.metrics.siamese_loss import SiameseLoss
from src.models.metrics.siamese_metric import SiameseMetric
from src.models.nmt_inspired import NMTInspiredModel
from src.training.build_engine import create_supervised_siamese_trainer, \
    create_supervised_siamese_evaluator
from src.training.pvdm_args import parse_eval_file, ModelArgs
from src.training.training import attach_stages
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


def train(training):
    # TODO: implement query
    work = do_training if training else do_training

    def proxy(cuda, arg_file, **model_args):
        cuda = cuda if cuda >= 0 else None
        client = get_database_client()
        db = client.test_database
        rt = ModelArgs(**model_args)
        work(cuda, arg_file, db, rt)
        client.close()
    return proxy


def do_training(cuda, data_args, db, rt):
    vocab_args, train_corpus, query_corpus = parse_eval_file(data_args)
    data_end = load_nmt_data_end(
        db, vocab_args, train_corpus, query_corpus, max_seq_length)

    # Load a trained w2v model
    w2v = models.Word2Vec.load(embedding_weights)
    # this is used to map word in text into one-hot encoding
    embeddings = _make_embedding(data_end.vocab.tkn2idx, rt.n_emb, w2v)
    embeddings = embeddings.cuda(device=cuda) if cuda else embeddings

    # l_data = t.tensor(data_end.data[0], dtype=t.long, device=cuda)
    # r_data = t.tensor(data_end.data[1], dtype=t.long, device=cuda)
    # data = t.tensor(data_end.data, dtype=t.long, device=cuda)
    data, label = data_end.data, t.tensor(data_end.label, dtype=t.float32)

    model = NMTInspiredModel(data_end.vocab.size, rt.n_emb,
                             embeddings, max_seq_length, n_lstm_hidden)
    optim = Adam(model.parameters(), lr=rt.init_lr)
    loss_fn = SiameseLoss(sim_fn, t.nn.MSELoss())

    ds, ds_val, ds_test = get_data_loaders(data, label, rt.n_batch)

    trainer = create_supervised_siamese_trainer(
        model, optim, loss_fn, device=cuda)
    evaluator = create_supervised_siamese_evaluator(
        model, metrics={
            'auc': SiameseMetric(sim_fn, ROC_AUC()),
            'mse': Loss(loss_fn)
        }, device=cuda
    )

    # attach the evaluator into different stages of trainer so that
    # once the trainer finish something, the evaluator will be called
    # and then its metrics will be computed and output
    attach_stages(trainer, evaluator, ds, ds_val, ds_test)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['batch_loss'])

    trainer.run(ds, max_epochs=rt.epochs)


def sim_fn(o1, o2):
    """
    The distance function used by nmt-inspired to compute the distance
    between two embeddings
    """
    return t.exp(-t.sum(t.abs(o1 - o2), dim=1))


def _make_embedding(vocabulary, n_emb, model):
    """
    Reconstruct embedding matrix by indexing the stored embedding
    weights with the new vocabulary index
    """
    embeddings = t.randn(len(vocabulary), n_emb, dtype=t.float32)
    embeddings[0] = 0  # So that the padding will be ignored

    for word, index in vocabulary.items():
        if word in model.wv:
            embeddings[index] = t.tensor(model.wv[word].copy(), dtype=t.float32)
    return embeddings


if __name__ == '__main__':
    fire.Fire({
        'train': train(True),
        'query': train(False)
    })
