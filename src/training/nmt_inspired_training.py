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

from src.database.database import load_nmt_data_end
from src.dataset.dataset import get_data_loaders
from src.models.metrics.siamese_loss import SiameseLoss
from src.models.metrics.siamese_metric import SiameseMetric
from src.models.nmt_inspired import NMTInspiredModel
from src.training.build_engine import create_supervised_siamese_trainer, \
    create_supervised_siamese_evaluator
from src.training.training import attach_stages, train
from src.utils.logger import get_logger

logger = get_logger('training')

tmp_folder = 'src/training/.tmp/nmt_inspired'

# saved_weights = f'{tmp_folder}/siamese_model_100DW2V_2HL_50HU_O2.ourown.hdf5'
embedding_weights = \
    f'{tmp_folder}/100D_MinWordCount0_downSample1e-5_trained100epoch_L.w2v'


def do_training(cuda, db, a):
    data_end = load_nmt_data_end(
        db, a.ds.vocab, a.ds.base_corpus, a.ds.find_corpus, a.m.max_seq_len)

    # Load a trained w2v model
    w2v = models.Word2Vec.load(embedding_weights)
    # this is used to map word in text into one-hot encoding
    embeddings = _make_embedding(data_end.vocab.tkn2idx, a.m.n_emb, w2v)
    embeddings = embeddings.cuda(device=cuda) if cuda else embeddings

    data, label = data_end.data, t.tensor(data_end.label, dtype=t.float32)

    model = NMTInspiredModel(data_end.vocab.size, a.m.n_emb,
                             embeddings, a.m.max_seq_len, a.m.n_lstm_hidden)
    optim = Adam(model.parameters(), lr=a.rt.init_lr)
    loss_fn = SiameseLoss(sim_fn, t.nn.MSELoss())

    ds, ds_val, ds_test = get_data_loaders(data, label, a.rt.n_batch)

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

    trainer.run(ds, max_epochs=a.rt.epochs)


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
    fire.Fire(train(do_training))
