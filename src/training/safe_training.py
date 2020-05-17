import fire
import torch as t
import torch.nn.functional as F
from gensim import models
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import RunningAverage, Loss
from torch.optim import Adam

from src.database.database import load_nmt_data_end
from src.dataset.dataset import get_data_loaders
from src.models.metrics.atten_cosine_mse_loss import AttenPenaltyLoss
from src.models.metrics.siamese_loss import SiameseLoss
from src.models.metrics.siamese_metric import SiameseMetric
from src.models.safe import SAFE
from src.training.build_engine import create_supervised_siamese_trainer, \
    create_supervised_siamese_evaluator
from src.training.training import attach_stages, train
from src.utils.logger import get_logger

logger = get_logger('training')

tmp_folder = 'src/training/.tmp/nmt_inspired'

embedding_weights = \
    f'{tmp_folder}/100D_MinWordCount0_downSample1e-5_trained100epoch_L.w2v'


def do_training(cuda, db, a):
    # todo: load safe dataset
    data_end = load_nmt_data_end(
        db, a.ds.vocab, a.ds.base_corpus, a.ds.find_corpus, a.m.max_seq_len)

    # Load a trained w2v model
    w2v = models.Word2Vec.load(embedding_weights)
    # this is used to map word in text into one-hot encoding
    embeddings = _make_embedding(data_end.vocab.tkn2idx, a.m.n_emb, w2v)
    embeddings = embeddings.cuda(device=cuda) if cuda else embeddings

    data, label = data_end.data, t.tensor(data_end.label, dtype=t.float32)

    model = SAFE(a.m, data_end.vocab.size, embeddings)

    train_optim = Adam(model.parameters(), lr=a.rt.init_lr)
    core_loss_fn = SiameseLoss(F.cosine_similarity, t.nn.MSELoss())
    loss_fn = AttenPenaltyLoss(core_loss_fn)

    ds, ds_val, ds_test = get_data_loaders(data, label, a.rt.n_batch)

    trainer = create_supervised_siamese_trainer(
        model, train_optim, loss_fn, device=cuda)
    evaluator = create_supervised_siamese_evaluator(
        model, metrics={
            'auc': SiameseMetric(F.cosine_similarity, ROC_AUC()),
            'mse': Loss(core_loss_fn)
        }, device=cuda,
        output_transform=out_transform_for_safe
    )

    attach_stages(trainer, evaluator, ds, ds_val, ds_test)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['batch_loss'])

    trainer.run(ds, max_epochs=a.rt.epochs)


def out_transform_for_safe(_x, y, o):
    """
    Unwrap each output of :class SAFE under the siamese architecture so
    that it is acceptable for siamese metric
    """
    (o1, _), (o2, _) = o
    return (o1, o2), y


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


if __name__ == "__main__":
    fire.Fire(train(do_training))
