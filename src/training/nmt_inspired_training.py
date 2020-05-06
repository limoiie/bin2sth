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
import ignite
import torch as t
from gensim import models
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import create_supervised_trainer, \
    create_supervised_evaluator, Events
from ignite.metrics import Loss, RunningAverage
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database.database import get_database_client, load_nmt_data_end
from src.dataset import NMTInspiredDataset
from src.models.nmt_inspired import NMTInspiredModel
from src.training.pvdm_args import parse_eval_file, ModelArgs
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
    embeddings = make_embedding(data_end.vocab.tkn2idx, rt.n_emb, w2v)
    embeddings = embeddings.cuda(device=cuda)

    data = t.tensor(data_end.data, dtype=t.long, device=cuda)
    label = t.tensor(data_end.label, dtype=t.float32, device=cuda)

    model = NMTInspiredModel(data_end.vocab.size, rt.n_emb,
                             embeddings, max_seq_length, n_lstm_hidden)
    optim = Adam(model.parameters(), lr=rt.init_lr)
    loss = t.nn.MSELoss()

    ds, ds_val, ds_test = get_data_loaders(data, label, rt.n_batch)

    trainer = create_supervised_trainer(model, optim, loss, device=cuda)
    evaluator = create_supervised_evaluator(
        model, metrics={
            'auc': ROC_AUC(),
            'mse': Loss(loss)
        }, device=cuda
    )

    attach(trainer, evaluator, ds, ds_val, ds_test)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['batch_loss'])

    trainer.run(ds, max_epochs=rt.epochs)

    # for epoch in range(1, rt.epochs + 1):
    #     _epoch_loos = train_one_epoch_supervised(
    #         epoch, model, ds, ds_val, optim, loss, rt.n_batch)

    # pred = model(X_test[:, 0], X_test[:, 1])
    # evaluate_auc(Y_test, pred.detach())


def attach(trainer, evaluator, ds_train, ds_val, ds_test):
    def eval_on(ds, tag, event):
        @trainer.on(event)
        def log_eval_results(engine):
            evaluator.run(ds)
            metrics = evaluator.state.metrics
            print("{} Results f Epoch: {}  "
                  "Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(tag, engine.state.epoch,
                          metrics['auc'], metrics['mse']))

    eval_on(ds_train, 'Training', Events.EPOCH_COMPLETED)
    eval_on(ds_val, 'Validation', Events.EPOCH_COMPLETED)
    eval_on(ds_test, 'Testing', Events.COMPLETED)
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(engine):
    #     evaluator.run(ds_train)
    #     metrics = evaluator.state.metrics
    #     print("Training Results f Epoch: {}  "
    #           "Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #           .format(engine.state.epoch, metrics['auc'], metrics['mse']))
    #
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     evaluator.run(ds_val)
    #     metrics = evaluator.state.metrics
    #     print("Validation Results - Epoch: {}  "
    #           "Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #           .format(engine.state.epoch, metrics['auc'], metrics['mse']))


def make_embedding(vocabulary, n_emb, model):
    # This will be the embedding matrix
    embeddings = t.randn(len(vocabulary), n_emb, dtype=t.float32)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix, please refer to the meeting slides for
    # more detailed explanation
    for word, index in vocabulary.items():
        if word in model.wv:
            embeddings[index] = t.tensor(model.wv[word].copy(), dtype=t.float32)
    return embeddings


def get_data_loaders(data, label, n_batch):
    X_train, X_valid, X_test = \
        data[:90000], data[90000:100821], data[:100821]
    Y_train, Y_valid, Y_test = \
        label[:90000], label[90000:100821], label[:100821]

    def create(x, y, batch_size, shuffle=True):
        return DataLoader(NMTInspiredDataset(x, y),
                          batch_size=batch_size, shuffle=shuffle)

    train_ds = create(X_train, Y_train, n_batch)
    valid_ds = create(X_valid, Y_valid, n_batch)
    test_ds = create(X_test, Y_test, n_batch)

    return train_ds, valid_ds, test_ds


if __name__ == '__main__':
    fire.Fire({
        'train': train(True),
        'query': train(False)
    })
