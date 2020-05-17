import fire
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import RunningAverage, TopKCategoricalAccuracy
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database.database import get_database_client, load_pvdm_data
from src.models.modules.word2vec import Word2Vec
from src.models.pvdm import CBowPVDM, FuncEmbedding, \
    doc_eval_transform, doc_eval_flatten_transform
from src.training.build_engine import \
    create_unsupervised_trainer, create_unsupervised_training_evaluator
from src.training.train_args import prepare_args
from src.training.training import attach_unsupervised_evaluator
from src.utils.logger import get_logger

logger = get_logger('training')


def train(cuda, data_args, model_args, epochs, n_batch, init_lr):
    cuda = None if cuda < 0 else cuda
    client = get_database_client()
    db = client.test_database
    args = prepare_args(data_args, model_args, epochs, n_batch, init_lr)
    do_training(cuda, db, args)
    client.close()


def do_training(cuda, db, a):
    vocab, train_corpus, query_corpus, train_ds, query_ds = \
        load_pvdm_data(db, a.ds.vocab, a.ds.base_corpus, a.ds.find_corpus,
                       a.m.window, a.m.ss)
    train_loader = DataLoader(
        train_ds, batch_size=a.rt.n_batch, collate_fn=_collect_fn)
    query_loader = DataLoader(
        query_ds, batch_size=a.rt.n_batch, collate_fn=_collect_fn)

    embedding = Word2Vec(vocab.size, a.m.n_emb, no_hdn=a.m.no_hdn)

    tf_embedding = FuncEmbedding(train_corpus.n_docs, a.m.n_emb)
    qf_embedding = FuncEmbedding(query_corpus.n_docs, a.m.n_emb)

    train_model = CBowPVDM(
        embedding, tf_embedding, vocab.size, a.m.n_negs,
        vocab.word_freq_ratio())
    query_model = CBowPVDM(
        None, qf_embedding, vocab.size, a.m.n_negs,
        vocab.word_freq_ratio())

    train_optim = Adam(train_model.parameters(), lr=a.rt.init_lr)
    query_optim = Adam(query_model.parameters(), lr=a.rt.init_lr)

    query_model.w2v = train_model.w2v

    trainer = create_unsupervised_trainer(
        train_model, train_optim, device=cuda)
    evaluator = create_unsupervised_training_evaluator(
        train_model, query_model, query_optim,
        metrics={
            'auc': ROC_AUC(doc_eval_flatten_transform),
            'topk-acc': TopKCategoricalAccuracy(
                k=1, output_transform=doc_eval_transform)
        }, device=cuda)

    train_ds.attach(trainer)  # re-sub-sample the dataset for each epoch
    query_ds.attach(evaluator)  # re-subsample the dataset for each epoch

    attach_unsupervised_evaluator(trainer, evaluator, query_loader)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['batch_loss'])

    trainer.run(train_loader, max_epochs=a.rt.epochs)


def _collect_fn(batch):
    func, word, ctx, labels = [], [], [], []
    for doc in batch:
        labels.append(doc[0][0])
        for f, w, c in doc:
            func.append(f)
            word.append(w)
            ctx.append(c)
    return (torch.stack(func), torch.stack(word),
            torch.stack(ctx)), torch.stack(labels)


if __name__ == "__main__":
    fire.Fire(train)
