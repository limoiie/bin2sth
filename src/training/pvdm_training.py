import fire
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import RunningAverage, TopKCategoricalAccuracy
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database.database import get_database_client, load_cbow_data
from src.evaluating.evaluate import doc_eval_transform, \
    doc_eval_flatten_transform
from src.models.pvdm import CBowPVDM, WordEmbedding, FuncEmbedding
from src.training.create_unsupervised_engine import \
    create_unsupervised_trainer, create_unsupervised_training_evaluator
from src.training.pvdm_args import ModelArgs
from src.training.pvdm_args import parse_eval_file
from src.training.training import attach_unsupervised_evaluator
from src.utils.logger import get_logger

logger = get_logger('training')


def train(cuda, data_args, **model_args):
    cuda = None if cuda < 0 else cuda
    client = get_database_client()
    db = client.test_database
    rt = ModelArgs(**model_args)
    do_training(cuda, data_args, db, rt)
    client.close()


def do_training(cuda, data_args, db, rt):
    vocab_arg, train_corpus_arg, query_corpus_arg = parse_eval_file(data_args)

    vocab, train_corpus, query_corpus, train_ds, query_ds = \
        load_cbow_data(db, vocab_arg, train_corpus_arg, query_corpus_arg,
                       rt.window, rt.ss)
    train_loader = DataLoader(train_ds, batch_size=rt.n_batch,
                              collate_fn=_collect_fn)
    query_loader = DataLoader(query_ds, batch_size=rt.n_batch,
                              collate_fn=_collect_fn)

    embedding = WordEmbedding(vocab.size, rt.n_emb, no_hdn=rt.no_hdn)

    tf_embedding = FuncEmbedding(train_corpus.n_docs, rt.n_emb)
    qf_embedding = FuncEmbedding(query_corpus.n_docs, rt.n_emb)

    train_model = CBowPVDM(
        embedding, tf_embedding, vocab.size, rt.n_negs,
        vocab.word_freq_ratio())
    query_model = CBowPVDM(
        None, qf_embedding, vocab.size, rt.n_negs,
        vocab.word_freq_ratio())

    train_optim = Adam(train_model.parameters(), lr=rt.init_lr)
    query_optim = Adam(query_model.parameters(), lr=rt.init_lr)

    query_model.embedding = train_model.embedding
    # ws = vocab.sub_sample_ratio(rt.ss)

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

    trainer.run(train_loader, max_epochs=rt.epochs)


def _collect_fn(batch):
    func, word, ctx, labels = [], [], [], []
    for doc in batch:
        labels.append(doc[0][0])
        for f, w, c in doc:
            func.append(f)
            word.append(w)
            ctx.append(c)
    return (torch.tensor(func), torch.tensor(word),
            torch.tensor(ctx)), torch.tensor(labels)


if __name__ == "__main__":
    fire.Fire(train)
