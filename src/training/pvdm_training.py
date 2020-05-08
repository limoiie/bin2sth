import fire
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Events
from ignite.metrics import RunningAverage, TopKCategoricalAccuracy
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database.database import get_database_client, load_cbow_data
from src.training.create_unsupervised_engine import \
    create_unsupervised_trainer, create_unsupervised_training_evaluator
from src.evaluating.funcs_accuracy import Evaluation
from src.models.pvdm import CBowPVDM, WordEmbedding, FuncEmbedding
from src.training.pvdm_args import ModelArgs
from src.training.pvdm_args import parse_eval_file
from src.training.training import train_one_epoch
from src.utils.logger import get_logger

logger = get_logger('training')


def train(cuda, data_args, **model_args):
    cuda = None if cuda < 0 else cuda
    client = get_database_client()
    db = client.test_database
    rt = ModelArgs(**model_args)
    # train_and_eval(cuda, data_args, db, rt)
    do_training2(cuda, data_args, db, rt)
    client.close()


def do_training(cuda, data_args, db, rt):
    vocab_arg, train_corpus_arg, query_corpus_arg = parse_eval_file(data_args)

    vocab, train_corpus, query_corpus, train_ds, query_ds = \
        load_cbow_data(db, vocab_arg, train_corpus_arg, query_corpus_arg,
                       rt.window, rt.ss)

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

    if cuda >= 0:
        train_model = train_model.cuda(device=cuda)
        query_model = query_model.cuda(device=cuda)

    query_model.embedding = train_model.embedding

    ws = vocab.sub_sample_ratio(rt.ss)
    evaluation = Evaluation(
        cuda, vocab, ws, train_corpus, query_corpus)

    for epoch in range(1, rt.epochs + 1):
        _train_loss = train_one_epoch(
            epoch, train_model, train_ds, train_optim, rt.n_batch)

        _query_loss = train_one_epoch(
            epoch, query_model, query_ds, query_optim, rt.n_batch)

        evaluation.evaluate1(tf_embedding, qf_embedding)
        evaluation.evaluate2(embedding)

    pass


def do_training2(cuda, data_args, db, rt):
    vocab_arg, train_corpus_arg, query_corpus_arg = parse_eval_file(data_args)

    vocab, train_corpus, query_corpus, train_ds, query_ds = \
        load_cbow_data(db, vocab_arg, train_corpus_arg, query_corpus_arg,
                       rt.window, rt.ss)
    train_ds = DataLoader(train_ds, batch_size=rt.n_batch,
                          collate_fn=_collect_fn)
    query_ds = DataLoader(query_ds, batch_size=rt.n_batch,
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
            'auc': ROC_AUC(_doc_eval_flatten_transform),
            'topk-acc': TopKCategoricalAccuracy(
                k=1, output_transform=_doc_eval_transform)
        }, device=cuda)

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_per_epoch(engine):
        evaluator.run(query_ds)
        metrics = evaluator.state.metrics
        print("Evaluation Results f Epoch: {}  "
              "AUC area: {:.2f}, Top-1 accuracy: {:.2f}"
              .format(engine.state.epoch,
                      metrics['auc'], metrics['topk-acc']))

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['batch_loss'])

    trainer.run(train_ds, max_epochs=rt.epochs)


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


def _doc_eval_transform(output):
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = base_doc_embedding.idx2vec.weight
    pred_embedding_w = doc_embedding(doc_ids)

    y_pred = torch.matmul(pred_embedding_w, true_embedding_w.T)
    y = doc_ids

    return y_pred, y.T


def _doc_eval_flatten_transform(output):
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = base_doc_embedding.idx2vec.weight
    pred_embedding_w = doc_embedding(doc_ids)

    y_pred = torch.matmul(true_embedding_w, pred_embedding_w.T)
    y = torch.zeros_like(y_pred, dtype=torch.int32)
    y[doc_ids] = torch.eye(len(doc_ids), dtype=torch.int32)

    return y_pred.reshape(-1), y.reshape(-1)


def normalize(m):
    return m / m.norm(dim=1, keepdim=True)


if __name__ == "__main__":
    fire.Fire(train)
