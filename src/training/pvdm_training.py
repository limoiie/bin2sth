import fire
from torch.optim import Adam

from src.database.database import load_cbow_data_end, get_database_client
from src.evaluating.funcs_accuracy import Evaluation
from src.models.pvdm import CBowPVDM, WordEmbedding, FuncEmbedding
from src.training.pvdm_args import ModelArgs
from src.training.pvdm_args import parse_eval_file
from src.training.training import train_one_epoch
from src.utils.logger import get_logger

logger = get_logger('training')


def do_training(cuda, data_args, db, rt):
    vocab_arg, train_corpus_arg, query_corpus_arg = parse_eval_file(data_args)

    train_vocab, train_corpus, train_ds = load_cbow_data_end(
        db, vocab_arg, train_corpus_arg, rt.window, rt.ss)
    query_vocab, query_corpus, query_ds = load_cbow_data_end(
        db, vocab_arg, query_corpus_arg, rt.window, rt.ss)

    embedding = WordEmbedding(train_vocab.size, rt.n_emb, no_hdn=rt.no_hdn)

    tf_embedding = FuncEmbedding(train_corpus.n_docs, rt.n_emb)
    qf_embedding = FuncEmbedding(query_corpus.n_docs, rt.n_emb)

    train_model = CBowPVDM(
        embedding, tf_embedding, train_vocab.size, rt.n_negs,
        train_vocab.word_freq_ratio())
    query_model = CBowPVDM(
        None, qf_embedding, train_vocab.size, rt.n_negs,
        query_vocab.word_freq_ratio())

    train_optim = Adam(train_model.parameters(), lr=rt.init_lr)
    query_optim = Adam(query_model.parameters(), lr=rt.init_lr)

    if cuda >= 0:
        train_model = train_model.cuda(device=cuda)
        query_model = query_model.cuda(device=cuda)

    query_model.embedding = train_model.embedding

    ws = train_vocab.sub_sample_ratio(rt.ss)
    evaluation = Evaluation(
        cuda, train_vocab, ws, train_corpus, query_corpus)

    for epoch in range(1, rt.epochs + 1):
        _train_loss = train_one_epoch(
            epoch, train_model, train_ds, train_optim, rt.n_batch)

        _query_loss = train_one_epoch(
            epoch, query_model, query_ds, query_optim, rt.n_batch)

        evaluation.evaluate1(tf_embedding, qf_embedding)
        evaluation.evaluate2(embedding)

    pass


def train(cuda, data_args, **model_args):
    client = get_database_client()
    db = client.test_database
    rt = ModelArgs(**model_args)
    # train_and_eval(cuda, data_args, db, rt)
    do_training(cuda, data_args, db, rt)
    client.close()


if __name__ == "__main__":
    fire.Fire(train)
