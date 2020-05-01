import copy
import fire

from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database import BinArgs
from src.database import load_cbow_data_end, get_database_client, load_json_file
from src.dataset import CBowDataset
from src.models.pvdm import CBowPVDM, WordEmbedding, FuncEmbedding
from src.utils.progress_bar import ProgressBar
from src.utils.logger import get_logger
from src.vocab import compute_word_freq_ratio, compute_sub_sample_ratio

from src.ida.code_elements import Serializable
from src.training.training import train_one_epoch
from src.training.pvdm_args import ModelArgs, parse_data_file
from src.training.pvdm_args import TrainArgs, QueryArgs, parse_eval_file

from src.evaluating.funcs_accuracy import make_evaluation


logger = get_logger('training')


def training(db, rt, cuda, wf, ws, data_end, embedding=None):
    vocab_size, corpus_size = data_end.vocab.size, data_end.corpus.n_docs
    
    fun_embedding = FuncEmbedding(corpus_size, rt.n_emb)

    if embedding:  # fix embedding
        model = CBowPVDM(None, fun_embedding, vocab_size, rt.n_negs, wf)
        optimizer = Adam(model.parameters(), lr=rt.init_lr)
        model.embedding = embedding
    else:
        embedding = WordEmbedding(vocab_size, rt.n_emb, no_hdn=rt.no_hdn)
        model = CBowPVDM(embedding, fun_embedding, vocab_size, rt.n_negs, wf)
        optimizer = Adam(model.parameters(), lr=rt.init_lr)

    loss_epochs = []
    for epoch in range(rt.epochs):
        dataset = CBowDataset(data_end, ws)
        loss = train_one_epoch(epoch, model, dataset, optimizer, rt.n_batch)
        loss_epochs.append(loss)

    return model, loss_epochs


def train_and_eval(cuda, data_args, db, rt):
    """ 
    Training process.
    @param data_args: a file which contains two serialized instances of
    @class ArgsBag, each of which represents a bag of bianries
    """
    vocab_args, train_corpus, query_corpus = parse_eval_file(data_args)

    train_data = load_cbow_data_end(db, vocab_args, train_corpus, rt.window)
    query_data = load_cbow_data_end(db, vocab_args, query_corpus, rt.window)

    wf = compute_word_freq_ratio(train_data.vocab)
    ws = compute_sub_sample_ratio(wf, rt.ss)

    # train_args = TrainArgs('PVDM', rt, vocab_args, train_corpus)
    # query_args = QueryArgs('PVDM', rt, vocab_args, train_corpus, query_corpus)

    train_model, train_loss = training(db, rt, cuda, wf, ws, train_data)
    query_model, query_loss = training(
        db, rt, cuda, wf, ws, query_data, train_model.embedding)

    evaluation = make_evaluation(train_data, query_data, cuda, ws)
    evaluation.evaluate1(train_model.doc_embedding, query_model.doc_embedding)
    evaluation.evaluate2(train_model.embedding)
    # logger.info(f'finish training - {args.serialize()}')


def train(cuda, data_args, **model_args):
    client = get_database_client()
    db = client.test_database
    rt = ModelArgs(**model_args)
    train_and_eval(cuda, data_args, db, rt)
    client.close()


if __name__ == "__main__":
    fire.Fire(train)
