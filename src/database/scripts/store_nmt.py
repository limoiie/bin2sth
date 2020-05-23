import torch

from gensim.models import Word2Vec
from torch.nn import Parameter

from src.database.factory import Factory
from src.database.repository import Repository
from src.models.builder import ModelBuilder
from src.preprocesses.arg_parser import ArgsParser
from src.training.args.train_args import RuntimeArgs, TrainArgs, ModelArgs
from src.utils.auto_json import AutoJson

tmp_folder = '../../../src/training/.tmp/nmt_inspired'

embedding_weights = \
    f'{tmp_folder}/100D_MinWordCount0_downSample1e-5_trained100epoch_L.w2v'


def do_that():
    epochs, n_batch, init_lr = 0, 0, 0.0
    data_args = '../../../settings/nmt_eval_args.json'
    model_args = 'store_nmt.json'

    parser = ArgsParser()
    parser.parser(data_args, 'ds')
    parser.parser(model_args)
    m_args = parser.make()

    ds = AutoJson.load(data_args)
    rt = RuntimeArgs(epochs, n_batch, init_lr)
    md = ModelArgs(m_args)

    args = TrainArgs(ds, rt, md)
    args = Repository.find_or_store_training(args)

    args.m.models = dict()
    for mod_name, cls_name in args.m.models_to_load.items():
        Cls = ModelBuilder.model_class(cls_name)
        args.m.models[mod_name] = \
            Factory.load_instance(Cls, args.m.args[mod_name])

    models = args.m.models
    w2v, vocab = models['w2v'], models['vocab']
    trained_w2v = Word2Vec.load(embedding_weights)
    embeddings = _make_embedding(vocab.tkn2idx, w2v.cfg.n_emb, trained_w2v)
    w2v.idx2vec.load_state_dict({
        'weight': embeddings
    })
    w2v.idx2hdn.load_state_dict({
        'weight': embeddings
    })
    Factory.save_instance(getattr(args, '_id'), {
        name: args.m.models[name] for name in args.m.models_to_save
    })


def _make_embedding(vocabulary, n_emb, model):
    """
    Reconstruct embedding matrix by indexing the stored embedding
    weights with the new vocabulary index
    """
    embeddings = torch.randn(len(vocabulary), n_emb, dtype=torch.float32)
    embeddings[0] = 0  # So that the padding will be ignored

    for word, index in vocabulary.items():
        if word in model.wv:
            embeddings[index] = torch.tensor(model.wv[word].copy(),
                                             dtype=torch.float32)
    return embeddings


if __name__ == '__main__':
    do_that()
