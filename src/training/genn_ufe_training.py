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

from src.database.database import load_genn_ufe_data, load_model, dump_model
from src.models.modules.word2vec import Word2Vec
from src.training.args.genn_ufe_args import GENNArgs
from src.training.training import train
from src.utils.logger import get_logger

logger = get_logger('training')

tmp_folder = 'src/training/.tmp/nmt_inspired'

# saved_weights = f'{tmp_folder}/siamese_model_100DW2V_2HL_50HU_O2.ourown.hdf5'
embedding_weights = \
    f'{tmp_folder}/100D_MinWordCount0_downSample1e-5_trained100epoch_L.w2v'


def do_training(cuda, db, a):
    vocab, corpus1, corpus2, train_ds = load_genn_ufe_data(
        db, a.ds.vocab, a.ds.base_corpus, a.ds.find_corpus)

    from IPython import embed; embed()
    # todo:
    #  1. load dataset;
    #    [v] build lua O0 O3 dataset
    #    [v] create supervised dataset
    #  2. load embedding
    #    [v] training an embedding
    #    [-] store and load
    #  3. load blk2vec model
    #    [-] training a blk2vec model
    #    [-] store and load
    #  4. extract block feature
    #  5. create model and evaluate

    # load model
    w2v = load_word2vec(db, a.m, vocab.size)

    # Load a trained w2v model
    # w2v = models.Word2Vec.load(embedding_weights)
    # # this is used to map word in text into one-hot encoding
    # embeddings = _make_embedding(data_end.vocab.tkn2idx, rt.n_emb, w2v)
    # embeddings = embeddings.cuda(device=cuda) if cuda else embeddings
    #
    # data, label = data_end.data, t.tensor(data_end.label, dtype=t.float32)
    #
    # model = NMTInspiredModel(data_end.vocab.size, rt.n_emb,
    #                          embeddings, max_seq_length, n_lstm_hidden)
    # optim = Adam(model.parameters(), lr=rt.init_lr)
    # loss_fn = SiameseLoss(sim_fn, t.nn.MSELoss())
    #
    # ds, ds_val, ds_test = get_data_loaders(data, label, rt.n_batch)
    #
    # trainer = create_supervised_siamese_trainer(
    #     model, optim, loss_fn, device=cuda)
    # evaluator = create_supervised_siamese_evaluator(
    #     model, metrics={
    #         'auc': SiameseMetric(sim_fn, ROC_AUC()),
    #         'mse': Loss(loss_fn)
    #     }, device=cuda
    # )
    #
    # # attach the evaluator into different stages of trainer so that
    # # once the trainer finish something, the evaluator will be called
    # # and then its metrics will be computed and output
    # attach_stages(trainer, evaluator, ds, ds_val, ds_test)
    #
    # RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    # pbar = ProgressBar()
    # pbar.attach(trainer, ['batch_loss'])
    #
    # trainer.run(ds, max_epochs=rt.epochs)

    # dump model
    dump_model(db, a.__dict__['_id'], {
        'word2vec': w2v.state_dict()
    })


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


def load_word2vec(db, m, vocab_size):
    """load the model if there is a dependency"""
    w2v = Word2Vec(vocab_size, m.n_emb, no_hdn=m.no_hdn)
    if not m.requires or 'word2vec' not in m.requires:
        return w2v
    req = m.requires['word2vec']
    state = load_model(db, req)
    if state:
        w2v.load_state_dict(state)
        return w2v
    raise ModuleNotFoundError(f'No such pre-trained checkpoint is found:'
                              f'training_id: {req["training_id"]},'
                              f'module: {req["module"]}')


if __name__ == '__main__':
    fire.Fire(train(do_training))
