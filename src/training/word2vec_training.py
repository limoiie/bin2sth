import fire
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
from ignite.utils import convert_tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database.database import get_database_client, load_word2vec_data
from src.models.modules.word2vec import Word2Vec, CBow, NegSample
from src.training.build_engine import \
    create_unsupervised_trainer
from src.training.pvdm_args import PVDMArgs
from src.training.train_args import prepare_args
from src.utils.logger import get_logger

logger = get_logger('training')


def train(cuda, data_args, epochs, n_batch, init_lr, **model_args):
    cuda = None if cuda < 0 else cuda
    client = get_database_client()
    db = client.test_database
    args = prepare_args(
        data_args, epochs, n_batch, init_lr, PVDMArgs, **model_args)
    do_training(cuda, db, args)
    client.close()


def do_training(cuda, db, a):
    vocab, corpus, train_ds = load_word2vec_data(
        db, a.ds.vocab, a.ds.corpus, a.m.window, a.m.ss)
    train_loader = DataLoader(train_ds, batch_size=a.rt.n_batch)

    w2v = Word2Vec(vocab.size, a.m.n_emb, no_hdn=a.m.no_hdn)

    sampler = NegSample(vocab.size, a.m.n_negs, vocab.word_freq_ratio())
    train_model = CBow(w2v, sampler)
    train_optim = Adam(train_model.parameters(), lr=a.rt.init_lr)

    trainer = create_unsupervised_trainer(
        train_model, train_optim, device=cuda, prepare_batch=_prepare_batch)

    train_ds.attach(trainer)  # re-sub-sample the dataset for each epoch

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['batch_loss'])

    trainer.run(train_loader, max_epochs=a.rt.epochs)


def _prepare_batch(batch, device=None, non_blocking=False):
    _, word, context = batch
    word = convert_tensor(word, device=device, non_blocking=non_blocking)
    context = convert_tensor(context, device=device, non_blocking=non_blocking)
    return (word, context), None  # None is the non-sense label


if __name__ == "__main__":
    fire.Fire(train)
