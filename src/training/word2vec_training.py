import fire
import torch
from ignite.contrib.handlers import ProgressBar
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.training.build_engine import \
    create_unsupervised_trainer
from src.training.training import train, show_batch_loss_bar
from src.utils.logger import get_logger

logger = get_logger('training')


def do_training(cuda, args):
    models = args.m.models
    base_ds = models['base_ds']
    train_loader = DataLoader(base_ds, args.rt.n_batch, collate_fn=_collect_fn)
    train_model = models['model']

    train_optim = Adam(train_model.parameters(), lr=args.rt.init_lr)
    trainer = create_unsupervised_trainer(
        train_model, train_optim, device=cuda)
    base_ds.attach(trainer)  # re-sub-sample the dataset for each epoch

    show_batch_loss_bar(trainer, ProgressBar())
    trainer.run(train_loader, max_epochs=args.rt.epochs)


def _collect_fn(batch):
    ctr = [w for doc in batch for _, w, _ in doc]
    ctx = [c for doc in batch for _, _, c in doc]
    return (torch.stack(ctr), torch.stack(ctx)), None


if __name__ == "__main__":
    fire.Fire(train(do_training))
