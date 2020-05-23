import fire
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import TopKCategoricalAccuracy
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.pvdm import doc_eval_flatten_transform, doc_eval_transform
from src.training.build_engine import create_unsupervised_trainer, \
    create_unsupervised_training_evaluator
from src.training.training import attach_unsupervised_evaluator, \
    show_batch_loss_bar, train
from src.utils.logger import get_logger

logger = get_logger('training')


def do_training(cuda, args):
    models = args.m.models
    base_ds, find_ds = models['base_ds'], models['find_ds']
    train_loader = DataLoader(base_ds, args.rt.n_batch, collate_fn=_collect_fn)
    query_loader = DataLoader(find_ds, args.rt.n_batch, collate_fn=_collect_fn)

    base_model, find_model = models['base_model'], models['find_model']
    find_model.w2v = None

    train_optim = Adam(base_model.parameters(), lr=args.rt.init_lr)
    query_optim = Adam(find_model.parameters(), lr=args.rt.init_lr)
    find_model.w2v = base_model.w2v

    trainer = create_unsupervised_trainer(
        base_model, train_optim, device=cuda)
    evaluator = create_unsupervised_training_evaluator(
        base_model, find_model, query_optim,
        metrics={
            'auc': ROC_AUC(doc_eval_flatten_transform),
            'topk-acc': TopKCategoricalAccuracy(
                k=1, output_transform=doc_eval_transform)
        }, device=cuda)

    base_ds.attach(trainer)  # re-sub-sample the dataset for each epoch
    find_ds.attach(evaluator)  # re-subsample the dataset for each epoch

    attach_unsupervised_evaluator(trainer, evaluator, query_loader)
    show_batch_loss_bar(trainer, ProgressBar())

    trainer.run(train_loader, max_epochs=args.rt.epochs)


def _collect_fn(batch):
    func, word, ctx, labels = [], [], [], []
    for doc in batch:
        labels.append(doc[0][0])
        for f, w, c in doc:
            func.append(f), word.append(w), ctx.append(c)
    return (torch.stack(func), torch.stack(word),
            torch.stack(ctx)), torch.stack(labels)


if __name__ == '__main__':
    fire.Fire(train(do_training))
