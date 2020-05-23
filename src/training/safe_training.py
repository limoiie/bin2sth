import fire
import torch as t
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import Loss
from torch.optim import Adam

from src.models.metrics.atten_cosine_mse_loss import AttenPenaltyLoss
from src.models.metrics.siamese_loss import SiameseLoss
from src.models.metrics.siamese_metric import SiameseMetric
from src.models.safe import SAFE, out_transform_for_safe
from src.preprocesses.dataset.dataset_spliter import DatasetSpliter
from src.training.args.train_args import TrainArgs
from src.training.build_engine import create_supervised_siamese_trainer, \
    create_supervised_siamese_evaluator
from src.training.training import attach_stages, train, show_batch_loss_bar
from src.utils.logger import get_logger

logger = get_logger('training')


def do_training(cuda, a: TrainArgs):
    model: SAFE = a.m.models['safe']
    dataset = a.m.models['dataset']

    train_optim = Adam(model.parameters(), lr=a.rt.init_lr)
    core_loss_fn = SiameseLoss(F.cosine_similarity, t.nn.MSELoss())
    loss_fn = AttenPenaltyLoss(core_loss_fn)

    ds, ds_val, ds_test = DatasetSpliter(dataset).split(a.rt.n_batch)

    trainer = create_supervised_siamese_trainer(
        model, train_optim, loss_fn, device=cuda)
    evaluator = create_supervised_siamese_evaluator(
        model, metrics={
            'auc': SiameseMetric(F.cosine_similarity, ROC_AUC()),
            'mse': Loss(core_loss_fn)
        }, device=cuda,
        output_transform=out_transform_for_safe
    )

    attach_stages(trainer, evaluator, ds, ds_val, ds_test)

    show_batch_loss_bar(trainer, ProgressBar())
    trainer.run(ds, max_epochs=a.rt.epochs)


if __name__ == "__main__":
    fire.Fire(train(do_training))
