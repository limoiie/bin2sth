from ignite.engine import Events
from torch.utils.data import DataLoader

from src.utils.progress_bar import ProgressBar


def train_one_epoch(epoch, model, dataset, optimizer, n_batch):
    data_loader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
    progress_bar = ProgressBar(data_loader, f'[Epoch {epoch}]', 100)

    # TODO: use average loss instead
    step_loss = 0
    for i, batch in enumerate(progress_bar.bar):
        # print(f'data entry: ({fun}, {word}, {ctx})')
        loss = model(*batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss = loss.item()
        progress_bar.step(loss=step_loss)
    return step_loss


def attach_stages(trainer, evaluator, ds_train, ds_val, ds_test):
    def eval_on(ds, tag, event):
        @trainer.on(event)
        def log_eval_results(engine):
            evaluator.run(ds)
            metrics = evaluator.state.metrics
            print("{} Results f Epoch: {}  "
                  "Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(tag, engine.state.epoch,
                          metrics['auc'], metrics['mse']))

    eval_on(ds_train, 'Training', Events.EPOCH_COMPLETED)
    eval_on(ds_val, 'Validation', Events.EPOCH_COMPLETED)
    eval_on(ds_test, 'Testing', Events.COMPLETED)
