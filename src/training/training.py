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


def train_one_epoch_supervised(epoch, model, dataset, valid_dataset,
                               optimizer, loss, n_batch, update_gap=100):
    data_loader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
    progress_bar = ProgressBar(data_loader, f'[Epoch {epoch}]', 100)

    step_loss = 0
    for i, (x, y) in enumerate(progress_bar.bar):
        pred_y = model(*x)
        optimizer.zero_grad()
        loss_v = loss(pred_y, y)
        loss_v.backward()
        optimizer.step()
        step_loss = loss_v.item()
        progress_bar.step(loss=f'{step_loss:.03F}')
    return step_loss
