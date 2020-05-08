from ignite.engine import Engine
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking=False):
    x, l = batch
    return convert_tensor(x, device=device, non_blocking=non_blocking), \
        convert_tensor(l, device=device, non_blocking=non_blocking)


def create_unsupervised_trainer(
        model, optimizer, device=None, non_blocking=False,
        prepare_batch=_prepare_batch,
        output_transform=lambda x, l, loss: loss.item()):
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, l = prepare_batch(batch, device=device, non_blocking=non_blocking)
        loss = model(x)
        loss.backward()
        optimizer.step()

        return output_transform(x, l, loss)

    return Engine(_update)


def create_unsupervised_training_evaluator(
        base_model, model, optimizer, metrics=None,
        device=None, non_blocking=False,
        prepare_batch=_prepare_batch,
        output_transform=lambda x, l, loss, b, e: (l, b, e)):
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(_engine, batch):
        base_model.eval()
        base_interested = base_model.interested_out()

        model.train()
        optimizer.zero_grad()
        x, l = prepare_batch(batch, device=device, non_blocking=non_blocking)
        loss = model(x)
        loss.backward()
        optimizer.step()

        interested = model.interested_out()
        return output_transform(x, l, loss, base_interested, interested)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
