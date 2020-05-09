import torch as t

from ignite.engine import Engine
from ignite.utils import convert_tensor

from src.models.metrics.siamese_metric import SiameseMetric


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


def _prepare_batch2(batch, device, non_blocking):
    # todo: call ignite.utils.convert_tensor
    (x1, x2), y = batch
    return x1, x2, y


def create_supervised_siamese_trainer(
        model, optimizer, loss_fn, device=None, non_blocking=False,
        prepare_batch=_prepare_batch2,
        output_transform=lambda x, l, o, loss: loss.item()):
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x1, x2, y = prepare_batch(
            batch, device=device, non_blocking=non_blocking)
        o1, o2 = model(x1), model(x2)
        loss = loss_fn((o1, o2), y)
        loss.backward()
        optimizer.step()

        return output_transform((x1, x2), y, (o1, o2), loss)

    return Engine(_update)


def create_supervised_siamese_evaluator(
        model, metrics=None,
        device=None, non_blocking=False,
        prepare_batch=_prepare_batch2,
        output_transform=lambda x, y, o: (o, y)):
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with t.no_grad():
            x1, x2, y = prepare_batch(
                batch, device=device, non_blocking=non_blocking)
            o1, o2 = model(x1), model(x2)
            return output_transform((x1, x2), y, (o1, o2))

    engine = Engine(_inference)

    for name, metric in metrics.items():
        # assert isinstance(metric, SiameseMetric), \
        #     'each metric in :param metrics should be an instance of ' \
        #     ':class SiameseMetric'
        # or the metric could be a siamese-style loss wrapped with a Loss
        metric.attach(engine, name)

    return engine
