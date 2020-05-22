import fire
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import TopKCategoricalAccuracy, RunningAverage
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.database.dao import to_filter
from src.database.factory import Factory
from src.database.repository import Repository
from src.models.pvdm import CBowPVDM, doc_eval_flatten_transform, \
    doc_eval_transform
from src.preprocesses.arg_parser import ArgsParser
from src.preprocesses.builder import ModelBuilder
from src.preprocesses.vocab_builder import VocabRecipe
from src.preprocesses.corpus_builder import CorpusRecipe
from src.training.args.train_args import RuntimeArgs, TrainArgs, ModelArgs
from src.training.build_engine import create_unsupervised_trainer, \
    create_unsupervised_training_evaluator
from src.training.training import attach_unsupervised_evaluator
from src.utils.auto_json import AutoJson
from src.utils.logger import get_logger

logger = get_logger('training')


def prepare_args(data_args, model_args, epochs, n_batch, init_lr):
    parser = ArgsParser()
    parser.parser(data_args, 'ds')
    parser.parser(model_args)
    m_args = parser.make()

    ds = AutoJson.load(data_args)
    rt = RuntimeArgs(epochs, n_batch, init_lr)
    md = ModelArgs(m_args)
    args = TrainArgs(ds, rt, md)
    return Repository.find_or_store_training(args)


def load_models(args):
    args.m.models = dict()
    for mod_name, cls_name in args.m.models_to_load.items():
        Cls = ModelBuilder.model_class(cls_name)
        args.m.models[mod_name] = Factory.load_model(Cls, args.m.args[mod_name])


def save_models(args):
    for mod_name in args.m.models_to_save:
        print(f'save model {mod_name} of {args.m.models[mod_name]}')


def train(fn_training):
    def do(cuda, data_args, model_args, epochs, n_batch, init_lr):
        cuda = None if cuda < 0 else cuda
        args = prepare_args(data_args, model_args, epochs, n_batch, init_lr)
        load_models(args)
        fn_training(cuda, args)
        save_models(args)
    return do


def do_training(cuda, args):
    base_ds, find_ds = args.m.models['base_ds'], args.m.models['find_ds']

    # train_loader = DataLoader(
    #     base_ds, batch_size=args.rt.n_batch, collate_fn=_collect_fn)
    # query_loader = DataLoader(
    #     find_ds, batch_size=args.rt.n_batch, collate_fn=_collect_fn)
    #
    # base_model, find_model = models['base_model'], models['find_model']
    # find_model.w2v = None
    #
    # train_optim = Adam(base_model.parameters(), lr=args.rt.init_lr)
    # query_optim = Adam(find_model.parameters(), lr=args.rt.init_lr)
    # find_model.w2v = base_model.w2v
    #
    # trainer = create_unsupervised_trainer(
    #     base_model, train_optim, device=cuda)
    # evaluator = create_unsupervised_training_evaluator(
    #     base_model, find_model, query_optim,
    #     metrics={
    #         'auc': ROC_AUC(doc_eval_flatten_transform),
    #         'topk-acc': TopKCategoricalAccuracy(
    #             k=1, output_transform=doc_eval_transform)
    #     }, device=cuda)
    #
    # base_ds.attach(trainer)  # re-sub-sample the dataset for each epoch
    # find_ds.attach(evaluator)  # re-subsample the dataset for each epoch
    #
    # attach_unsupervised_evaluator(trainer, evaluator, query_loader)
    #
    # RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    # pbar = ProgressBar()
    # pbar.attach(trainer, ['batch_loss'])
    #
    # trainer.run(train_loader, max_epochs=args.rt.epochs)


def _collect_fn(batch):
    func, word, ctx, labels = [], [], [], []
    for doc in batch:
        labels.append(doc[0][0])
        for f, w, c in doc:
            func.append(f)
            word.append(w)
            ctx.append(c)
    return (torch.stack(func), torch.stack(word),
            torch.stack(ctx)), torch.stack(labels)


if __name__ == '__main__':
    fire.Fire(train(do_training))
