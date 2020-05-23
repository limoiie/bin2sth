from ignite.engine import Events
from ignite.metrics import RunningAverage

from src.database.factory import Factory
from src.database.repository import Repository
from src.models.builder import ModelBuilder
from src.preprocesses.arg_parser import ArgsParser
from src.training.args.train_args import RuntimeArgs, ModelArgs, TrainArgs
from src.utils.auto_json import AutoJson


def train(fn_training):
    def do(cuda, data_args, model_args, epochs, n_batch, init_lr):
        cuda = None if cuda < 0 else cuda
        args = _prepare_args(
            data_args, model_args, epochs, n_batch, init_lr)
        _load_models(args)
        fn_training(cuda, args)
        _save_models(args)
    return do


def _prepare_args(data_args, model_args, epochs, n_batch, init_lr):
    parser = ArgsParser()
    parser.parser(data_args, 'ds')
    parser.parser(model_args)
    m_args = parser.make()

    ds = AutoJson.load(data_args)
    rt = RuntimeArgs(epochs, n_batch, init_lr)
    md = ModelArgs(m_args)
    args = TrainArgs(ds, rt, md)
    return Repository.find_or_store_training(args)


def _load_models(args):
    args.m.models = dict()
    for mod_name, cls_name in args.m.models_to_load.items():
        Cls = ModelBuilder.model_class(cls_name)
        args.m.models[mod_name] = \
            Factory.load_instance(Cls, args.m.args[mod_name])


def _save_models(args):
    Factory.save_instance(getattr(args, '_id'), {
        name: args.m.models[name] for name in args.m.models_to_save
    })


def show_batch_loss_bar(trainer, pbar):
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'batch_loss')
    pbar.attach(trainer, ['batch_loss'])


# def train(fn_training):
#     def do(cuda, data_args, model_args, epochs, n_batch, init_lr):
#         cuda = None if cuda < 0 else cuda
#         db = get_database()
#         args = prepare_args(
#             db, data_args, model_args, epochs, n_batch, init_lr)
#         fn_training(cuda, db, args)
#     return do


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


def attach_unsupervised_evaluator(trainer, evaluator, query_loader):
    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_per_epoch(engine):
        evaluator.run(query_loader)
        metrics = evaluator.state.metrics
        print("Evaluation Results f Epoch: {}  "
              "AUC area: {:.2f}, Top-1 accuracy: {:.2f}"
              .format(engine.state.epoch,
                      metrics['auc'], metrics['topk-acc']))
