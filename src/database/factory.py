import json

from src.database.beans.check_point import CheckPoint, CheckPointEntry
from src.database.repository import Repository
from src.preprocesses.builder import ModelKeeper, ModelBuilder
from src.utils.auto_json import AutoJson
from src.utils.logger import get_logger

logger = get_logger('factory')


class Factory:
    __cache = {}

    @staticmethod
    def load_instance(Cls, args):
        logger.info(f'Start loading {Cls}...')
        key = hash_args(args)
        if key in Factory.__cache:
            logger.info(f'Find in cache for args: {args}, use this')
            return Factory.__cache[key]

        if isinstance(args, CheckPointEntry):
            model_state = Repository.find_checkpoint_entry(args)
            keeper = ModelKeeper.instance(Cls)
            model = keeper.from_state(model_state)
        else:
            Builder = ModelBuilder.clazz(Cls)
            if Builder:
                # build with recipe args
                _load_dependency(Builder, args)
                model = Builder(**args).build()
            else:
                _load_dependency(Cls, args)
                model = Cls(**args)

        logger.info(f'Finish loading {Cls}')
        Factory.__cache[key] = model
        return model

    @staticmethod
    def save_instance(training_id, named_models):
        for name, model in named_models.items():
            Cls = type(model)
            logger.info(f'Getting state from <{name}> of {Cls}...')
            keeper: ModelKeeper = ModelKeeper.instance(Cls)

            state = keeper.state(model)
            logger.info(f'Saving checkpoints...')
            Repository.save_checkpoint_entry(
                CheckPointEntry(training_id, name, state))
        logger.info(f'Saved')


def hash_args(obj):
    dic = AutoJson.to_dict(obj)
    return hash(json.dumps(dic, sort_keys=True))


def is_checkpoint_args(args):
    return 'training_id' in args and 'module' in args


def _load_dependency(Builder: type, args):
    if hasattr(Builder, '__annotations__'):
        for field, FieldCls in Builder.__annotations__.items():
            if field not in args:
                raise ValueError(f'Failed to prepare param {field} for '
                                 f'builder {Builder}!')
            # load dependency
            args[field] = Factory.load_instance(FieldCls, args[field])
