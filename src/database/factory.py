import json

from src.database.beans.check_point import CheckPointEntry
from src.database.repository import Repository
from src.models.builder import ModelKeeper, ModelBuilder
from src.utils.auto_json import AutoJson
from src.utils.logger import get_logger

logger = get_logger('factory')


class Factory:
    __cache = {}

    @staticmethod
    def load_instance(Cls, args):
        args_recipe = AutoJson.to_dict(args)
        logger.info(f'Start loading {Cls} with \n\t {args_recipe}...')

        key = hash_args(args)
        if key in Factory.__cache:
            logger.info(f'Find in cache for args: {args}, use this')
            model = Factory.__cache[key]
        elif isinstance(args, CheckPointEntry):
            model_state = Repository.find_checkpoint_entry(args)
            keeper = ModelKeeper.instance(Cls)
            model = keeper.from_state(model_state)
        else:
            Builder = ModelBuilder.clazz(Cls)
            _load_dependency(Builder, args)
            model = Builder(**args).build()

        # embed the recipe into the model in case the model depends on
        # some other model which may not exist at the time of saving.
        # NOTE: now i save the model directly instead of saving the recipe
        # since it is not enough to restore some model from only recipe,
        # there may still need a training process
        # setattr(model, '__recipe__', args_recipe)
        # cache the model incase of redundant creation
        Factory.__cache[key] = model
        logger.info(f'Finish loading {Cls}')
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


def _load_dependency(Builder: type, args):
    if hasattr(Builder, '__annotations__'):
        for field, FieldCls in Builder.__annotations__.items():
            if field not in args:
                raise ValueError(f'Failed to prepare param {field} for '
                                 f'builder {Builder}!')
            # load dependency
            args[field] = Factory.load_instance(FieldCls, args[field])
