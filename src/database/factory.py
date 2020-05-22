import json

from bson import ObjectId
from gridfs import GridFS

from src.database.beans.check_point import CheckPoint
from src.database.dao import Dao, to_filter
from src.database.database import get_database
from src.ida.code_elements import Program
from src.preprocesses.builder import ModelKeeper, ModelBuilder
from src.training.args.train_args import TrainArgs, BinArgs
from src.utils.auto_json import AutoJson
from src.utils.logger import get_logger

logger = get_logger('factory')


class Repository:
    db = get_database()
    fs = GridFS(db)

    @staticmethod
    def dao(Cls):
        return Dao.instance(Cls, Repository.db, Repository.fs)

    @staticmethod
    def find_model(args):
        id_, mod = args['training_id'], args['module']
        dao = Repository.dao(CheckPoint)
        cp = dao.find_one({'training_id': ObjectId(id_)})
        if cp and mod in cp.checkpoints:
            return cp.checkpoints[mod]
        raise RuntimeError(f'Not found such checkpoint: '
                           f'training_id: {id_}, module: {mod}')

    @staticmethod
    def save_model(cp):
        dao = Repository.dao(CheckPoint)
        if not dao.find_one({'training_id': cp.training_id}):
            return dao.store_one(cp)
        raise RuntimeError(f'There already is a checkpoint with '
                           f'training_id: {cp.training_id}!')

    @staticmethod
    def find_prog(args: BinArgs):
        prog_dao = Repository.dao(Program)
        for (prog, prog_ver), (cc, cc_ver), arch, opt, obf in args.joint():
            prog = Program(prog=prog, prog_ver=prog_ver, cc=cc, cc_ver=cc_ver,
                           arch=arch, opt=opt, obf=obf)
            ps = prog_dao.find(to_filter(prog, with_cls=False))
            if ps is None:
                raise ValueError(f'Not found such Program info in database: '
                                 f'prog={prog}, prog_ver={prog_ver}, '
                                 f'cc={cc}, cc_ver={cc_ver}, '
                                 f'arch={arch}, opt={opt}, obf={obf}')
            for p in ps:
                yield p

    @staticmethod
    def find_or_store_training(args: TrainArgs):
        filtor = {'hash_id': args.hash_id}
        dao = Repository.dao(TrainArgs)
        return dao.find_or_store(filtor, args)


class Factory:
    __cache = {}

    @staticmethod
    def load_model(Cls, args):
        logger.info(f'Start loading {Cls}...')
        key = hash_args(args)
        if key in Factory.__cache:
            logger.info(f'Find in cache for args: {args}, use this')
            return Factory.__cache[key]

        if is_checkpoint_args(args):
            model_state = Repository.find_model(args)
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
    def save_model(training_id, named_models):
        named_states = dict()
        for name, model in named_models.items():
            Cls = type(model)
            logger.info(f'Getting state from <{name}> of {Cls}...')
            keeper: ModelKeeper = ModelKeeper.instance(Cls)
            named_states[name] = keeper.state(model)
        logger.info(f'Saving checkpoints...')
        Repository.save_model(CheckPoint(training_id, named_states))
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
            args[field] = Factory.load_model(FieldCls, args[field])
