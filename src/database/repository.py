from bson import ObjectId
from gridfs import GridFS

from src.database.beans.check_point import CheckPointEntry
from src.database.dao import Dao, to_filter
from src.database.database import get_database
from src.ida.code_elements import Program
from src.training.args.train_args import BinArgs, TrainArgs
from src.utils.auto_json import AutoJson
from src.utils.collection_op import strip_dict


class Repository:
    db = get_database()
    fs = GridFS(db)

    @staticmethod
    def dao(Cls):
        return Dao.instance(Cls, Repository.db, Repository.fs)

    @staticmethod
    def find_checkpoint_entry(filtor):
        if isinstance(filtor, CheckPointEntry):
            filtor = strip_dict(AutoJson.to_dict(filtor))
        if isinstance(filtor['training_id'], str):
            filtor['training_id'] = ObjectId(filtor['training_id'])
        dao = Repository.dao(CheckPointEntry)
        cp: CheckPointEntry = dao.find_one(filtor)
        if cp:
            return cp.data
        raise RuntimeError(f'Not found checkpoint: {filtor}')

    @staticmethod
    def save_checkpoint_entry(cp: CheckPointEntry):
        filtor = AutoJson.to_dict(cp)
        filtor['data'] = None
        filtor = strip_dict(filtor)

        dao = Repository.dao(CheckPointEntry)
        if not dao.find_one(filtor):
            return dao.store_one(cp)
        raise RuntimeError(f'There already is a checkpoint entry for:'
                           f'{filtor}')

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
