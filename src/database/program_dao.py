from gridfs import GridFS

from src.database.dao import Dao, Adapter
from src.ida.as_json import AsJson
from src.ida.code_elements import Program
from src.training.train_args import BinArgs
from src.utils.collection_op import strip_dict


@Adapter.register(Program)
class ProgramAdapter(Adapter):
    def wrap(self, dao, prog: Program):
        dic = AsJson.to_dict(prog)
        dic['funcs'] = dao.put_into_fs(dic['funcs'])
        dic['cg'] = dao.put_into_fs(dic['cg'])
        return dic

    def unwrap(self, dao, dic) -> Program:
        dic['funcs'] = dao.get_from_fs(dic['funcs'])
        dic['cg'] = dao.get_from_fs(dic['cg'])
        return AsJson.from_dict(Program, dic)


class ProgramDAO(Dao):
    def __init__(self, db, fs: GridFS):
        super().__init__(db, fs, db.binaries, Program)


def make_prog_filter(prog=None, prog_ver=None, cc=None, cc_ver=None, arch=None,
                     opt=None, obf=None):
    prog = Program(prog, prog_ver, cc, cc_ver, arch, opt, obf)
    the_filter = prog.dict()
    return strip_dict(the_filter)


def load_progs_jointly(db, args: BinArgs):
    """
    Joint product the args to form a set of binaries and then load the
    info of these binaries into a list.
    """
    prog_dao = ProgramDAO(db, GridFS(db))
    for (prog, prog_ver), (cc, cc_ver), arch, opt, obf in args.joint():
        filtor = make_prog_filter(
            prog=prog, prog_ver=prog_ver, cc=cc, cc_ver=cc_ver,
            arch=arch, opt=opt, obf=obf)
        ps = prog_dao.find(filtor)
        if ps is None:
            raise ValueError(f'No such Program info in the database: \
                prog={prog}, prog_ver={prog_ver}, cc={cc}, cc_ver={cc_ver}, \
                arch={arch}, opt={opt}, obf={obf}')
        for p in ps:
            yield p
