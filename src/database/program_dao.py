from gridfs import GridFS

from src.database.dao import Dao
from src.ida.as_json import AsJson
from src.ida.code_elements import Program
from src.training.train_args import BinArgs
from src.utils.collection_op import strip_dict


class ProgramDAO(Dao):
    def __init__(self, db, fs: GridFS):
        super().__init__(db, fs)

    def store(self, prog: Program):
        """
        Store program into db. The functions and callgraphs
        therein should be stored into GridFS separatively
        """
        self.db.binaries.insert_one(self.__prog2dict(prog))

    def find(self, **args):
        the_filter = make_prog_filter(**args)
        prog = self.db.binaries.find(the_filter)
        return map(self.__dict2prog, prog)

    def find_one(self, **args):
        the_filter = make_prog_filter(**args)
        prog = self.db.binaries.find_one(the_filter)
        return self.__dict2prog(prog)

    def __prog2dict(self, prog):
        prog = prog.dict()
        prog['funcs'] = self._put_into_fs(prog['funcs'])
        prog['cg'] = self._put_into_fs(prog['cg'])
        return prog

    def __dict2prog(self, prog):
        prog['funcs'] = self._get_from_fs(prog['funcs'])
        prog['cg'] = self._get_from_fs(prog['cg'])
        return AsJson.from_dict(Program, prog)


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
        ps = prog_dao.find(prog=prog, prog_ver=prog_ver,
                           cc=cc, cc_ver=cc_ver,
                           arch=arch, opt=opt, obf=obf)
        if ps is None:
            raise ValueError(f'No such Program info in the database: \
                prog={prog}, prog_ver={prog_ver}, cc={cc}, cc_ver={cc_ver}, \
                arch={arch}, opt={opt}, obf={obf}')
        for p in ps:
            yield p
