from gridfs import GridFS

from src.database.dao import Dao, Adapter
from src.ida.as_json import AsJson
from src.ida.code_elements import Program


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


@Dao.register(Program)
class ProgramDAO(Dao):
    def __init__(self, db, fs: GridFS):
        super().__init__(db, fs, db.binaries, Program)

    def _cascade_delete(self, dic):
        self.delete_file(dic['funcs'])
        self.delete_file(dic['cg'])
