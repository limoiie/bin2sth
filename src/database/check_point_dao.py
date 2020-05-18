from gridfs import GridFS

from src.database.beans.check_point import CheckPoint
from src.database.dao import Adapter, Dao
from src.utils.auto_json import AutoJson


@Adapter.register(CheckPoint)
class CheckPointAdapter(Adapter):
    def wrap(self, dao, cp: CheckPoint):
        dic = AutoJson.to_dict(cp)
        for k, model in dic['checkpoints']:
            # todo: current put_into_fs will convert arguments into json str
            dic['checkpoints'][k] = dao.put_into_fs(model)
        return dic

    def unwrap(self, dao, dic) -> CheckPoint:
        for k, model in dic['checkpoints']:
            # todo: current get_from_fs will decode json string
            dic['checkpoints'][k] = dao.get_from_fs(model)
        return AutoJson.from_dict(dic)


@Dao.register(CheckPoint)
class CheckPointDAO(Dao):
    def __init__(self, db, fs: GridFS):
        super().__init__(db, fs, db.checkpoints, CheckPoint)

    def _cascade_delete(self, dic):
        for name, model_file in dic['checkpoints'].items():
            logger.info(f'Deleting attached file of model `{name}` '
                        f'while deleting checkpoint')
            self.delete_file(model_file)
