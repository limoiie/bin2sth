from gridfs import GridFS

from src.database.beans.check_point import CheckPoint, CheckPointEntry
from src.database.dao import Adapter, Dao
from src.utils.auto_json import AutoJson
from src.utils.logger import get_logger

logger = get_logger('checkpoint dao')


@Adapter.register(CheckPoint)
class CheckPointAdapter(Adapter):
    def wrap(self, dao, cp: CheckPoint):
        dic = AutoJson.to_dict(cp)
        for k, model in dic['checkpoints'].items():
            logger.info(f'dumping checkpoint: {k}')
            dic['checkpoints'][k] = dao.put_into_fs_(model)
        return dic

    def unwrap(self, dao, dic) -> CheckPoint:
        for k, model in dic['checkpoints'].items():
            logger.info(f'loading checkpoint: {k}')
            dic['checkpoints'][k] = dao.get_from_fs_(model)
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


@Adapter.register(CheckPointEntry)
class CheckPointEntryAdapter(Adapter):
    def wrap(self, dao, cpe: CheckPointEntry):
        dic = AutoJson.to_dict(cpe)
        dic['data'] = dao.put_into_fs_(cpe.data)
        logger.info(f'Dump checkpoint entry as {dic}')
        return dic

    def unwrap(self, dao, dic):
        logger.info(f'Load checkpoint entry as {dic}')
        dic['data'] = dao.get_from_fs_(dic['data'])
        return AutoJson.from_dict(dic)


@Dao.register(CheckPointEntry)
class CheckPointEntryDAO(Dao):
    def __init__(self, db, fs: GridFS):
        super().__init__(db, fs, db.checkpointentry, CheckPointEntry)

    def _cascade_delete(self, dic):
        name, data = dic['name'], dic['data']
        logger.info(f'Deleting attached file of model `{name}` '
                    f'while deleting checkpoint')
        self.delete_file(data)
