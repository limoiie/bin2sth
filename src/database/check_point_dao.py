from gridfs import GridFS

from src.database.beans.check_point import CheckPointEntry
from src.database.dao import Adapter, Dao
from src.utils.auto_json import AutoJson
from src.utils.logger import get_logger

logger = get_logger('checkpoint dao')


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
