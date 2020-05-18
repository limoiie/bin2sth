from gridfs import GridFS
from pymongo.database import Database

from src.database.dao import Dao, Adapter
from src.training.args.train_args import TrainArgs
from src.utils.auto_json import AutoJson


@Adapter.register(TrainArgs)
class TrainArgsAdapter(Adapter):
    def wrap(self, dao, obj: TrainArgs):
        return AutoJson.to_dict(obj)

    def unwrap(self, dao, js) -> TrainArgs:
        return AutoJson.from_dict(js)


@Dao.register(TrainArgs)
class TrainArgDAO(Dao):
    def __init__(self, db: Database, fs: GridFS):
        super().__init__(db, fs, db.training_process, TrainArgs)
