import json

from gridfs import GridFS
from pymongo.collection import Collection
from pymongo.database import Database

encoding = 'utf-8'


class Dao:
    def __init__(self, db: Database, fs: GridFS, col: Collection, ele_cls):
        self.db, self.fs, self.col = db, fs, col
        self.EleCls = ele_cls

    def put_into_fs(self, obj):
        return self.fs.put(json.dumps(obj), encoding=encoding)

    def get_from_fs(self, f_id):
        return json.loads(self.fs.get(f_id).read().decode(encoding=encoding))

    def find(self, filtor):
        return map(Adapter.unwrap_(self), self.col.find(filtor))

    def find_one(self, filtor):
        return Adapter.unwrap_(self)(self.col.find_one(filtor))

    def id(self, filtor):
        return map(lambda o: o['_id'], self.col.find(filtor))

    def id_one(self, filtor):
        return self.col.find_one(filtor)['_id']

    def store(self, objs):
        return self.col.insert_many(map(Adapter.wrap_(self), objs))

    def store_one(self, obj):
        return self.col.insert_one(Adapter.wrap_(self)(obj))


class Adapter:
    __register = {}

    def wrap(self, dao, obj):
        raise NotImplementedError()

    def unwrap(self, dao, js):
        raise NotImplementedError()

    @staticmethod
    def register(raw_cls):
        def f(cls):
            if raw_cls in Adapter.__register:
                raise ValueError(F'{raw_cls} has been registered already!')
            Adapter.__register[raw_cls] = cls()
        return f

    @staticmethod
    def wrap_(dao):
        def f(obj):
            Cls = type(obj)
            if Cls not in Adapter.__register:
                raise ValueError(F'Failed to wrap: {Cls} is not registered!')
            adapter = Adapter.__register[Cls]
            return adapter.wrap(dao, obj)
        return f

    @staticmethod
    def unwrap_(dao):
        def f(js):
            Cls = dao.EleCls
            if Cls not in Adapter.__register:
                raise ValueError(F'Failed to unwrap: {Cls} is not registered!')
            adapter = Adapter.__register[Cls]
            return adapter.unwrap(dao, js)
        return f
