import json
import tempfile
import os

import torch
from gridfs import GridFS, GridIn
from pymongo.collection import Collection
from pymongo.database import Database

from src.utils.collection_op import strip_dict
from src.utils.logger import get_logger

logger = get_logger('DAO')
encoding = 'utf-8'


def to_filter(bean, with_cls=True):
    return strip_dict(bean.dict(with_cls))


class Adapter:
    """
    Adapter for beans to convert custom objects into and from what
    language the db speaks
    """
    __register = {}
    __register_by_name = {}

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
            Adapter.__register_by_name[raw_cls.__name__] = cls()
            # **no-return** means you cannot create this cls on your own.
            # you should never create it manually. instead, to use the
            # functions of Adapter, you should call Adapter.wrap_(dao)(...)
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


class Dao:
    __register = {}
    __register_by_name = {}

    def __init__(self, db: Database, fs: GridFS, col: Collection, ele_cls):
        self.db, self.fs, self.col = db, fs, col
        self.EleCls = ele_cls

    @staticmethod
    def register(raw_cls):
        def f(cls):
            if raw_cls in Dao.__register:
                raise ValueError(F'{raw_cls} has been registered already!')
            Dao.__register[raw_cls] = cls
            Dao.__register_by_name[raw_cls.__name__] = cls
            logger.debug(f'Have registered {raw_cls} into Dao')
            # **no-return** means you cannot create this cls on your own.
            # the only way to get dao is through the :func instance.
        return f

    @staticmethod
    def instance(bean_cls, *args, **kwargs):
        if type(bean_cls) is str:
            assert bean_cls in Dao.__register_by_name, \
                f'Bean Class {bean_cls} is not registered!'
            return Dao.__register_by_name[bean_cls](*args, **kwargs)
        assert bean_cls in Dao.__register, \
            f'Bean Class {bean_cls} is not registered!'
        return Dao.__register[bean_cls](*args, **kwargs)

    def put_into_fs(self, obj):
        return self.fs.put(json.dumps(obj), encoding=encoding)

    def get_from_fs(self, f_id):
        return json.loads(self.fs.get(f_id).read().decode(encoding=encoding))

    def find(self, filtor):
        return map(Adapter.unwrap_(self), self.col.find(filtor))

    def find_one(self, filtor):
        o = self.col.find_one(filtor)
        return Adapter.unwrap_(self)(o) if o else None

    def id(self, filtor):
        return map(lambda o: o['_id'], self.col.find(filtor))

    def id_one(self, filtor):
        o = self.col.find_one(filtor)
        return o['_id'] if o else None

    def store(self, objs):
        return self.col.insert_many(map(Adapter.wrap_(self), objs))

    def store_one(self, obj):
        return self.col.insert_one(Adapter.wrap_(self)(obj))

    def find_or_store(self, key, obj):
        o = self.find_one(key)
        if not o:
            res = self.store_one(obj)
            if not res or not res.inserted_id:
                raise IOError(F'Failed to store the object: {obj}!')
            o = self.find_one({'_id': res.inserted_id})
        return o

    def count(self, filtor=None):
        return self.col.count_documents(filtor or {})

    def delete(self, filtor):
        for dic in self.col.find(filtor):
            logger.info(f'Deleting checkpoint {dic["_id"]}')
            self._cascade_delete(dic)
            res = self.col.delete_one(dic)
            logger.info(f'Deleted with result: {res}')

    def delete_one(self, filtor):
        dic = self.col.delete_one(filtor)
        logger.info(f'Deleting checkpoint {dic["_id"]}')
        self._cascade_delete(dic)
        res = self.col.delete_one(dic)
        logger.info(f'Deleted with result: {res}')

    def delete_file(self, file_id):
        logger.info(f'Deleting file {file_id}')
        res = self.fs.delete(file_id)
        logger.info(f'Deleted with result: {res}')
        return res

    def _cascade_delete(self, dic):
        pass
