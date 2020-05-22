import copy
import json
from functools import wraps
from types import FunctionType
from typing import Type

_json_cls_key = '@'


class _Register:
    id_cls = {}
    cls_id = {}


def auto_json(cls: Type):
    assert cls.__name__ not in _Register.id_cls

    @wraps(cls, updated=[])
    class Cls(cls):
        def dict(self, with_cls=True):
            return AutoJson.to_dict(self, with_cls)
    Cls.__name__ = cls.__name__
    _Register.id_cls[Cls.__name__] = Cls
    _Register.cls_id[Cls] = Cls.__name__
    return Cls


class AutoJson(object):
    """ `json` is the object in dict type  """

    def __getitem__(self, item):
        return getattr(self, item)

    def dict(self):
        return AutoJson.to_dict(self)

    @staticmethod
    def dump(file, obj):
        """
        Dump obj into :param file in json
        :param file: file path to which to be dumped
        :param obj: the object to be dumped
        """
        with open(file, 'w') as f:
            json.dump(AutoJson.to_dict(obj), f)

    @staticmethod
    def load(file):
        """
        Load obj from :param file of obj in json
        :param file: file path where stores the obj in json
        :return: the loaded instance of the given class :param cls
        """
        with open(file, 'r') as f:
            js = json.load(f)
            return AutoJson.from_dict(js)

    @staticmethod
    def to_dict(obj, with_cls=True):
        if obj.__class__ in _Register.cls_id:
            dic = copy.deepcopy(obj.__dict__)
            if with_cls:
                dic[_json_cls_key] = obj.__class__.__name__
            return AutoJson.to_dict(dic, with_cls)

        cls = type(obj)
        if cls in [list, set]:
            return [AutoJson.to_dict(v, with_cls) for v in obj]
        if cls is dict:
            return {
                AutoJson.to_dict(k, with_cls): AutoJson.to_dict(v, with_cls)
                for k, v in obj.items() if not isinstance(v, FunctionType)
            }

        return obj

    @staticmethod
    def from_dict(data):
        typ = type(data)
        if isinstance(data, list):
            return [AutoJson.from_dict(v) for v in data]
        if isinstance(data, set):
            return {AutoJson.from_dict(v) for v in data}
        if isinstance(data, dict):
            new_data = typ()
            for k, v in data.items():
                new_data[AutoJson.from_dict(k)] = AutoJson.from_dict(v)
            data = new_data
            if _json_cls_key in data:
                if data[_json_cls_key] not in _Register.id_cls:
                    raise ValueError(F'Not registered: {data[_json_cls_key]}')
                Cls = _Register.id_cls[data[_json_cls_key]]
                del data[_json_cls_key]
                obj = Cls()
                obj.__dict__.update(data)
                return obj
        return data
