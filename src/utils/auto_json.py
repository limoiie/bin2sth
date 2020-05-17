import copy
import json


_json_cls_key = '@'


class _Register:
    id_cls = {}
    cls_id = {}


def auto_json(cls):
    assert cls.__name__ not in _Register.id_cls
    _Register.id_cls[cls.__name__] = cls
    _Register.cls_id[cls] = cls.__name__
    return cls


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
    def to_dict(obj):
        if obj.__class__ in _Register.cls_id:
            dic = copy.deepcopy(obj.__dict__)
            dic[_json_cls_key] = obj.__class__.__name__
            return AutoJson.to_dict(dic)

        cls = type(obj)
        if cls in [list, set]:
            return [AutoJson.to_dict(v) for v in obj]
        if cls is dict:
            return {
                AutoJson.to_dict(k): AutoJson.to_dict(v)
                for k, v in obj.items()
            }

        return obj

    @staticmethod
    def from_dict(data):
        typ = type(data)
        if typ is list:
            return [AutoJson.from_dict(v) for v in data]
        if typ is set:
            return {AutoJson.from_dict(v) for v in data}
        if typ is dict:
            dic = {
                AutoJson.from_dict(k): AutoJson.from_dict(v)
                for k, v in data.items()
            }
            if _json_cls_key in dic:
                if dic[_json_cls_key] not in _Register.id_cls:
                    raise ValueError(F'Not registered: {dic[_json_cls_key]}')
                Cls = _Register.id_cls[dic[_json_cls_key]]
                del dic[_json_cls_key]
                obj = Cls()
                obj.__dict__.update(dic)
                return obj
        return data
