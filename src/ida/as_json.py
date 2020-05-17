import inspect
import json


class AsJson(object):
    """ `json` is the object in dict type  """

    def __getitem__(self, item):
        return getattr(self, item)

    def dict(self, _=True):
        return AsJson.to_dict(self, _)

    @staticmethod
    def dump(file, obj):
        """
        Dump obj into :param file in json
        :param file: file path to which to be dumped
        :param obj: the object to be dumped
        """
        with open(file, 'w') as f:
            json.dump(AsJson.to_dict(obj), f)

    @staticmethod
    def load(file, cls):
        """
        Load obj from :param file of obj in json
        :param file: file path where stores the obj in json
        :param cls: the target class
        :return: the loaded instance of the given class :param cls
        """
        with open(file, 'r') as f:
            js = json.load(f)
            return AsJson.from_dict(cls, js)

    @staticmethod
    def to_dict(obj, _=True):
        """
        :param obj: object to be dicted
        :param _ compatible with AutoJson.to_dict
        """
        if issubclass(obj.__class__, AsJson):
            return AsJson.to_dict(obj.__dict__)

        cls = type(obj)
        if cls in [list, set]:
            return [AsJson.to_dict(v) for v in obj]
        if cls is dict:
            return {
                AsJson.to_dict(k): AsJson.to_dict(v)
                for k, v in obj.items()
            }

        return obj

    @staticmethod
    def from_dict(cls, dic):
        if inspect.isclass(cls) and issubclass(cls, AsJson):
            obj = cls()
            obj.__dict__.update(dic)
            if hasattr(cls, '__annotations__'):
                for k, field_cls in cls.__annotations__.items():
                    obj.__dict__[k] = AsJson.from_dict(field_cls, dic[k])
            return obj

        # when cls is an instance of typing._GenericAlias
        if hasattr(cls, '__origin__'):
            ori = cls.__origin__
            if ori in [list, set]:
                inner_cls, = cls.__args__
                return [AsJson.from_dict(inner_cls, v) for v in dic]
            if ori in [dict]:
                k_cls, v_cls = cls.__args__
                return {
                    AsJson.from_dict(k_cls, k): AsJson.from_dict(v_cls, v)
                    for k, v in dic.items()
                }

        return dic
