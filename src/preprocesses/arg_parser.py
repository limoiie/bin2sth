from collections import OrderedDict

from src.utils.json_utils import load_json_file, json_update

_reference_symb = '&'
_overwrite_symb = '!'
_preserved_symb = '*'


class ArgsParser:
    def __init__(self):
        self.data = dict()
        self.root = OrderedDict()
        self.namespace = None
        self.preparing = set()

    def parser(self, args_file, namespace=None):
        self.data = load_json_file(args_file)
        self.namespace = namespace
        self._resolve_symbols_and_links()

    def make(self):
        return self.root

    def _resolve_symbols_and_links(self):
        for k, v in self.data.items():
            self.__guard_visit(k, v)

    def __resolve(self, data):
        typ = type(data)
        if typ is list:
            return [self.__resolve(item) for item in data]
        if typ is dict:
            new_data = dict()
            for key, val in data.items():
                new_data[key] = self.__resolve_link(val) or \
                                self.__resolve(val)
            if _overwrite_symb in data:
                base_data = new_data[_overwrite_symb]
                del new_data[_overwrite_symb]
                new_data = json_update(base_data, new_data)
            return new_data
        return data

    def __resolve_link(self, val):
        if isinstance(val, str) and val.startswith(_reference_symb):
            link = val[len(_reference_symb):]
            if not self._in_root(link):
                if not self._in_data(link):
                    raise ValueError(f'Invalid link: {val}! You can only link '
                                     f'to the field at the root of the dict.')
                self.__guard_visit(link, self.data[link])
            return self._get_from_root(link)
        return None

    def __guard_visit(self, k, v):
        if k in self.preparing:
            raise ValueError('Circling dependency while parsing args!')
        if not self._in_root(k):
            self.preparing.add(k)
            self._set_into_root(k, self.__resolve(v))
            self.preparing.remove(k)

    def _namespace_key(self, k):
        return f'{self.namespace}.{k}' if self.namespace else k

    def _in_root(self, k):
        return self._namespace_key(k) in self.root

    def _in_data(self, k):
        return k in self.data

    def _set_into_root(self, k, v):
        self.root[self._namespace_key(k)] = v

    def _get_from_root(self, k):
        return self.root[self._namespace_key(k)]


def is_preserved_key(k):
    return k.startswith(_preserved_symb)
