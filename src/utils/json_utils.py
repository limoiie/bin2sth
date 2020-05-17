import copy
import json


def json_update(src, delta):
    src = copy.deepcopy(src)
    src.update(delta)
    return src


def load_json_file(file):
    with open(file, 'r') as f:
        return json.load(f)
