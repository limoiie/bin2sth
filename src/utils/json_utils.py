import copy
import json

from src.utils.collection_op import strip_dict


def json_update(src, delta):
    src = copy.deepcopy(src)
    delta = strip_dict(delta)
    src.update(delta)
    return src


def obj_update(src, tgt):
    tgt.__dict__ = json_update(src.__dict__, tgt.__dict__)
    return tgt


def load_json_file(file):
    with open(file, 'r') as f:
        return json.load(f)
