import copy
import json


def json_update(src, delta):
    src = copy.deepcopy(src)
    src.update(delta)
    return src


def obj_update(src, tgt):
    tgt.__dict__ = json_update(src.__dict__, tgt.__dict__)
    return tgt


def load_json_file(file):
    with open(file, 'r') as f:
        return json.load(f)
