import copy


def json_update(src, delta):
    src = copy.deepcopy(src)
    src.update(delta)
    return src
