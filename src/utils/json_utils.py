import copy


def json_update(src, delta):
    src = copy.copy(src)
    src.update(delta)
    return src
