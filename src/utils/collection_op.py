def strip_dict(d: dict):
    for k in list(d.keys()):
        if not d[k]:
            del d[k]
    return d


def is_iterable(obj):
    # noinspection PyBroadException
    try:
        iter(obj)
    except Exception:
        return False
    return True
