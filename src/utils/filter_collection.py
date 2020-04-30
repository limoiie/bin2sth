def filter_dict(d: dict):
    for k in list(d.keys()):
        if not d[k]:
            del d[k]
    return d
