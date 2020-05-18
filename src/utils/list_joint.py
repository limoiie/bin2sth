# todo move into collection_op
def joint(lhs, rhs, c=False):
    """ Joint-product the two given lists """
    if not lhs:
        lhs = [None]
    if not rhs:
        rhs = [None]

    rhs = list(rhs)  # in case that rhs is an iterator
    for l in lhs:
        for r in rhs:
            yield [l, r] if c else l + [r]


def joint_list(l):
    """ Reduce the list by applying joint """
    a, c = l[0], True
    for b in l[1:]:
        a = joint(a, b, c)
        c = False
    return a


def flat(lhs, rhs):
    """ Repeat the elem in left up to the size in right """
    for elem, vers in zip(lhs, rhs):
        for ver in vers:
            yield elem, ver
