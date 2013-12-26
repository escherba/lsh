__author__ = 'escherba'

from itertools import izip


''' Helper functions
'''


def first(it):
    """return first element of an iterable"""
    return it[0]


def second(it):
    """return second element of an iterable"""
    return it[1]


def last(it):
    """return last element of an iterable"""
    return it[-1]


def gapply(n, func, *args, **kwargs):
    """Apply a generating function n times to the argument list"""
    for _ in xrange(n):
        yield func(*args, **kwargs)


def lapply(*args, **kwargs):
    """Same as gapply except treturn a list"""
    return list(gapply(*args, **kwargs))


def dot(u, v):
    """Return dot product of two vectors"""
    return sum(ui * vi for ui, vi in izip(u, v))
