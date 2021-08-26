import functools
import operator


def flatten_array(x):
    return functools.reduce(operator.iconcat, x, [])