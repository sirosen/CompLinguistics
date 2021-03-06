#!/usr/bin/python
#Author -- Stephen Rosen

from functools import wraps

def memoize(fun):
    """
    Takes a function and creates a closure with it and a variable "_cache", the memoization cache.
    Looks up args,kwargs pairs in the cache, and when there is a hit, returns that hit.
    """
    _cache = {}

    @wraps(fun)
    def wrapped(*args, **kwargs):
        if args not in _cache:
            _cache[args] = fun(*args,**kwargs)
        return _cache[args]

    return wrapped

def normalize(name_to_value):
    """
    Normalizes a probability distribution to 1.

    Args:
        name_to_value
        A dictionary mapping names onto probability values.
        We modify the dictionary in place, so that the original is destroyed.
    """
    ntv = name_to_value
    tot = sum(ntv[n] for n in ntv)
    assert (tot > 0)

    for n in ntv:
        ntv[n] = ntv[n]/tot
