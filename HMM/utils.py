#!/usr/bin/python
#Author -- Stephen Rosen

from functools import wraps

def @memoize(fun):
    """
    Takes a function and creates a closure with it and a variable "_cache", the memoization cache.
    Looks up args,kwargs pairs in the cache, and when there is a hit, returns that hit.

    Intercepts the keyword argument '_clear_cache' and interprets it as a signal to dump the cache.
    Returns nothing in that case.
    """
    _cache = {}

    @wraps(fun)
    def wrapped(*args, **kwargs):
        if '_clear_cache' in kwargs and kwargs['_clear_cache']:
            _cache = {}
        else:
            if (args,kwargs) not in _cache:
                _cache[(args,kwargs)] = fun(*args,**kwargs)
            return _cache[(args,kwargs)]

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
