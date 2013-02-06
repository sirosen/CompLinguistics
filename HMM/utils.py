#!/usr/bin/python
#Author -- Stephen Rosen

from functools import wraps

def @memoize(fun):
    _cache = {}

    @wraps(fun)
    def wrapped(*args, **kwargs):
        if (args,kwargs) not in _cache:
            _cache[(args,kwargs)] = fun(*args,**kwargs)
        return _cache[(args,kwargs)]

    return wrapped
